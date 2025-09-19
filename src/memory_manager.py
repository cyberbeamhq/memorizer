import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dateutil.relativedelta import relativedelta

from . import compression_agent, db, retrieval, vector_db
from .cache import get_cache_manager, invalidate_user_cache
from .validation import validate_memory_input, validate_query_input

logger = logging.getLogger(__name__)

# Simple in-process embedding queue & worker for demo purposes.
# In production, replace with Celery/Kafka/Cloud Tasks.
_embedding_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
_stop_worker = threading.Event()

# Thread pool for vector operations to prevent thread exhaustion
_vector_thread_pool = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="vector_worker"
)


def _send_to_monitoring(
    level: str, message: str, exc: Optional[Exception] = None
) -> None:
    """Stub for sending critical errors to monitoring (Sentry/Prometheus).
    Replace with real integration in production."""
    logger.warning(f"MONITORING[{level}]: {message}")


def _embedding_worker() -> None:
    while not _stop_worker.is_set():
        try:
            item = _embedding_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            vector_db.insert_embedding(
                item["memory_id"],
                item["user_id"],
                item["content"],
                item.get("metadata", {}),
            )
            logger.debug(f"Inserted embedding for memory {item['memory_id']}")
        except Exception as e:
            logger.exception(
                f"Background embedding insert failed for {item['memory_id']}: {e}"
            )
            _send_to_monitoring(
                "error", f"Embedding insert failed for {item['memory_id']}", e
            )
            # Persist to retry queue/table so another process can pick up later.
            try:
                db.enqueue_embedding_retry(
                    item["memory_id"],
                    item["user_id"],
                    item["content"],
                    item.get("metadata", {}),
                )
            except Exception as inner:
                logger.exception(
                    f"Failed to enqueue embedding retry for {item['memory_id']}: {inner}"
                )
        finally:
            _embedding_queue.task_done()


# Start background worker thread (daemon so it won't block shutdown)
_worker_thread = threading.Thread(
    target=_embedding_worker, daemon=True, name="embedding-worker"
)
_worker_thread.start()


# ---------------------------
# Helpers
# ---------------------------

MAX_CONTENT_LENGTH = 32_000  # characters
ALLOWED_METADATA_KEYS = {"source", "channel", "created_by", "tags"}


def sanitize_content(content: str) -> str:
    if not content:
        return ""
    # Truncate to maximum allowed length to avoid huge embeddings.
    if len(content) > MAX_CONTENT_LENGTH:
        logger.debug(
            "Content length exceeds MAX_CONTENT_LENGTH; truncating before storing/embedding"
        )
        return content[:MAX_CONTENT_LENGTH]
    return content


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not metadata:
        return {}
    sanitized = {k: metadata[k] for k in metadata if k in ALLOWED_METADATA_KEYS}
    # Ensure small, serializable values
    for k, v in list(sanitized.items()):
        if isinstance(v, (list, dict)) and len(str(v)) > 1000:
            sanitized[k] = str(v)[:1000]
    return sanitized


def utc_now() -> datetime:
    return datetime.utcnow()


# ---------------------------
# Add New Session
# ---------------------------


def add_session(
    user_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add a new interaction (goes into very_new tier).
    Push embedding insertion to a background worker and persist a retry record on failure.
    """
    try:
        # Validate input parameters
        validated_data = validate_memory_input(user_id, content, metadata)
        user_id = validated_data["user_id"]
        content = validated_data["content"]
        metadata = validated_data["metadata"]

        # Use UTC created_at if db.insert_session accepts it; otherwise db should fill created_at in UTC.
        memory_id = db.insert_session(user_id, content, metadata or {}, tier="very_new")

        # Invalidate user cache since new memory was added
        invalidate_user_cache(user_id)

        # Enqueue embedding insertion to background worker (non-blocking)
        try:
            _embedding_queue.put_nowait(
                {
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "content": content,
                    "metadata": metadata,
                }
            )
        except Exception as e:
            logger.exception(f"Failed to enqueue embedding for memory {memory_id}: {e}")
            # Persist to retry queue immediately
            try:
                db.enqueue_embedding_retry(memory_id, user_id, content, metadata)
            except Exception as inner:
                logger.exception(
                    f"Failed to persist embedding retry for {memory_id}: {inner}"
                )
                _send_to_monitoring(
                    "error",
                    f"Critical: could not persist embedding retry for {memory_id}",
                    inner,
                )

        return memory_id

    except Exception as e:
        logger.exception(f"Failed to add session for user {user_id}: {e}")
        _send_to_monitoring("error", f"add_session failed for user {user_id}", e)
        raise


# ---------------------------
# Move Memories Between Tiers
# ---------------------------


def move_memory_between_tiers(
    user_id: str,
    very_new_limit: int = 20,
    mid_term_limit: int = 200,
    very_new_days: int = 10,
    mid_term_months: int = 12,
) -> Dict[str, List[str]]:
    """
    Periodic job:
      - very_new -> mid_term after limit or time
      - mid_term -> long_term after limit or time

    Uses enumerate() (O(n)) and relativedelta for month arithmetic.
    Uses transactional semantics where available and only deletes after successful insert.
    """
    moved = {"to_mid_term": [], "to_long_term": []}

    try:
        # Step 1: Move old very_new memories to mid_term
        cutoff_date = utc_now() - timedelta(days=very_new_days)
        # Ensure db.fetch_memories returns ordered results (most recent first)
        very_new_memories = db.fetch_memories(
            user_id, "very_new", limit=None, order_by="created_at DESC"
        )

        memories_to_compress = []
        total = len(very_new_memories) if very_new_memories else 0
        for idx, mem in enumerate(very_new_memories or []):
            # Move if over limit (older ones) or over time threshold
            if (total > very_new_limit and idx >= very_new_limit) or (
                mem.get("created_at") and mem["created_at"] < cutoff_date
            ):
                memories_to_compress.append(mem)

        # Compress and move to mid_term
        for mem in memories_to_compress:
            try:
                # Compress first (no destructive action yet)
                compressed = compression_agent.compress_to_mid_term(
                    mem["content"], mem.get("metadata", {})
                )

                new_content = compressed.get("summary") or mem["content"]
                new_metadata = compressed.get("metadata", mem.get("metadata", {}))

                # Insert new mid_term memory first then delete old on success
                try:
                    # Prefer transactional move if db supports it
                    if hasattr(db, "transaction"):
                        with db.transaction():
                            new_id = db.insert_session(
                                user_id=user_id,
                                content=new_content,
                                metadata=new_metadata,
                                tier="mid_term",
                            )
                            db.delete_memory(user_id, mem["id"])
                    else:
                        new_id = db.insert_session(
                            user_id=user_id,
                            content=new_content,
                            metadata=new_metadata,
                            tier="mid_term",
                        )
                        db.delete_memory(user_id, mem["id"])

                    moved["to_mid_term"].append(new_id)
                    logger.info(
                        f"Moved memory {mem['id']} from very_new to mid_term as {new_id}"
                    )

                except Exception as e:
                    logger.exception(
                        f"Failed to insert new mid_term memory for {mem['id']}: {e}"
                    )
                    _send_to_monitoring(
                        "error", f"Failed mid_term insert for {mem['id']}", e
                    )
                    # do not delete source memory; leave for retry on next run

            except Exception as e:
                logger.exception(f"Failed to compress memory {mem['id']}: {e}")
                _send_to_monitoring("error", f"compression failed for {mem['id']}", e)

        # Step 2: Move old mid_term memories to long_term
        cutoff_date = utc_now() - relativedelta(months=mid_term_months)
        mid_term_memories = db.fetch_memories(
            user_id, "mid_term", limit=None, order_by="created_at DESC"
        )

        memories_to_aggregate = []
        total_mid = len(mid_term_memories) if mid_term_memories else 0
        for idx, mem in enumerate(mid_term_memories or []):
            if (total_mid > mid_term_limit and idx >= mid_term_limit) or (
                mem.get("created_at") and mem["created_at"] < cutoff_date
            ):
                memories_to_aggregate.append(mem)

        # Aggregate multiple mid_term memories into long_term insights
        if memories_to_aggregate:
            try:
                contents = [m["content"] for m in memories_to_aggregate]
                metadata_list = [m.get("metadata", {}) for m in memories_to_aggregate]

                aggregated = compression_agent.compress_to_long_term(
                    contents, metadata_list
                )

                new_content = aggregated.get("summary") or "\n".join(contents)
                new_metadata = aggregated.get("metadata", {})

                # Insert new long_term memory first; verify insertion before deleting old ones
                try:
                    if hasattr(db, "transaction"):
                        with db.transaction():
                            new_id = db.insert_session(
                                user_id=user_id,
                                content=new_content,
                                metadata=new_metadata,
                                tier="long_term",
                            )
                            for mem in memories_to_aggregate:
                                db.delete_memory(user_id, mem["id"])
                    else:
                        new_id = db.insert_session(
                            user_id=user_id,
                            content=new_content,
                            metadata=new_metadata,
                            tier="long_term",
                        )
                        for mem in memories_to_aggregate:
                            db.delete_memory(user_id, mem["id"])

                    moved["to_long_term"].append(new_id)
                    logger.info(
                        f"Aggregated {len(memories_to_aggregate)} mid_term memories into long_term {new_id}"
                    )

                except Exception as e:
                    logger.exception(
                        f"Failed to insert/cleanup during aggregation for user {user_id}: {e}"
                    )
                    _send_to_monitoring(
                        "error",
                        f"aggregation insert/delete failed for user {user_id}",
                        e,
                    )
                    # Do not delete source memories if insertion failed

            except Exception as e:
                logger.exception(f"Failed to aggregate mid_term memories: {e}")
                _send_to_monitoring(
                    "error", f"aggregation compression failed for user {user_id}", e
                )

    except Exception as e:
        logger.exception(f"Error in move_memory_between_tiers for user {user_id}: {e}")
        _send_to_monitoring(
            "error", f"move_memory_between_tiers failed for user {user_id}", e
        )

    return moved


# ---------------------------
# Retrieve Context
# ---------------------------


def _normalize_id(raw_id: Any) -> str:
    return str(raw_id)


def _aggregate_score(items: List[Dict[str, Any]], top_n: int = 3) -> float:
    if not items:
        return 0.0
    scores = sorted([float(i.get("score", 0.0)) for i in items], reverse=True)
    return sum(scores[:top_n]) / min(len(scores), top_n)


def get_context(user_id: str, query: str, max_items: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve context using hybrid retrieval where both DB keyword search and vector semantic
    search are used and combined (weighted).

    Improvements:
    - Compute aggregate relevance (top-3 mean) to decide sufficiency
    - Always run vector search supplementary and merge results (concurrently to save time)
    - Normalize ids and handle duplicate ids across sources
    - Cache results for improved performance
    """
    try:
        # Validate input parameters
        validated_data = validate_query_input(user_id, query, max_items)
        user_id = validated_data["user_id"]
        query = validated_data["query"]
        max_items = validated_data.get("limit", max_items)

        # Check cache first
        cache = get_cache_manager()
        cache_key = f"{user_id}:{hash(query)}:{max_items}"
        cached_result = cache.get("query_result", cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached_result

        db_result: List[Dict[str, Any]] = []
        vector_result: List[Dict[str, Any]] = []

        # Run DB retrieval synchronously (usually fast)
        try:
            db_result = retrieval.retrieve_context(user_id, query, max_items=max_items)
        except Exception as e:
            logger.exception(f"DB retrieval failed for user {user_id}: {e}")
            _send_to_monitoring("warning", f"DB retrieval failed for user {user_id}", e)
            db_result = []

        # Run vector retrieval using thread pool to prevent thread exhaustion
        def _run_vector():
            try:
                return vector_db.query_embeddings(user_id, query, top_k=max_items)
            except Exception as e:
                logger.exception(f"Vector retrieval failed for user {user_id}: {e}")
                _send_to_monitoring(
                    "warning", f"vector retrieval failed for user {user_id}", e
                )
                return []

        # Submit vector retrieval to thread pool
        vector_future = _vector_thread_pool.submit(_run_vector)

        # Evaluate DB sufficiency using aggregate scoring (top-3 mean)
        db_score = _aggregate_score(db_result, top_n=3)
        # Thresholds can be tuned; use combined rules to avoid false positives/negatives
        sufficient = False
        if len(db_result) >= max(1, min(max_items, 3)) and db_score >= 0.6:
            # If DB has at least 1-3 items and aggregate score is reasonably high, consider sufficient
            sufficient = True

        # Wait for vector retrieval with timeout
        try:
            vector_result = vector_future.result(timeout=2.0)
        except FutureTimeoutError:
            # If timeout, we'll proceed with what we have; background worker will finish soon.
            logger.debug("Vector retrieval taking long; proceeding with DB results")
            vector_result = []
        except Exception as e:
            logger.exception(f"Vector retrieval failed for user {user_id}: {e}")
            vector_result = []

        # If DB is sufficient, still augment with vector results but give DB preference
        combined: List[Dict[str, Any]] = []
        seen_ids = set()

        # Prefer DB results (preserve their order); normalize ids
        for item in db_result:
            item_id = _normalize_id(item.get("id") or item.get("memory_id"))
            if item_id in seen_ids:
                continue
            item["id"] = item_id
            combined.append(item)
            seen_ids.add(item_id)

        # Add vector results with de-duplication and value normalization
        for item in vector_result:
            if len(combined) >= max_items:
                break
            item_id = _normalize_id(item.get("id") or item.get("memory_id"))
            if item_id in seen_ids:
                continue
            # Ensure vector items have a 'score' float
            item["score"] = float(item.get("score", 0.0))
            item["id"] = item_id
            combined.append(item)
            seen_ids.add(item_id)

        # If DB was not sufficient and we didn't get enough vector results in the short wait,
        # ensure we attempt a blocking vector query (longer) to fill slots (last resort)
        if not sufficient and len(combined) < min(max_items, 3):
            try:
                remaining = max_items - len(combined)
                more_vector = vector_db.query_embeddings(
                    user_id, query, top_k=remaining
                )
                for item in more_vector:
                    if len(combined) >= max_items:
                        break
                    item_id = _normalize_id(item.get("id") or item.get("memory_id"))
                    if item_id in seen_ids:
                        continue
                    item["score"] = float(item.get("score", 0.0))
                    item["id"] = item_id
                    combined.append(item)
                    seen_ids.add(item_id)
            except Exception as e:
                logger.exception(
                    f"Blocking vector retrieval failed for user {user_id}: {e}"
                )
                _send_to_monitoring(
                    "warning", f"blocking vector retrieval failed for user {user_id}", e
                )

        # Sort combined by score descending (prefer higher relevance), but keep stable order for ties
        combined.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        # Final trim
        result = combined[:max_items]

        # Cache the result
        cache.set("query_result", cache_key, result, ttl=900)  # 15 minutes

        return result

    except Exception as e:
        logger.exception(f"Error retrieving context for user {user_id}: {e}")
        _send_to_monitoring("error", f"get_context failed for user {user_id}", e)
        return []


# ---------------------------
# Utility Functions
# ---------------------------


def get_memory_stats(user_id: str) -> Dict[str, int]:
    """Get memory statistics for a user. Handles per-tier errors individually."""
    stats = {"very_new": 0, "mid_term": 0, "long_term": 0}
    for tier in ["very_new", "mid_term", "long_term"]:
        try:
            memories = db.fetch_memories(user_id, tier, limit=None)
            stats[tier] = len(memories) if memories else 0
        except Exception as e:
            logger.exception(
                f"Error fetching memories for tier {tier} for user {user_id}: {e}"
            )
            _send_to_monitoring(
                "warning", f"fetch_memories failed for tier {tier} user {user_id}", e
            )
            stats[tier] = 0
    return stats


def cleanup_old_memories(user_id: str, max_age_days: int = 365 * 2) -> int:
    """Clean up very old memories beyond the retention period. Ensure DB has index on created_at."""
    try:
        cutoff_date = utc_now() - timedelta(days=max_age_days)
        # db.delete_old_memories should be implemented with efficient query using indexed created_at
        return db.delete_old_memories(user_id, cutoff_date)
    except Exception as e:
        logger.exception(f"Error cleaning up old memories for user {user_id}: {e}")
        _send_to_monitoring(
            "error", f"cleanup_old_memories failed for user {user_id}", e
        )
        return 0


# Graceful shutdown helper for worker thread - call at application shutdown
def shutdown_background_workers():
    global _vector_thread_pool
    _stop_worker.set()
    _worker_thread.join(timeout=5)

    # Shutdown thread pool gracefully
    _vector_thread_pool.shutdown(wait=True, timeout=5.0)
    logger.info("Background workers and thread pool stopped")
