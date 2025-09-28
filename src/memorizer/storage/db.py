"""
db.py
Database schema and query helpers for Memorizer.
Handles connections, inserts, updates, and lifecycle transitions.
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from psycopg2.pool import SimpleConnectionPool

# Import configuration manager
try:
    from config import get_config_manager
except ImportError:
    # Fallback for when config module is not available
    def get_config_manager():
        return None

logger = logging.getLogger(__name__)

# Get database configuration
DATABASE_URL = None
try:
    config_manager = get_config_manager()
    db_config = config_manager.get_database_config()
    DATABASE_URL = db_config.url
except Exception as e:
    logger.warning(f"Failed to load database configuration: {e}")
    # Don't raise here - let functions handle missing config gracefully

# Connection pool for better performance and connection management
_connection_pool = None


# ---------------------------
# Connection Helpers
# ---------------------------
def init_connection_pool(
    min_conn: Optional[int] = None, max_conn: Optional[int] = None
) -> None:
    """Initialize connection pool with improved configuration."""
    global _connection_pool

    if DATABASE_URL is None:
        logger.warning("DATABASE_URL not configured, cannot initialize connection pool")
        return

    try:
        # Use configuration values if not provided
        if min_conn is None:
            try:
                config_manager = get_config_manager()
                db_config = config_manager.get_database_config()
                min_conn = db_config.min_connections
            except:
                min_conn = 1  # Default fallback

        if max_conn is None:
            try:
                config_manager = get_config_manager()
                db_config = config_manager.get_database_config()
                max_conn = db_config.max_connections
            except:
                max_conn = 10  # Default fallback

        # Enhanced connection pool configuration
        _connection_pool = SimpleConnectionPool(
            min_conn,
            max_conn,
            DATABASE_URL,
            cursor_factory=RealDictCursor,
            # Connection pool settings for better performance
            kwargs={
                "connect_timeout": 10,  # Connection timeout
                "application_name": "memorizer_framework",  # Application identifier
                "options": "-c default_transaction_isolation=read committed",  # Transaction isolation
            },
        )
        logger.info(
            f"Database connection pool initialized: {min_conn}-{max_conn} connections"
        )

        # Test the connection pool
        _test_connection_pool()

    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}")
        raise


def _test_connection_pool() -> None:
    """Test the connection pool to ensure it's working."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result:
                    logger.info("Connection pool test successful")
                else:
                    raise Exception("Connection pool test failed")
    except Exception as e:
        logger.error(f"Connection pool test failed: {e}")
        raise


def get_connection_pool():
    """Get the connection pool instance."""
    global _connection_pool
    if _connection_pool is None:
        init_connection_pool()
    return _connection_pool

def get_connection_pool_stats() -> Dict[str, Any]:
    """Get connection pool statistics."""
    if _connection_pool is None:
        return {"status": "not_initialized"}

    return {
        "status": "active",
        "min_connections": _connection_pool.minconn,
        "max_connections": _connection_pool.maxconn,
        "current_connections": len(_connection_pool._pool),
        "available_connections": _connection_pool.maxconn - len(_connection_pool._pool),
    }


@contextmanager
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    """Get a connection from the pool with proper cleanup and error recovery."""
    if _connection_pool is None:
        init_connection_pool()

    conn = None
    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            conn = _connection_pool.getconn()

            # Test connection before yielding
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

            yield conn
            break

        except Exception as e:
            retry_count += 1
            logger.warning(
                f"Database connection error (attempt {retry_count}/{max_retries}): {e}"
            )

            # Return bad connection to pool
            if conn:
                try:
                    _connection_pool.putconn(conn, close=True)
                except:
                    pass
                conn = None

            if retry_count >= max_retries:
                logger.error(
                    f"Failed to get database connection after {max_retries} attempts"
                )
                raise

            # Wait before retry
            import time

            time.sleep(0.1 * retry_count)

        finally:
            if conn:
                try:
                    _connection_pool.putconn(conn)
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")


# ---------------------------
# Schema Setup
# ---------------------------
def init_schema():
    """Initialize the database schema if not exists."""
    # Use a unified table with tier column instead of separate tables
    # This is more maintainable and allows for easier queries across tiers
    schema_queries = [
        """
        CREATE TABLE IF NOT EXISTS memories (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            tier TEXT NOT NULL CHECK (tier IN ('very_new', 'mid_term', 'long_term')),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            source_memory_ids INTEGER[] DEFAULT ARRAY[]::INTEGER[]
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS embedding_retry_queue (
            id SERIAL PRIMARY KEY,
            memory_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            retry_count INTEGER DEFAULT 0,
            max_retries INTEGER DEFAULT 3,
            created_at TIMESTAMP DEFAULT NOW(),
            next_retry_at TIMESTAMP DEFAULT NOW(),
            status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'failed', 'completed'))
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memories_user_tier 
        ON memories(user_id, tier, created_at DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memories_user_created 
        ON memories(user_id, created_at DESC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memories_metadata 
        ON memories USING GIN(metadata);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_embedding_retry_queue_status 
        ON embedding_retry_queue(status, next_retry_at);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_embedding_retry_queue_memory_id 
        ON embedding_retry_queue(memory_id);
        """,
        """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        """,
        """
        DROP TRIGGER IF EXISTS update_memories_updated_at ON memories;
        CREATE TRIGGER update_memories_updated_at 
        BEFORE UPDATE ON memories 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """,
    ]

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for query in schema_queries:
                    cur.execute(query)
                conn.commit()
        logger.info("Database schema initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize schema: {e}")
        raise


def migrate_from_old_schema():
    """Migrate from old separate table schema to unified schema."""
    migration_queries = [
        """
        INSERT INTO memories (user_id, content, metadata, tier, created_at)
        SELECT user_id, content, metadata, 'very_new', created_at 
        FROM very_new 
        WHERE NOT EXISTS (
            SELECT 1 FROM memories 
            WHERE memories.user_id = very_new.user_id 
            AND memories.content = very_new.content 
            AND memories.tier = 'very_new'
        );
        """,
        """
        INSERT INTO memories (user_id, content, metadata, tier, created_at)
        SELECT user_id, content, metadata, 'mid_term', created_at 
        FROM mid_term 
        WHERE NOT EXISTS (
            SELECT 1 FROM memories 
            WHERE memories.user_id = mid_term.user_id 
            AND memories.content = mid_term.content 
            AND memories.tier = 'mid_term'
        );
        """,
        """
        INSERT INTO memories (user_id, content, metadata, tier, created_at)
        SELECT user_id, content, metadata, 'long_term', created_at 
        FROM long_term 
        WHERE NOT EXISTS (
            SELECT 1 FROM memories 
            WHERE memories.user_id = long_term.user_id 
            AND memories.content = long_term.content 
            AND memories.tier = 'long_term'
        );
        """,
        "DROP TABLE IF EXISTS very_new;",
        "DROP TABLE IF EXISTS mid_term;",
        "DROP TABLE IF EXISTS long_term;",
    ]

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for query in migration_queries:
                    cur.execute(query)
                conn.commit()
        logger.info("Migration from old schema completed")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


# ---------------------------
# Insert Helpers
# ---------------------------
def insert_session(
    user_id: str,
    content: str,
    metadata: Dict[str, Any] = None,
    tier: str = "very_new",
    source_memory_ids: List[int] = None,
) -> str:
    """
    Insert a new session into the specified tier.

    Args:
        user_id: User identifier
        content: Session content
        metadata: Additional metadata
        tier: Memory tier (very_new, mid_term, long_term)
        source_memory_ids: List of memory IDs that were compressed to create this

    Returns:
        The ID of the inserted memory
    """
    if metadata is None:
        metadata = {}
    if source_memory_ids is None:
        source_memory_ids = []

    # Validate tier
    if tier not in ["very_new", "mid_term", "long_term"]:
        raise ValueError(f"Invalid tier: {tier}")

    query = """
        INSERT INTO memories (user_id, content, metadata, tier, source_memory_ids)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query, (user_id, content, Json(metadata), tier, source_memory_ids)
                )
                row = cur.fetchone()
                conn.commit()
                return str(row["id"])
    except Exception as e:
        logger.error(f"Failed to insert session for user {user_id}: {e}")
        raise


# ---------------------------
# Fetch Helpers
# ---------------------------
def fetch_memories(
    user_id: str,
    tier: str = None,
    limit: int = None,
    offset: int = 0,
    order_by: str = "created_at DESC",
) -> List[Dict[str, Any]]:
    """
    Fetch memories from specified tier(s).

    Args:
        user_id: User identifier
        tier: Specific tier to fetch from (None for all tiers)
        limit: Maximum number of records to return
        offset: Number of records to skip
        order_by: Order clause

    Returns:
        List of memory dictionaries
    """
    # Validate inputs
    from .validation import InputValidator

    user_result = InputValidator.validate_user_id(user_id)
    if not user_result.is_valid:
        logger.error(f"Invalid user_id: {user_result.errors}")
        return []

    # Use validated user_id
    user_id = user_result.sanitized_value

    conditions = ["user_id = %s"]
    params = [user_id]

    if tier:
        # Validate tier
        tier_result = InputValidator.validate_tier(tier)
        if not tier_result.is_valid:
            logger.error(f"Invalid tier: {tier_result.errors}")
            return []

        conditions.append("tier = %s")
        params.append(tier_result.sanitized_value)

    # Validate limit and offset
    if limit is not None:
        limit_result = InputValidator.validate_limit(limit)
        if not limit_result.is_valid:
            logger.error(f"Invalid limit: {limit_result.errors}")
            return []
        limit = limit_result.sanitized_value

    offset_result = InputValidator.validate_offset(offset)
    if not offset_result.is_valid:
        logger.error(f"Invalid offset: {offset_result.errors}")
        return []
    offset = offset_result.sanitized_value

    # Validate order_by to prevent SQL injection
    allowed_order_columns = ["created_at", "updated_at", "id"]
    allowed_order_directions = ["ASC", "DESC"]

    order_parts = order_by.split()
    if len(order_parts) != 2:
        order_by = "created_at DESC"  # Default safe value
    else:
        column, direction = order_parts
        if (
            column not in allowed_order_columns
            or direction.upper() not in allowed_order_directions
        ):
            order_by = "created_at DESC"  # Default safe value

    # Build query with proper parameterization
    query = f"""
        SELECT id, user_id, content, metadata, tier, created_at, updated_at, source_memory_ids
        FROM memories
        WHERE {' AND '.join(conditions)}
        ORDER BY {order_by}
    """

    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    if offset > 0:
        query += " OFFSET %s"
        params.append(offset)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to fetch memories for user {user_id}: {e}")
        raise


def search_memories(
    user_id: str, search_term: str, tier: str = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search memories by content using full-text search.

    Args:
        user_id: User identifier
        search_term: Text to search for
        tier: Specific tier to search in
        limit: Maximum results to return

    Returns:
        List of matching memories with relevance scores
    """
    # Validate inputs
    from .validation import InputValidator

    user_result = InputValidator.validate_user_id(user_id)
    if not user_result.is_valid:
        logger.error(f"Invalid user_id: {user_result.errors}")
        return []

    query_result = InputValidator.validate_query(search_term)
    if not query_result.is_valid:
        logger.error(f"Invalid search_term: {query_result.errors}")
        return []

    limit_result = InputValidator.validate_limit(limit)
    if not limit_result.is_valid:
        logger.error(f"Invalid limit: {limit_result.errors}")
        return []

    # Use validated values
    user_id = user_result.sanitized_value
    search_term = query_result.sanitized_value
    limit = limit_result.sanitized_value

    # Build query with proper parameterization
    conditions = ["user_id = %s"]
    params = [user_id]

    if tier:
        # Validate tier
        tier_result = InputValidator.validate_tier(tier)
        if not tier_result.is_valid:
            logger.error(f"Invalid tier: {tier_result.errors}")
            return []

        conditions.append("tier = %s")
        params.append(tier_result.sanitized_value)

    # Use PostgreSQL full-text search with proper parameterization
    conditions.append(
        "to_tsvector('english', content) @@ plainto_tsquery('english', %s)"
    )
    params.append(search_term)

    # Build query safely - all dynamic parts are parameterized
    query = (
        """
        SELECT *, 
               ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as relevance_score
        FROM memories
        WHERE """
        + " AND ".join(conditions)
        + """
        ORDER BY relevance_score DESC, created_at DESC
        LIMIT %s;
    """
    )

    # Add parameters for ts_rank and limit
    params.append(search_term)  # for ts_rank
    params.append(limit)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to search memories for user {user_id}: {e}")
        return []


# ---------------------------
# Move Between Tiers
# ---------------------------
def move_memory(
    user_id: str,
    source_tier: str,
    target_tier: str,
    memory_id: int,
    new_content: str,
    new_metadata: Dict[str, Any] = None,
) -> str:
    """
    Move memory from one tier to another, updating content and metadata.

    Args:
        user_id: User identifier
        source_tier: Source tier name
        target_tier: Target tier name
        memory_id: ID of memory to move
        new_content: Updated content (compressed)
        new_metadata: Updated metadata

    Returns:
        ID of the updated memory record
    """
    if new_metadata is None:
        new_metadata = {}

    # Validate tiers
    valid_tiers = ["very_new", "mid_term", "long_term"]
    if source_tier not in valid_tiers or target_tier not in valid_tiers:
        raise ValueError("Invalid tier specified")

    query = """
        UPDATE memories 
        SET tier = %s, content = %s, metadata = %s, updated_at = NOW()
        WHERE id = %s AND user_id = %s AND tier = %s
        RETURNING id;
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        target_tier,
                        new_content,
                        Json(new_metadata),
                        memory_id,
                        user_id,
                        source_tier,
                    ),
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError(
                        f"Memory {memory_id} not found in {source_tier} for user {user_id}"
                    )
                conn.commit()
                return str(row["id"])
    except Exception as e:
        logger.error(f"Failed to move memory {memory_id}: {e}")
        raise


def delete_memory(user_id: str, memory_id: int) -> bool:
    """Delete a specific memory."""
    query = "DELETE FROM memories WHERE id = %s AND user_id = %s;"

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (memory_id, user_id))
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted
    except Exception as e:
        logger.error(f"Failed to delete memory {memory_id}: {e}")
        raise


def delete_old_memories(user_id: str, cutoff_date: datetime) -> int:
    """Delete memories older than the cutoff date."""
    query = "DELETE FROM memories WHERE user_id = %s AND created_at < %s;"

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (user_id, cutoff_date))
                deleted_count = cur.rowcount
                conn.commit()
                return deleted_count
    except Exception as e:
        logger.error(f"Failed to delete old memories for user {user_id}: {e}")
        raise


# ---------------------------
# Analytics and Stats
# ---------------------------
def get_user_memory_stats(user_id: str) -> Dict[str, Any]:
    """Get comprehensive memory statistics for a user."""
    query = """
        SELECT 
            tier,
            COUNT(*) as count,
            AVG(LENGTH(content)) as avg_content_length,
            MIN(created_at) as oldest_memory,
            MAX(created_at) as newest_memory
        FROM memories 
        WHERE user_id = %s 
        GROUP BY tier;
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (user_id,))
                rows = cur.fetchall()

                stats = {
                    "very_new": {"count": 0, "avg_length": 0},
                    "mid_term": {"count": 0, "avg_length": 0},
                    "long_term": {"count": 0, "avg_length": 0},
                    "total_memories": 0,
                }

                for row in rows:
                    tier_stats = {
                        "count": row["count"],
                        "avg_length": int(row["avg_content_length"] or 0),
                        "oldest": row["oldest_memory"],
                        "newest": row["newest_memory"],
                    }
                    stats[row["tier"]] = tier_stats
                    stats["total_memories"] += row["count"]

                return stats
    except Exception as e:
        logger.error(f"Failed to get stats for user {user_id}: {e}")
        return {}


# ---------------------------
# Initialization
# ---------------------------
def initialize_db():
    """Initialize the database with schema and connection pool."""
    try:
        init_connection_pool()
        init_schema()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


# ---------------------------
# Embedding Retry Queue Functions
# ---------------------------
def enqueue_embedding_retry(
    memory_id: str, user_id: str, content: str, metadata: Dict[str, Any] = None
) -> bool:
    """
    Enqueue an embedding for retry processing.

    Args:
        memory_id: ID of the memory that needs embedding
        user_id: User identifier
        content: Content to embed
        metadata: Additional metadata

    Returns:
        True if successfully enqueued, False otherwise
    """
    if metadata is None:
        metadata = {}

    query = """
        INSERT INTO embedding_retry_queue (memory_id, user_id, content, metadata)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (memory_id) DO UPDATE SET
            retry_count = embedding_retry_queue.retry_count + 1,
            next_retry_at = NOW() + INTERVAL '1 hour' * embedding_retry_queue.retry_count,
            status = CASE 
                WHEN embedding_retry_queue.retry_count >= embedding_retry_queue.max_retries 
                THEN 'failed' 
                ELSE 'pending' 
            END;
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (memory_id, user_id, content, Json(metadata)))
                conn.commit()
                logger.info(f"Enqueued embedding retry for memory {memory_id}")
                return True
    except Exception as e:
        logger.error(f"Failed to enqueue embedding retry for {memory_id}: {e}")
        return False


def get_pending_embedding_retries(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get pending embedding retries that are ready for processing.

    Args:
        limit: Maximum number of retries to return

    Returns:
        List of pending retry records
    """
    query = """
        SELECT id, memory_id, user_id, content, metadata, retry_count, max_retries
        FROM embedding_retry_queue
        WHERE status = 'pending' 
        AND next_retry_at <= NOW()
        ORDER BY created_at ASC
        LIMIT %s;
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (limit,))
                return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get pending embedding retries: {e}")
        return []


def mark_embedding_retry_completed(retry_id: int) -> bool:
    """Mark an embedding retry as completed."""
    query = "UPDATE embedding_retry_queue SET status = 'completed' WHERE id = %s;"

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (retry_id,))
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to mark retry {retry_id} as completed: {e}")
        return False


def mark_embedding_retry_failed(retry_id: int) -> bool:
    """Mark an embedding retry as failed."""
    query = """
        UPDATE embedding_retry_queue 
        SET status = CASE 
            WHEN retry_count >= max_retries THEN 'failed'
            ELSE 'pending'
        END,
        next_retry_at = CASE 
            WHEN retry_count < max_retries THEN NOW() + INTERVAL '1 hour' * (retry_count + 1)
            ELSE next_retry_at
        END
        WHERE id = %s;
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (retry_id,))
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to mark retry {retry_id} as failed: {e}")
        return False


# ---------------------------
# Transaction Support
# ---------------------------
@contextmanager
def transaction():
    """
    Database transaction context manager.
    Provides transactional semantics for operations that need to be atomic.
    """
    conn = None
    try:
        if _connection_pool is None:
            init_connection_pool()

        conn = _connection_pool.getconn()
        conn.autocommit = False
        yield conn

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Transaction failed: {e}")
        raise
    finally:
        if conn:
            try:
                conn.commit()
            except Exception as commit_error:
                logger.error(f"Failed to commit transaction: {commit_error}")
                conn.rollback()
            finally:
                _connection_pool.putconn(conn)
