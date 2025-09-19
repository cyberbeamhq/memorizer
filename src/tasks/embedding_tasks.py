"""
embedding_tasks.py
Celery tasks for embedding processing.
"""

import logging
from typing import Any, Dict, List

from celery import current_task

from .. import db, memory_manager, vector_db
from ..celery_app import app

logger = logging.getLogger(__name__)


@app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_embedding(
    self, memory_id: str, user_id: str, content: str, metadata: Dict[str, Any] = None
):
    """
    Process embedding insertion in background.

    Args:
        memory_id: Memory ID
        user_id: User ID
        content: Content to embed
        metadata: Additional metadata
    """
    try:
        logger.info(f"Processing embedding for memory {memory_id}")

        # Update task status
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting embedding process"},
        )

        # Insert embedding
        success = vector_db.insert_embedding(
            memory_id, user_id, content, metadata or {}
        )

        if success:
            logger.info(f"Successfully processed embedding for memory {memory_id}")
            return {
                "status": "SUCCESS",
                "memory_id": memory_id,
                "message": "Embedding processed successfully",
            }
        else:
            raise Exception(f"Failed to insert embedding for memory {memory_id}")

    except Exception as exc:
        logger.error(f"Embedding processing failed for memory {memory_id}: {exc}")

        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            logger.info(
                f"Retrying embedding processing for memory {memory_id} (attempt {self.request.retries + 1})"
            )
            raise self.retry(countdown=60 * (2**self.request.retries))
        else:
            # Mark as failed and enqueue for retry
            logger.error(
                f"Max retries exceeded for memory {memory_id}, enqueueing for retry"
            )
            db.enqueue_embedding_retry(memory_id, user_id, content, metadata)
            raise exc


@app.task
def process_retry_queue():
    """Process failed embedding retries."""
    try:
        logger.info("Processing embedding retry queue")

        # Get pending retries
        retries = db.get_pending_embedding_retries(limit=10)

        processed = 0
        for retry_item in retries:
            try:
                # Process the retry
                success = vector_db.insert_embedding(
                    retry_item["memory_id"],
                    retry_item["user_id"],
                    retry_item["content"],
                    retry_item["metadata"],
                )

                if success:
                    db.mark_embedding_retry_completed(retry_item["id"])
                    processed += 1
                else:
                    db.mark_embedding_retry_failed(retry_item["id"])

            except Exception as e:
                logger.error(f"Failed to process retry {retry_item['id']}: {e}")
                db.mark_embedding_retry_failed(retry_item["id"])

        logger.info(f"Processed {processed} embedding retries")
        return {"processed": processed, "total": len(retries)}

    except Exception as e:
        logger.error(f"Failed to process retry queue: {e}")
        raise


@app.task
def batch_process_embeddings(embedding_requests: List[Dict[str, Any]]):
    """
    Process multiple embeddings in batch.

    Args:
        embedding_requests: List of embedding request dictionaries
    """
    try:
        logger.info(f"Processing batch of {len(embedding_requests)} embeddings")

        results = []
        for request in embedding_requests:
            try:
                success = vector_db.insert_embedding(
                    request["memory_id"],
                    request["user_id"],
                    request["content"],
                    request.get("metadata", {}),
                )
                results.append({"memory_id": request["memory_id"], "success": success})
            except Exception as e:
                logger.error(
                    f"Failed to process embedding for {request['memory_id']}: {e}"
                )
                results.append(
                    {
                        "memory_id": request["memory_id"],
                        "success": False,
                        "error": str(e),
                    }
                )

        successful = sum(1 for r in results if r["success"])
        logger.info(
            f"Batch processing completed: {successful}/{len(embedding_requests)} successful"
        )

        return {
            "total": len(embedding_requests),
            "successful": successful,
            "failed": len(embedding_requests) - successful,
            "results": results,
        }

    except Exception as e:
        logger.error(f"Batch embedding processing failed: {e}")
        raise
