"""
celery_app.py
Production-ready Celery configuration for background job processing.
"""

import logging
import os

from celery import Celery
from celery.signals import worker_ready, worker_shutdown

logger = logging.getLogger(__name__)

# Create Celery app
app = Celery("memorizer")

# Configuration
app.conf.update(
    broker_url=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    result_backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    # Retry settings
    task_default_retry_delay=60,
    task_max_retries=3,
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Routing
    task_routes={
        "memorizer.tasks.embedding_tasks.*": {"queue": "embeddings"},
        "memorizer.tasks.compression_tasks.*": {"queue": "compression"},
        "memorizer.tasks.cleanup_tasks.*": {"queue": "cleanup"},
    },
    # Beat schedule for periodic tasks
    beat_schedule={
        "process-embedding-retries": {
            "task": "memorizer.tasks.embedding_tasks.process_retry_queue",
            "schedule": 300.0,  # Every 5 minutes
        },
        "move-memories-between-tiers": {
            "task": "memorizer.tasks.compression_tasks.move_memories_between_tiers",
            "schedule": 3600.0,  # Every hour
        },
        "cleanup-old-memories": {
            "task": "memorizer.tasks.cleanup_tasks.cleanup_old_memories",
            "schedule": 86400.0,  # Daily
        },
    },
)


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready signal."""
    logger.info(f"Worker {sender} is ready")


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Handle worker shutdown signal."""
    logger.info(f"Worker {sender} is shutting down")


# Import tasks
from . import tasks
