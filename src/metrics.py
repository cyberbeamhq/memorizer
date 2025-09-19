"""
metrics.py
Prometheus metrics collection for production monitoring.
"""

import logging
import time
from functools import wraps
from typing import Any, Dict, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

logger = logging.getLogger(__name__)

# Create custom registry
registry = CollectorRegistry()

# Memory metrics
memory_operations_total = Counter(
    "memorizer_memory_operations_total",
    "Total number of memory operations",
    ["operation", "tier", "status"],
    registry=registry,
)

memory_operation_duration = Histogram(
    "memorizer_memory_operation_duration_seconds",
    "Duration of memory operations",
    ["operation", "tier"],
    registry=registry,
)

memory_count = Gauge(
    "memorizer_memory_count", "Number of memories by tier", ["tier"], registry=registry
)

# Embedding metrics
embedding_operations_total = Counter(
    "memorizer_embedding_operations_total",
    "Total number of embedding operations",
    ["operation", "status"],
    registry=registry,
)

embedding_operation_duration = Histogram(
    "memorizer_embedding_operation_duration_seconds",
    "Duration of embedding operations",
    ["operation"],
    registry=registry,
)

# Compression metrics
compression_operations_total = Counter(
    "memorizer_compression_operations_total",
    "Total number of compression operations",
    ["operation", "status"],
    registry=registry,
)

compression_operation_duration = Histogram(
    "memorizer_compression_operation_duration_seconds",
    "Duration of compression operations",
    ["operation"],
    registry=registry,
)

compression_ratio = Histogram(
    "memorizer_compression_ratio",
    "Compression ratio (original_size / compressed_size)",
    ["operation"],
    registry=registry,
)

# Retrieval metrics
retrieval_operations_total = Counter(
    "memorizer_retrieval_operations_total",
    "Total number of retrieval operations",
    ["method", "status"],
    registry=registry,
)

retrieval_operation_duration = Histogram(
    "memorizer_retrieval_operation_duration_seconds",
    "Duration of retrieval operations",
    ["method"],
    registry=registry,
)

retrieval_results_count = Histogram(
    "memorizer_retrieval_results_count",
    "Number of results returned by retrieval",
    ["method"],
    registry=registry,
)

# Database metrics
database_operations_total = Counter(
    "memorizer_database_operations_total",
    "Total number of database operations",
    ["operation", "status"],
    registry=registry,
)

database_operation_duration = Histogram(
    "memorizer_database_operation_duration_seconds",
    "Duration of database operations",
    ["operation"],
    registry=registry,
)

database_connections_active = Gauge(
    "memorizer_database_connections_active",
    "Number of active database connections",
    registry=registry,
)

# System metrics
system_info = Info("memorizer_system_info", "System information", registry=registry)

system_memory_usage = Gauge(
    "memorizer_system_memory_usage_bytes",
    "System memory usage in bytes",
    registry=registry,
)

system_cpu_usage = Gauge(
    "memorizer_system_cpu_usage_percent",
    "System CPU usage percentage",
    registry=registry,
)

# User metrics
user_operations_total = Counter(
    "memorizer_user_operations_total",
    "Total number of user operations",
    ["user_id", "operation", "status"],
    registry=registry,
)

active_users = Gauge(
    "memorizer_active_users", "Number of active users", registry=registry
)


def track_operation(operation_name: str, labels: Dict[str, str] = None):
    """Decorator to track operation metrics."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                logger.error(f"Operation {operation_name} failed: {e}")
                raise
            finally:
                duration = time.time() - start_time

                # Update metrics based on operation type
                if "memory" in operation_name.lower():
                    tier = labels.get("tier", "unknown") if labels else "unknown"
                    memory_operations_total.labels(
                        operation=operation_name, tier=tier, status=status
                    ).inc()
                    memory_operation_duration.labels(
                        operation=operation_name, tier=tier
                    ).observe(duration)

                elif "embedding" in operation_name.lower():
                    embedding_operations_total.labels(
                        operation=operation_name, status=status
                    ).inc()
                    embedding_operation_duration.labels(
                        operation=operation_name
                    ).observe(duration)

                elif "compression" in operation_name.lower():
                    compression_operations_total.labels(
                        operation=operation_name, status=status
                    ).inc()
                    compression_operation_duration.labels(
                        operation=operation_name
                    ).observe(duration)

                elif "retrieval" in operation_name.lower():
                    method = labels.get("method", "unknown") if labels else "unknown"
                    retrieval_operations_total.labels(
                        method=method, status=status
                    ).inc()
                    retrieval_operation_duration.labels(method=method).observe(duration)

                elif "database" in operation_name.lower():
                    database_operations_total.labels(
                        operation=operation_name, status=status
                    ).inc()
                    database_operation_duration.labels(
                        operation=operation_name
                    ).observe(duration)

        return wrapper

    return decorator


def update_memory_count(tier: str, count: int):
    """Update memory count metric."""
    memory_count.labels(tier=tier).set(count)


def update_compression_ratio(operation: str, original_size: int, compressed_size: int):
    """Update compression ratio metric."""
    if compressed_size > 0:
        ratio = original_size / compressed_size
        compression_ratio.labels(operation=operation).observe(ratio)


def update_retrieval_results(method: str, count: int):
    """Update retrieval results count metric."""
    retrieval_results_count.labels(method=method).observe(count)


def update_database_connections(count: int):
    """Update database connections metric."""
    database_connections_active.set(count)


def update_system_metrics():
    """Update system metrics."""
    try:
        import psutil

        # Memory usage
        memory_info = psutil.virtual_memory()
        system_memory_usage.set(memory_info.used)

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        system_cpu_usage.set(cpu_percent)

        # System info
        system_info.info(
            {
                "version": "0.1.0",
                "python_version": psutil.sys.version,
                "platform": psutil.sys.platform,
            }
        )

    except ImportError:
        logger.warning("psutil not available for system metrics")
    except Exception as e:
        logger.error(f"Failed to update system metrics: {e}")


def track_user_operation(user_id: str, operation: str, status: str = "success"):
    """Track user operation."""
    user_operations_total.labels(
        user_id=user_id, operation=operation, status=status
    ).inc()


def update_active_users(count: int):
    """Update active users count."""
    active_users.set(count)


def get_metrics() -> str:
    """Get all metrics in Prometheus format."""
    update_system_metrics()
    return generate_latest(registry).decode("utf-8")


def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary for health checks."""
    try:
        # This is a simplified version - in production you'd want more sophisticated aggregation
        return {
            "total_memory_operations": sum(
                [
                    memory_operations_total.labels(
                        operation=op, tier=tier, status=status
                    )._value
                    for op in ["create", "read", "update", "delete"]
                    for tier in ["very_new", "mid_term", "long_term"]
                    for status in ["success", "error"]
                ]
            ),
            "total_embedding_operations": sum(
                [
                    embedding_operations_total.labels(
                        operation=op, status=status
                    )._value
                    for op in ["insert", "query", "delete"]
                    for status in ["success", "error"]
                ]
            ),
            "total_compression_operations": sum(
                [
                    compression_operations_total.labels(
                        operation=op, status=status
                    )._value
                    for op in ["mid_term", "long_term"]
                    for status in ["success", "error"]
                ]
            ),
            "total_retrieval_operations": sum(
                [
                    retrieval_operations_total.labels(
                        method=method, status=status
                    )._value
                    for method in ["database", "vector", "hybrid"]
                    for status in ["success", "error"]
                ]
            ),
        }
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        return {}
