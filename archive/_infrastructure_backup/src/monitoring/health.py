"""
health.py
Health check system for production monitoring.
"""

import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import psycopg2
import redis

from . import db, memory_manager, vector_db

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """Individual health check component."""

    def __init__(
        self, name: str, check_func, timeout: float = 5.0, critical: bool = True
    ):
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
        self.critical = critical
        self.last_check = None
        self.last_status = None
        self.last_error = None

    def run_check(self) -> Dict[str, Any]:
        """Run the health check."""
        start_time = time.time()

        try:
            result = self.check_func()
            duration = time.time() - start_time

            self.last_check = datetime.now(timezone.utc)
            self.last_status = HealthStatus.HEALTHY
            self.last_error = None

            return {
                "status": HealthStatus.HEALTHY.value,
                "duration_ms": round(duration * 1000, 2),
                "details": result,
                "timestamp": self.last_check.isoformat(),
            }

        except Exception as e:
            duration = time.time() - start_time

            self.last_check = datetime.now(timezone.utc)
            self.last_status = (
                HealthStatus.UNHEALTHY if self.critical else HealthStatus.DEGRADED
            )
            self.last_error = str(e)

            logger.error(f"Health check '{self.name}' failed: {e}")

            return {
                "status": self.last_status.value,
                "duration_ms": round(duration * 1000, 2),
                "error": str(e),
                "timestamp": self.last_check.isoformat(),
            }


class HealthChecker:
    """Main health checker system."""

    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("database", self._check_database, critical=True)
        self.register_check("redis", self._check_redis, critical=True)
        self.register_check("vector_db", self._check_vector_db, critical=False)
        self.register_check("memory_manager", self._check_memory_manager, critical=True)
        self.register_check("disk_space", self._check_disk_space, critical=True)
        self.register_check("memory_usage", self._check_memory_usage, critical=False)

    def register_check(
        self, name: str, check_func, timeout: float = 5.0, critical: bool = True
    ):
        """Register a new health check."""
        self.checks[name] = HealthCheck(name, check_func, timeout, critical)

    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Test basic connectivity
                    cur.execute("SELECT 1")

                    # Test query performance
                    start_time = time.time()
                    cur.execute("SELECT COUNT(*) FROM memories LIMIT 1")
                    query_time = time.time() - start_time

                    # Get connection pool info
                    cur.execute(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
                    )
                    active_connections = cur.fetchone()[0]

                    return {
                        "connected": True,
                        "query_time_ms": round(query_time * 1000, 2),
                        "active_connections": active_connections,
                        "database_name": conn.info.dbname,
                    }
        except Exception as e:
            raise Exception(f"Database check failed: {e}")

    def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
            r = redis.from_url(redis_url)

            # Test basic operations
            r.ping()

            # Test set/get
            test_key = f"health_check_{int(time.time())}"
            r.set(test_key, "test", ex=10)
            value = r.get(test_key)
            r.delete(test_key)

            # Get Redis info
            info = r.info()

            return {
                "connected": True,
                "version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
            }
        except Exception as e:
            raise Exception(f"Redis check failed: {e}")

    def _check_vector_db(self) -> Dict[str, Any]:
        """Check vector database connectivity."""
        try:
            provider = vector_db.get_vector_db_provider()

            # Test basic operations
            test_content = "health check test content"
            test_metadata = {"health_check": True}

            # Try to insert and query
            success = provider.insert_embedding(
                "health_check", "health_user", test_content, test_metadata
            )

            if success:
                results = provider.query_embeddings("health_user", "test", top_k=1)

                return {
                    "connected": True,
                    "provider_type": type(provider).__name__,
                    "test_query_results": len(results),
                }
            else:
                raise Exception("Vector DB insert failed")

        except Exception as e:
            raise Exception(f"Vector DB check failed: {e}")

    def _check_memory_manager(self) -> Dict[str, Any]:
        """Check memory manager functionality."""
        try:
            # Test basic operations
            test_user = "health_check_user"
            test_content = "Health check test memory"

            # Add a test memory
            memory_id = memory_manager.add_session(
                test_user, test_content, {"health_check": True}
            )

            # Get stats
            stats = memory_manager.get_memory_stats(test_user)

            # Clean up
            db.delete_memory(test_user, memory_id)

            return {
                "functional": True,
                "test_memory_id": memory_id,
                "user_stats": stats,
            }
        except Exception as e:
            raise Exception(f"Memory manager check failed: {e}")

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil

            # Check current directory disk usage
            total, used, free = shutil.disk_usage(".")

            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            usage_percent = (used / total) * 100

            if free_gb < 1.0:  # Less than 1GB free
                raise Exception(f"Low disk space: {free_gb:.2f}GB free")

            return {
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "usage_percent": round(usage_percent, 2),
            }
        except Exception as e:
            raise Exception(f"Disk space check failed: {e}")

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()

            return {
                "process_memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                "system_memory_percent": system_memory.percent,
                "system_memory_available_gb": round(
                    system_memory.available / 1024 / 1024 / 1024, 2
                ),
            }
        except Exception as e:
            raise Exception(f"Memory usage check failed: {e}")

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY

        for name, check in self.checks.items():
            try:
                result = check.run_check()
                results[name] = result

                # Update overall status
                if result["status"] == HealthStatus.UNHEALTHY.value:
                    overall_status = HealthStatus.UNHEALTHY
                elif (
                    result["status"] == HealthStatus.DEGRADED.value
                    and overall_status == HealthStatus.HEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED

            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results[name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                overall_status = HealthStatus.UNHEALTHY

        return {
            "status": overall_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": results,
            "summary": {
                "total_checks": len(self.checks),
                "healthy": sum(
                    1
                    for r in results.values()
                    if r["status"] == HealthStatus.HEALTHY.value
                ),
                "degraded": sum(
                    1
                    for r in results.values()
                    if r["status"] == HealthStatus.DEGRADED.value
                ),
                "unhealthy": sum(
                    1
                    for r in results.values()
                    if r["status"] == HealthStatus.UNHEALTHY.value
                ),
            },
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return self.run_all_checks()


# Global health checker instance
_health_checker = HealthChecker()


def get_health_status() -> Dict[str, Any]:
    """Get system health status."""
    return _health_checker.get_health_status()


def register_health_check(
    name: str, check_func, timeout: float = 5.0, critical: bool = True
):
    """Register a custom health check."""
    _health_checker.register_check(name, check_func, timeout, critical)
