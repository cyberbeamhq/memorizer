"""
health_monitor.py
Comprehensive health monitoring system for the Memorizer framework.
Provides health checks, dependency monitoring, and automated testing.
"""

import logging
import time
import threading
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check data structure."""

    name: str
    status: HealthStatus
    message: str
    response_time_ms: float
    last_checked: datetime
    details: Dict[str, Any]
    critical: bool = False


@dataclass
class ComponentHealth:
    """Component health information."""

    name: str
    status: HealthStatus
    checks: List[HealthCheck]
    last_updated: datetime
    uptime: float
    version: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health information."""

    overall_status: HealthStatus
    components: Dict[str, ComponentHealth]
    uptime: float
    version: str
    last_updated: datetime
    summary: Dict[str, Any]


class HealthCheckRegistry:
    """Registry for health check functions."""

    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.check_configs: Dict[str, Dict[str, Any]] = {}

    def register(
        self, name: str, check_func: Callable, config: Dict[str, Any] = None
    ) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
        self.check_configs[name] = config or {}
        logger.info(f"Registered health check: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a health check function."""
        if name in self.checks:
            del self.checks[name]
            del self.check_configs[name]
            logger.info(f"Unregistered health check: {name}")

    def get_check(self, name: str) -> Optional[Callable]:
        """Get a health check function by name."""
        return self.checks.get(name)

    def list_checks(self) -> List[str]:
        """List all registered health check names."""
        return list(self.checks.keys())


class HealthMonitor:
    """Main health monitoring system."""

    def __init__(self):
        self.registry = HealthCheckRegistry()
        self.start_time = time.time()
        self.version = "1.0.0"  # This should come from configuration

        # Health check results cache
        self.health_cache: Dict[str, HealthCheck] = {}
        self.cache_ttl = 30  # seconds
        self.lock = threading.Lock()

        # Background monitoring
        self.monitoring_active = True
        self.monitor_thread = None
        self._start_background_monitoring()

        # Register default health checks
        self._register_default_checks()

    def _start_background_monitoring(self) -> None:
        """Start background health monitoring."""

        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._run_all_checks()
                    time.sleep(self.cache_ttl)
                except Exception as e:
                    logger.error(f"Background health monitoring error: {e}")
                    time.sleep(60)  # Wait longer on error

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Background health monitoring started")

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        # Database health check
        self.registry.register(
            "database", self._check_database, {"timeout": 5, "critical": True}
        )

        # Cache health check
        self.registry.register(
            "cache", self._check_cache, {"timeout": 3, "critical": False}
        )

        # Vector database health check
        self.registry.register(
            "vector_db", self._check_vector_db, {"timeout": 5, "critical": False}
        )

        # Memory manager health check
        self.registry.register(
            "memory_manager",
            self._check_memory_manager,
            {"timeout": 3, "critical": True},
        )

        # System resources health check
        self.registry.register(
            "system_resources",
            self._check_system_resources,
            {"timeout": 2, "critical": False},
        )

        # External services health check
        self.registry.register(
            "external_services",
            self._check_external_services,
            {"timeout": 10, "critical": False},
        )

    def _check_database(self) -> HealthCheck:
        """Check database connectivity and performance."""
        start_time = time.time()

        try:
            from . import db

            # Test connection
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()

                    if result and result[0] == 1:
                        # Test query performance
                        cur.execute("SELECT COUNT(*) FROM memories LIMIT 1")
                        cur.fetchone()

                        response_time = (time.time() - start_time) * 1000

                        return HealthCheck(
                            name="database",
                            status=HealthStatus.HEALTHY,
                            message="Database is accessible and responding",
                            response_time_ms=response_time,
                            last_checked=datetime.now(timezone.utc),
                            details={
                                "connection_pool_stats": db.get_connection_pool_stats(),
                                "test_query_successful": True,
                            },
                            critical=True,
                        )
                    else:
                        return HealthCheck(
                            name="database",
                            status=HealthStatus.UNHEALTHY,
                            message="Database test query failed",
                            response_time_ms=(time.time() - start_time) * 1000,
                            last_checked=datetime.now(timezone.utc),
                            details={"test_query_result": result},
                            critical=True,
                        )

        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=datetime.now(timezone.utc),
                details={"error": str(e), "error_type": type(e).__name__},
                critical=True,
            )

    def _check_cache(self) -> HealthCheck:
        """Check cache connectivity and performance."""
        start_time = time.time()

        try:
            from .cache import get_cache_manager

            cache = get_cache_manager()

            # Test cache operations
            test_key = "health_check_test"
            test_value = "test_value"

            # Test set operation
            set_success = cache.set("test", test_key, test_value, ttl=60)

            # Test get operation
            retrieved_value = cache.get("test", test_key)

            # Test delete operation
            cache.delete("test", test_key)

            response_time = (time.time() - start_time) * 1000

            if set_success and retrieved_value == test_value:
                return HealthCheck(
                    name="cache",
                    status=HealthStatus.HEALTHY,
                    message="Cache is accessible and functioning",
                    response_time_ms=response_time,
                    last_checked=datetime.now(timezone.utc),
                    details={
                        "cache_stats": cache.get_stats(),
                        "test_operations_successful": True,
                    },
                    critical=False,
                )
            else:
                return HealthCheck(
                    name="cache",
                    status=HealthStatus.DEGRADED,
                    message="Cache operations partially failed",
                    response_time_ms=response_time,
                    last_checked=datetime.now(timezone.utc),
                    details={
                        "set_success": set_success,
                        "retrieved_value": retrieved_value,
                        "expected_value": test_value,
                    },
                    critical=False,
                )

        except Exception as e:
            return HealthCheck(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=datetime.now(timezone.utc),
                details={"error": str(e), "error_type": type(e).__name__},
                critical=False,
            )

    def _check_vector_db(self) -> HealthCheck:
        """Check vector database connectivity and performance."""
        start_time = time.time()

        try:
            from . import vector_db

            # Test vector database operations
            test_user_id = "health_check_user"
            test_content = "This is a test for health check"

            # Test embedding generation
            embedding = vector_db.embed_text(test_content)

            if embedding and len(embedding) > 0:
                response_time = (time.time() - start_time) * 1000

                return HealthCheck(
                    name="vector_db",
                    status=HealthStatus.HEALTHY,
                    message="Vector database is accessible and functioning",
                    response_time_ms=response_time,
                    last_checked=datetime.now(timezone.utc),
                    details={
                        "embedding_dimension": len(embedding),
                        "embedding_generation_successful": True,
                    },
                    critical=False,
                )
            else:
                return HealthCheck(
                    name="vector_db",
                    status=HealthStatus.UNHEALTHY,
                    message="Vector database embedding generation failed",
                    response_time_ms=(time.time() - start_time) * 1000,
                    last_checked=datetime.now(timezone.utc),
                    details={"embedding_result": embedding},
                    critical=False,
                )

        except Exception as e:
            return HealthCheck(
                name="vector_db",
                status=HealthStatus.UNHEALTHY,
                message=f"Vector database check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=datetime.now(timezone.utc),
                details={"error": str(e), "error_type": type(e).__name__},
                critical=False,
            )

    def _check_memory_manager(self) -> HealthCheck:
        """Check memory manager functionality."""
        start_time = time.time()

        try:
            from . import memory_manager

            # Test memory manager operations
            test_user_id = "health_check_user"
            test_content = "This is a test memory for health check"

            # Test memory addition
            memory_id = memory_manager.add_session(test_user_id, test_content)

            if memory_id:
                # Test memory retrieval
                stats = memory_manager.get_memory_stats(test_user_id)

                response_time = (time.time() - start_time) * 1000

                return HealthCheck(
                    name="memory_manager",
                    status=HealthStatus.HEALTHY,
                    message="Memory manager is functioning correctly",
                    response_time_ms=response_time,
                    last_checked=datetime.now(timezone.utc),
                    details={
                        "memory_id": memory_id,
                        "user_stats": stats,
                        "operations_successful": True,
                    },
                    critical=True,
                )
            else:
                return HealthCheck(
                    name="memory_manager",
                    status=HealthStatus.UNHEALTHY,
                    message="Memory manager failed to create memory",
                    response_time_ms=(time.time() - start_time) * 1000,
                    last_checked=datetime.now(timezone.utc),
                    details={"memory_id": memory_id},
                    critical=True,
                )

        except Exception as e:
            return HealthCheck(
                name="memory_manager",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory manager check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=datetime.now(timezone.utc),
                details={"error": str(e), "error_type": type(e).__name__},
                critical=True,
            )

    def _check_system_resources(self) -> HealthCheck:
        """Check system resource usage."""
        start_time = time.time()

        try:
            import psutil

            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            response_time = (time.time() - start_time) * 1000

            # Determine status based on resource usage
            status = HealthStatus.HEALTHY
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = HealthStatus.CRITICAL
            elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 70:
                status = HealthStatus.DEGRADED

            return HealthCheck(
                name="system_resources",
                status=status,
                message=f"System resources: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%",
                response_time_ms=response_time,
                last_checked=datetime.now(timezone.utc),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                },
                critical=False,
            )

        except ImportError:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message="psutil not available for system resource monitoring",
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=datetime.now(timezone.utc),
                details={"error": "psutil not installed"},
                critical=False,
            )
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"System resource check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=datetime.now(timezone.utc),
                details={"error": str(e), "error_type": type(e).__name__},
                critical=False,
            )

    def _check_external_services(self) -> HealthCheck:
        """Check external service dependencies."""
        start_time = time.time()

        try:
            import requests

            # Check external services
            services_status = {}
            overall_healthy = True

            # Check OpenAI API (if configured)
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    response = requests.get(
                        "https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {openai_key}"},
                        timeout=5,
                    )
                    services_status["openai"] = (
                        "healthy" if response.status_code == 200 else "unhealthy"
                    )
                except Exception as e:
                    services_status["openai"] = f"error: {str(e)}"
                    overall_healthy = False

            # Check other external services as needed
            # Add more service checks here

            response_time = (time.time() - start_time) * 1000

            status = HealthStatus.HEALTHY if overall_healthy else HealthStatus.DEGRADED

            return HealthCheck(
                name="external_services",
                status=status,
                message=f"External services status: {services_status}",
                response_time_ms=response_time,
                last_checked=datetime.now(timezone.utc),
                details={"services": services_status},
                critical=False,
            )

        except ImportError:
            return HealthCheck(
                name="external_services",
                status=HealthStatus.UNKNOWN,
                message="requests not available for external service monitoring",
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=datetime.now(timezone.utc),
                details={"error": "requests not installed"},
                critical=False,
            )
        except Exception as e:
            return HealthCheck(
                name="external_services",
                status=HealthStatus.UNHEALTHY,
                message=f"External service check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=datetime.now(timezone.utc),
                details={"error": str(e), "error_type": type(e).__name__},
                critical=False,
            )

    def _run_all_checks(self) -> None:
        """Run all registered health checks."""
        with self.lock:
            for check_name in self.registry.list_checks():
                try:
                    check_func = self.registry.get_check(check_name)
                    if check_func:
                        result = check_func()
                        self.health_cache[check_name] = result
                except Exception as e:
                    logger.error(f"Health check {check_name} failed: {e}")
                    self.health_cache[check_name] = HealthCheck(
                        name=check_name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {str(e)}",
                        response_time_ms=0,
                        last_checked=datetime.now(timezone.utc),
                        details={"error": str(e), "error_type": type(e).__name__},
                        critical=True,
                    )

    def get_health_status(self) -> SystemHealth:
        """Get overall system health status."""
        with self.lock:
            # Run checks if cache is empty or stale
            if not self.health_cache:
                self._run_all_checks()

            # Determine overall status
            overall_status = HealthStatus.HEALTHY
            critical_failures = 0
            total_checks = len(self.health_cache)

            for check in self.health_cache.values():
                if check.status == HealthStatus.UNHEALTHY:
                    if check.critical:
                        critical_failures += 1
                    overall_status = HealthStatus.UNHEALTHY
                elif (
                    check.status == HealthStatus.DEGRADED
                    and overall_status == HealthStatus.HEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED

            # Group checks by component
            components = {}
            for check in self.health_cache.values():
                component_name = check.name.split("_")[0]  # Simple grouping
                if component_name not in components:
                    components[component_name] = ComponentHealth(
                        name=component_name,
                        status=check.status,
                        checks=[check],
                        last_updated=check.last_checked,
                        uptime=time.time() - self.start_time,
                    )
                else:
                    components[component_name].checks.append(check)
                    # Update component status based on worst check status
                    if check.status == HealthStatus.UNHEALTHY:
                        components[component_name].status = HealthStatus.UNHEALTHY
                    elif (
                        check.status == HealthStatus.DEGRADED
                        and components[component_name].status == HealthStatus.HEALTHY
                    ):
                        components[component_name].status = HealthStatus.DEGRADED

            # Create summary
            summary = {
                "total_checks": total_checks,
                "healthy_checks": sum(
                    1
                    for c in self.health_cache.values()
                    if c.status == HealthStatus.HEALTHY
                ),
                "degraded_checks": sum(
                    1
                    for c in self.health_cache.values()
                    if c.status == HealthStatus.DEGRADED
                ),
                "unhealthy_checks": sum(
                    1
                    for c in self.health_cache.values()
                    if c.status == HealthStatus.UNHEALTHY
                ),
                "critical_failures": critical_failures,
                "uptime_seconds": time.time() - self.start_time,
            }

            return SystemHealth(
                overall_status=overall_status,
                components=components,
                uptime=time.time() - self.start_time,
                version=self.version,
                last_updated=datetime.now(timezone.utc),
                summary=summary,
            )

    def get_check_status(self, check_name: str) -> Optional[HealthCheck]:
        """Get status of a specific health check."""
        with self.lock:
            return self.health_cache.get(check_name)

    def run_check(self, check_name: str) -> Optional[HealthCheck]:
        """Run a specific health check immediately."""
        check_func = self.registry.get_check(check_name)
        if check_func:
            try:
                result = check_func()
                with self.lock:
                    self.health_cache[check_name] = result
                return result
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                return None
        return None

    def register_check(
        self, name: str, check_func: Callable, config: Dict[str, Any] = None
    ) -> None:
        """Register a new health check."""
        self.registry.register(name, check_func, config)

    def shutdown(self) -> None:
        """Shutdown the health monitor."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitor shutdown complete")


# Global health monitor instance
_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def get_health_status() -> Dict[str, Any]:
    """Get health status (compatibility function)."""
    monitor = get_health_monitor()
    system_health = monitor.get_health_status()
    return asdict(system_health)


# Import os for environment variables
import os
