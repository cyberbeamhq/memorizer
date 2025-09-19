"""
automated_testing.py
Automated testing system for the Memorizer framework.
Provides comprehensive test suites for health checks and system validation.
"""

import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Test status enumeration."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Test result data structure."""

    test_name: str
    status: TestStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class TestSuite:
    """Test suite data structure."""

    name: str
    tests: List[TestResult]
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    duration_ms: float
    timestamp: datetime


class TestRunner:
    """Automated test runner for system validation."""

    def __init__(self):
        self.test_functions: Dict[str, Callable] = {}
        self.test_configs: Dict[str, Dict[str, Any]] = {}

    def register_test(
        self, name: str, test_func: Callable, config: Dict[str, Any] = None
    ) -> None:
        """Register a test function."""
        self.test_functions[name] = test_func
        self.test_configs[name] = config or {}
        logger.info(f"Registered test: {name}")

    def run_test(self, test_name: str) -> TestResult:
        """Run a single test."""
        if test_name not in self.test_functions:
            return TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Test {test_name} not found",
                duration_ms=0,
                timestamp=datetime.now(timezone.utc),
                details={"error": "test_not_found"},
            )

        test_func = self.test_functions[test_name]
        config = self.test_configs[test_name]

        start_time = time.time()

        try:
            # Check if test should be skipped
            if config.get("skip", False):
                return TestResult(
                    test_name=test_name,
                    status=TestStatus.SKIPPED,
                    message=f"Test {test_name} skipped",
                    duration_ms=0,
                    timestamp=datetime.now(timezone.utc),
                    details={
                        "skip_reason": config.get("skip_reason", "No reason provided")
                    },
                )

            # Run the test
            result = test_func()

            duration_ms = (time.time() - start_time) * 1000

            if result is True:
                return TestResult(
                    test_name=test_name,
                    status=TestStatus.PASSED,
                    message=f"Test {test_name} passed",
                    duration_ms=duration_ms,
                    timestamp=datetime.now(timezone.utc),
                    details={"result": result},
                )
            elif result is False:
                return TestResult(
                    test_name=test_name,
                    status=TestStatus.FAILED,
                    message=f"Test {test_name} failed",
                    duration_ms=duration_ms,
                    timestamp=datetime.now(timezone.utc),
                    details={"result": result},
                )
            else:
                # Test returned a result object
                return TestResult(
                    test_name=test_name,
                    status=(
                        TestStatus.PASSED
                        if result.get("passed", False)
                        else TestStatus.FAILED
                    ),
                    message=result.get("message", f"Test {test_name} completed"),
                    duration_ms=duration_ms,
                    timestamp=datetime.now(timezone.utc),
                    details=result.get("details", {}),
                    error=result.get("error"),
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                message=f"Test {test_name} encountered an error",
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc),
                details={"error_type": type(e).__name__},
                error=str(e),
            )

    def run_test_suite(
        self, suite_name: str, test_names: List[str] = None
    ) -> TestSuite:
        """Run a test suite."""
        start_time = time.time()

        if test_names is None:
            test_names = list(self.test_functions.keys())

        results = []
        for test_name in test_names:
            result = self.run_test(test_name)
            results.append(result)

        duration_ms = (time.time() - start_time) * 1000

        # Count results
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        error = sum(1 for r in results if r.status == TestStatus.ERROR)

        return TestSuite(
            name=suite_name,
            tests=results,
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=skipped,
            error_tests=error,
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc),
        )


class AutomatedTestingSystem:
    """Main automated testing system."""

    def __init__(self):
        self.test_runner = TestRunner()
        self._register_default_tests()

    def _register_default_tests(self) -> None:
        """Register default test functions."""
        # Database tests
        self.test_runner.register_test(
            "database_connection", self._test_database_connection
        )
        self.test_runner.register_test(
            "database_operations", self._test_database_operations
        )
        self.test_runner.register_test(
            "database_performance", self._test_database_performance
        )

        # Cache tests
        self.test_runner.register_test("cache_connection", self._test_cache_connection)
        self.test_runner.register_test("cache_operations", self._test_cache_operations)

        # Memory manager tests
        self.test_runner.register_test(
            "memory_manager_basic", self._test_memory_manager_basic
        )
        self.test_runner.register_test(
            "memory_manager_operations", self._test_memory_manager_operations
        )

        # Vector database tests
        self.test_runner.register_test(
            "vector_db_embedding", self._test_vector_db_embedding
        )
        self.test_runner.register_test(
            "vector_db_operations", self._test_vector_db_operations
        )

        # API tests
        self.test_runner.register_test(
            "api_health_endpoints", self._test_api_health_endpoints
        )
        self.test_runner.register_test(
            "api_memory_endpoints", self._test_api_memory_endpoints
        )

        # Integration tests
        self.test_runner.register_test(
            "end_to_end_workflow", self._test_end_to_end_workflow
        )
        self.test_runner.register_test(
            "performance_benchmarks", self._test_performance_benchmarks
        )

    def _test_database_connection(self) -> Union[bool, Dict[str, Any]]:
        """Test database connection."""
        try:
            from . import db

            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()

                    if result and result[0] == 1:
                        return {
                            "passed": True,
                            "message": "Database connection successful",
                            "details": {"test_query_result": result[0]},
                        }
                    else:
                        return {
                            "passed": False,
                            "message": "Database test query failed",
                            "details": {"test_query_result": result},
                        }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Database connection failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def _test_database_operations(self) -> Union[bool, Dict[str, Any]]:
        """Test database CRUD operations."""
        try:
            from . import db

            test_user_id = "test_user_automated_testing"
            test_content = "Automated testing content"
            test_metadata = {"test": True, "automated": True}

            # Test insert
            memory_id = db.insert_session(test_user_id, test_content, test_metadata)

            if not memory_id:
                return {
                    "passed": False,
                    "message": "Database insert operation failed",
                    "details": {"memory_id": memory_id},
                }

            # Test fetch
            memories = db.fetch_memories(test_user_id, limit=1)

            if not memories:
                return {
                    "passed": False,
                    "message": "Database fetch operation failed",
                    "details": {"memories_count": len(memories)},
                }

            # Test search
            search_results = db.search_memories(test_user_id, "automated", limit=1)

            # Cleanup
            try:
                db.delete_memory(test_user_id, int(memory_id))
            except:
                pass  # Cleanup failure is not critical for test

            return {
                "passed": True,
                "message": "Database operations successful",
                "details": {
                    "insert_successful": bool(memory_id),
                    "fetch_successful": len(memories) > 0,
                    "search_successful": len(search_results) > 0,
                },
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Database operations test failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def _test_database_performance(self) -> Union[bool, Dict[str, Any]]:
        """Test database performance."""
        try:
            from . import db

            test_user_id = "test_user_performance"
            start_time = time.time()

            # Test multiple operations
            operations = []
            for i in range(10):
                content = f"Performance test content {i}"
                memory_id = db.insert_session(test_user_id, content)
                operations.append(memory_id)

            insert_time = time.time() - start_time

            # Test fetch performance
            start_time = time.time()
            memories = db.fetch_memories(test_user_id, limit=10)
            fetch_time = time.time() - start_time

            # Cleanup
            for memory_id in operations:
                try:
                    db.delete_memory(test_user_id, int(memory_id))
                except:
                    pass

            # Performance thresholds
            insert_threshold = 1.0  # 1 second for 10 inserts
            fetch_threshold = 0.5  # 0.5 seconds for fetch

            passed = insert_time < insert_threshold and fetch_time < fetch_threshold

            return {
                "passed": passed,
                "message": f"Database performance test {'passed' if passed else 'failed'}",
                "details": {
                    "insert_time_seconds": insert_time,
                    "fetch_time_seconds": fetch_time,
                    "insert_threshold": insert_threshold,
                    "fetch_threshold": fetch_threshold,
                    "operations_count": len(operations),
                },
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Database performance test failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def _test_cache_connection(self) -> Union[bool, Dict[str, Any]]:
        """Test cache connection."""
        try:
            from .cache import get_cache_manager

            cache = get_cache_manager()
            stats = cache.get_stats()

            return {
                "passed": True,
                "message": "Cache connection successful",
                "details": {"cache_stats": stats},
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Cache connection failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def _test_cache_operations(self) -> Union[bool, Dict[str, Any]]:
        """Test cache operations."""
        try:
            from .cache import get_cache_manager

            cache = get_cache_manager()
            test_key = "automated_test_key"
            test_value = "automated_test_value"

            # Test set
            set_result = cache.set("test", test_key, test_value, ttl=60)

            # Test get
            get_result = cache.get("test", test_key)

            # Test delete
            cache.delete("test", test_key)

            passed = set_result and get_result == test_value

            return {
                "passed": passed,
                "message": f"Cache operations test {'passed' if passed else 'failed'}",
                "details": {
                    "set_successful": set_result,
                    "get_successful": get_result == test_value,
                    "expected_value": test_value,
                    "actual_value": get_result,
                },
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Cache operations test failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def _test_memory_manager_basic(self) -> Union[bool, Dict[str, Any]]:
        """Test basic memory manager functionality."""
        try:
            from . import memory_manager

            test_user_id = "test_user_memory_manager"
            test_content = "Automated testing memory content"

            # Test add session
            memory_id = memory_manager.add_session(test_user_id, test_content)

            if not memory_id:
                return {
                    "passed": False,
                    "message": "Memory manager add_session failed",
                    "details": {"memory_id": memory_id},
                }

            # Test get stats
            stats = memory_manager.get_memory_stats(test_user_id)

            return {
                "passed": True,
                "message": "Memory manager basic test passed",
                "details": {"memory_id": memory_id, "user_stats": stats},
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Memory manager basic test failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def _test_memory_manager_operations(self) -> Union[bool, Dict[str, Any]]:
        """Test memory manager operations."""
        try:
            from . import memory_manager

            test_user_id = "test_user_memory_operations"
            test_content = "Test content for memory operations"

            # Test add session
            memory_id = memory_manager.add_session(test_user_id, test_content)

            # Test get context
            context = memory_manager.get_context(test_user_id, "test", max_items=5)

            # Test move between tiers (this might take time)
            moved = memory_manager.move_memory_between_tiers(test_user_id)

            return {
                "passed": True,
                "message": "Memory manager operations test passed",
                "details": {
                    "memory_id": memory_id,
                    "context_results": len(context),
                    "moved_memories": moved,
                },
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Memory manager operations test failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def _test_vector_db_embedding(self) -> Union[bool, Dict[str, Any]]:
        """Test vector database embedding generation."""
        try:
            from . import vector_db

            test_content = "Test content for embedding generation"

            # Test embedding generation
            embedding = vector_db.embed_text(test_content)

            if not embedding or len(embedding) == 0:
                return {
                    "passed": False,
                    "message": "Vector database embedding generation failed",
                    "details": {"embedding": embedding},
                }

            return {
                "passed": True,
                "message": "Vector database embedding test passed",
                "details": {
                    "embedding_dimension": len(embedding),
                    "embedding_type": type(embedding).__name__,
                },
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Vector database embedding test failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def _test_vector_db_operations(self) -> Union[bool, Dict[str, Any]]:
        """Test vector database operations."""
        try:
            from . import vector_db

            test_user_id = "test_user_vector_operations"
            test_content = "Test content for vector operations"

            # Test embedding and insertion
            embedding = vector_db.embed_text(test_content)
            memory_id = "test_memory_vector_ops"

            # Test insert embedding
            vector_db.insert_embedding(memory_id, test_user_id, test_content)

            # Test query embeddings
            results = vector_db.query_embeddings(test_user_id, test_content, top_k=5)

            return {
                "passed": True,
                "message": "Vector database operations test passed",
                "details": {
                    "embedding_dimension": len(embedding),
                    "query_results": len(results),
                },
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Vector database operations test failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def _test_api_health_endpoints(self) -> Union[bool, Dict[str, Any]]:
        """Test API health endpoints."""
        try:
            import requests

            base_url = "http://localhost:8000"  # This should be configurable

            # Test health endpoint
            response = requests.get(f"{base_url}/health", timeout=5)

            if response.status_code != 200:
                return {
                    "passed": False,
                    "message": f"Health endpoint returned status {response.status_code}",
                    "details": {
                        "status_code": response.status_code,
                        "response": response.text,
                    },
                }

            health_data = response.json()

            return {
                "passed": True,
                "message": "API health endpoints test passed",
                "details": {
                    "status_code": response.status_code,
                    "health_status": health_data.get("status"),
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                },
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"API health endpoints test failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def _test_api_memory_endpoints(self) -> Union[bool, Dict[str, Any]]:
        """Test API memory endpoints."""
        try:
            import requests

            base_url = "http://localhost:8000"  # This should be configurable

            # Test memory endpoints (this would require authentication in production)
            # For now, just test that endpoints exist
            endpoints_to_test = ["/memories", "/query", "/stats"]

            results = {}
            for endpoint in endpoints_to_test:
                try:
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                    results[endpoint] = {
                        "status_code": response.status_code,
                        "accessible": True,
                    }
                except Exception as e:
                    results[endpoint] = {"error": str(e), "accessible": False}

            # Consider test passed if at least some endpoints are accessible
            accessible_count = sum(
                1 for r in results.values() if r.get("accessible", False)
            )
            passed = accessible_count > 0

            return {
                "passed": passed,
                "message": f"API memory endpoints test {'passed' if passed else 'failed'}",
                "details": {
                    "endpoints_tested": len(endpoints_to_test),
                    "accessible_endpoints": accessible_count,
                    "results": results,
                },
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"API memory endpoints test failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def _test_end_to_end_workflow(self) -> Union[bool, Dict[str, Any]]:
        """Test end-to-end workflow."""
        try:
            from . import memory_manager

            test_user_id = "test_user_e2e"
            test_content = "End-to-end test content"

            # Complete workflow test
            # 1. Add memory
            memory_id = memory_manager.add_session(test_user_id, test_content)

            # 2. Query memory
            context = memory_manager.get_context(test_user_id, "test", max_items=5)

            # 3. Get stats
            stats = memory_manager.get_memory_stats(test_user_id)

            # 4. Move between tiers
            moved = memory_manager.move_memory_between_tiers(test_user_id)

            passed = bool(memory_id) and len(context) >= 0 and stats is not None

            return {
                "passed": passed,
                "message": f"End-to-end workflow test {'passed' if passed else 'failed'}",
                "details": {
                    "memory_id": memory_id,
                    "context_results": len(context),
                    "stats": stats,
                    "moved_memories": moved,
                },
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"End-to-end workflow test failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def _test_performance_benchmarks(self) -> Union[bool, Dict[str, Any]]:
        """Test performance benchmarks."""
        try:
            from . import memory_manager

            test_user_id = "test_user_performance_benchmark"

            # Benchmark memory operations
            start_time = time.time()

            # Add multiple memories
            memory_ids = []
            for i in range(10):
                content = f"Performance benchmark content {i}"
                memory_id = memory_manager.add_session(test_user_id, content)
                memory_ids.append(memory_id)

            add_time = time.time() - start_time

            # Benchmark query operations
            start_time = time.time()
            context = memory_manager.get_context(
                test_user_id, "benchmark", max_items=10
            )
            query_time = time.time() - start_time

            # Performance thresholds
            add_threshold = 2.0  # 2 seconds for 10 adds
            query_threshold = 1.0  # 1 second for query

            passed = add_time < add_threshold and query_time < query_threshold

            return {
                "passed": passed,
                "message": f"Performance benchmarks test {'passed' if passed else 'failed'}",
                "details": {
                    "add_time_seconds": add_time,
                    "query_time_seconds": query_time,
                    "add_threshold": add_threshold,
                    "query_threshold": query_threshold,
                    "memories_added": len(memory_ids),
                    "query_results": len(context),
                },
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Performance benchmarks test failed: {str(e)}",
                "error": str(e),
                "details": {"error_type": type(e).__name__},
            }

    def run_all_tests(self) -> TestSuite:
        """Run all registered tests."""
        return self.test_runner.run_test_suite("comprehensive_test_suite")

    def run_component_tests(self, component: str) -> TestSuite:
        """Run tests for a specific component."""
        component_tests = {
            "database": [
                "database_connection",
                "database_operations",
                "database_performance",
            ],
            "cache": ["cache_connection", "cache_operations"],
            "memory_manager": ["memory_manager_basic", "memory_manager_operations"],
            "vector_db": ["vector_db_embedding", "vector_db_operations"],
            "api": ["api_health_endpoints", "api_memory_endpoints"],
            "integration": ["end_to_end_workflow", "performance_benchmarks"],
        }

        test_names = component_tests.get(component, [])
        return self.test_runner.run_test_suite(f"{component}_test_suite", test_names)

    def register_test(
        self, name: str, test_func: Callable, config: Dict[str, Any] = None
    ) -> None:
        """Register a custom test."""
        self.test_runner.register_test(name, test_func, config)


# Global automated testing system instance
_automated_testing = None


def get_automated_testing() -> AutomatedTestingSystem:
    """Get global automated testing system instance."""
    global _automated_testing
    if _automated_testing is None:
        _automated_testing = AutomatedTestingSystem()
    return _automated_testing
