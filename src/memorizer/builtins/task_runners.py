"""
Built-in Task Runner Components
Provides default task execution implementations for the Memorizer framework.
"""

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
from abc import ABC, abstractmethod
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class BaseTaskRunner(ABC):
    """Abstract base class for task runners."""

    @abstractmethod
    def submit_task(self, task_func: Callable, *args, **kwargs) -> str:
        """Submit a task for execution."""
        pass

    @abstractmethod
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        pass

    @abstractmethod
    def shutdown(self):
        """Shutdown the task runner."""
        pass

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the task runner."""
        return {"status": "healthy", "type": self.__class__.__name__}


class ThreadTaskRunner(BaseTaskRunner):
    """Thread-based task runner using ThreadPoolExecutor."""

    def __init__(self, max_workers: int = 4, **kwargs):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def submit_task(self, task_func: Callable, *args, **kwargs) -> str:
        """Submit a task for execution in thread pool."""
        with self._lock:
            task_id = str(uuid.uuid4())

            try:
                future = self._executor.submit(task_func, *args, **kwargs)

                self._tasks[task_id] = {
                    "id": task_id,
                    "status": "pending",
                    "future": future,
                    "submitted_at": time.time(),
                    "started_at": None,
                    "completed_at": None,
                    "result": None,
                    "error": None
                }

                # Update status asynchronously
                def update_status():
                    try:
                        self._tasks[task_id]["status"] = "running"
                        self._tasks[task_id]["started_at"] = time.time()

                        result = future.result()

                        self._tasks[task_id]["status"] = "completed"
                        self._tasks[task_id]["completed_at"] = time.time()
                        self._tasks[task_id]["result"] = result

                    except Exception as e:
                        self._tasks[task_id]["status"] = "failed"
                        self._tasks[task_id]["completed_at"] = time.time()
                        self._tasks[task_id]["error"] = str(e)

                future.add_done_callback(lambda f: update_status())

                logger.debug(f"Submitted task {task_id}")
                return task_id

            except Exception as e:
                logger.error(f"Failed to submit task: {e}")
                raise

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        with self._lock:
            if task_id not in self._tasks:
                return {"error": "Task not found"}

            task = self._tasks[task_id].copy()
            # Remove the future object from the response
            task.pop("future", None)
            return task

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get status of all tasks."""
        with self._lock:
            tasks = []
            for task in self._tasks.values():
                task_copy = task.copy()
                task_copy.pop("future", None)
                tasks.append(task_copy)
            return tasks

    def shutdown(self):
        """Shutdown the thread pool."""
        try:
            self._executor.shutdown(wait=True)
            logger.info("Thread task runner shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


class QueueTaskRunner(BaseTaskRunner):
    """Simple queue-based task runner."""

    def __init__(self, num_workers: int = 2, **kwargs):
        self.num_workers = num_workers
        self._task_queue = Queue()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._workers: List[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()

        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)

    def submit_task(self, task_func: Callable, *args, **kwargs) -> str:
        """Submit a task to the queue."""
        task_id = str(uuid.uuid4())

        with self._lock:
            self._tasks[task_id] = {
                "id": task_id,
                "status": "pending",
                "submitted_at": time.time(),
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None
            }

        # Add task to queue
        self._task_queue.put({
            "id": task_id,
            "func": task_func,
            "args": args,
            "kwargs": kwargs
        })

        logger.debug(f"Queued task {task_id}")
        return task_id

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        with self._lock:
            if task_id not in self._tasks:
                return {"error": "Task not found"}
            return self._tasks[task_id].copy()

    def _worker_loop(self):
        """Worker thread loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                task = self._task_queue.get(timeout=1.0)

                task_id = task["id"]

                with self._lock:
                    if task_id in self._tasks:
                        self._tasks[task_id]["status"] = "running"
                        self._tasks[task_id]["started_at"] = time.time()

                try:
                    # Execute the task
                    result = task["func"](*task["args"], **task["kwargs"])

                    with self._lock:
                        if task_id in self._tasks:
                            self._tasks[task_id]["status"] = "completed"
                            self._tasks[task_id]["completed_at"] = time.time()
                            self._tasks[task_id]["result"] = result

                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")

                    with self._lock:
                        if task_id in self._tasks:
                            self._tasks[task_id]["status"] = "failed"
                            self._tasks[task_id]["completed_at"] = time.time()
                            self._tasks[task_id]["error"] = str(e)

                finally:
                    self._task_queue.task_done()

            except Empty:
                # Timeout waiting for task, continue loop
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def shutdown(self):
        """Shutdown the task runner."""
        self._shutdown_event.set()

        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)

        logger.info("Queue task runner shutdown complete")


class CeleryTaskRunner(BaseTaskRunner):
    """Celery-based task runner (mock implementation)."""

    def __init__(self, broker_url: str = "", **kwargs):
        self.broker_url = broker_url
        logger.warning("Celery task runner not fully implemented, using thread fallback")
        self._fallback = ThreadTaskRunner(**kwargs)

    def submit_task(self, task_func: Callable, *args, **kwargs) -> str:
        """Submit task using Celery (fallback to threads)."""
        return self._fallback.submit_task(task_func, *args, **kwargs)

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status from Celery (fallback to threads)."""
        return self._fallback.get_task_status(task_id)

    def shutdown(self):
        """Shutdown Celery worker (fallback to threads)."""
        self._fallback.shutdown()


class RQTaskRunner(BaseTaskRunner):
    """RQ (Redis Queue) based task runner (mock implementation)."""

    def __init__(self, redis_url: str = "", **kwargs):
        self.redis_url = redis_url
        logger.warning("RQ task runner not fully implemented, using thread fallback")
        self._fallback = ThreadTaskRunner(**kwargs)

    def submit_task(self, task_func: Callable, *args, **kwargs) -> str:
        """Submit task using RQ (fallback to threads)."""
        return self._fallback.submit_task(task_func, *args, **kwargs)

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status from RQ (fallback to threads)."""
        return self._fallback.get_task_status(task_id)

    def shutdown(self):
        """Shutdown RQ worker (fallback to threads)."""
        self._fallback.shutdown()


__all__ = [
    "BaseTaskRunner",
    "ThreadTaskRunner",
    "QueueTaskRunner",
    "CeleryTaskRunner",
    "RQTaskRunner",
]