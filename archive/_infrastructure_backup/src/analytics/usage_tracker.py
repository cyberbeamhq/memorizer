"""
Usage Tracker Module
Real-time tracking of memory usage events and metrics.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of usage events."""
    MEMORY_CREATED = "memory_created"
    MEMORY_ACCESSED = "memory_accessed"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_COMPRESSED = "memory_compressed"
    SEARCH_PERFORMED = "search_performed"
    BATCH_OPERATION = "batch_operation"


@dataclass
class UsageEvent:
    """Represents a single usage event."""
    event_type: EventType
    user_id: str
    memory_id: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]
    duration_ms: Optional[float] = None
    size_bytes: Optional[int] = None


@dataclass
class UsageMetrics:
    """Aggregated usage metrics."""
    user_id: str
    time_period: timedelta
    total_events: int
    events_by_type: Dict[EventType, int]
    total_size_processed: int
    avg_response_time_ms: float
    peak_usage_time: Optional[datetime]
    error_rate: float


class UsageTracker:
    """Real-time usage tracking and metrics collection."""

    def __init__(self, max_events: int = 10000, retention_hours: int = 24):
        """Initialize usage tracker."""
        self.max_events = max_events
        self.retention_hours = retention_hours
        self._events = deque(maxlen=max_events)
        self._user_metrics = defaultdict(lambda: defaultdict(int))
        self._lock = threading.RLock()
        self._start_time = datetime.now()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_events, daemon=True)
        self._cleanup_thread.start()

        logger.info(f"Usage tracker initialized with {max_events} max events, {retention_hours}h retention")

    def track_event(
        self,
        event_type: EventType,
        user_id: str,
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        size_bytes: Optional[int] = None
    ):
        """Track a usage event."""
        event = UsageEvent(
            event_type=event_type,
            user_id=user_id,
            memory_id=memory_id,
            timestamp=datetime.now(),
            metadata=metadata or {},
            duration_ms=duration_ms,
            size_bytes=size_bytes
        )

        with self._lock:
            self._events.append(event)
            self._update_user_metrics(event)

        logger.debug(f"Tracked event: {event_type.value} for user {user_id}")

    def track_memory_creation(self, user_id: str, memory_id: str, content_size: int, metadata: Optional[Dict] = None):
        """Track memory creation event."""
        self.track_event(
            EventType.MEMORY_CREATED,
            user_id,
            memory_id,
            metadata,
            size_bytes=content_size
        )

    def track_memory_access(self, user_id: str, memory_id: str, response_time_ms: float, metadata: Optional[Dict] = None):
        """Track memory access event."""
        self.track_event(
            EventType.MEMORY_ACCESSED,
            user_id,
            memory_id,
            metadata,
            duration_ms=response_time_ms
        )

    def track_search(self, user_id: str, query: str, results_count: int, response_time_ms: float):
        """Track search operation."""
        self.track_event(
            EventType.SEARCH_PERFORMED,
            user_id,
            metadata={"query_length": len(query), "results_count": results_count},
            duration_ms=response_time_ms
        )

    def track_compression(self, user_id: str, memory_id: str, original_size: int, compressed_size: int):
        """Track memory compression event."""
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        self.track_event(
            EventType.MEMORY_COMPRESSED,
            user_id,
            memory_id,
            metadata={
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "savings_bytes": original_size - compressed_size
            },
            size_bytes=compressed_size
        )

    def get_user_metrics(self, user_id: str, hours: int = 24) -> UsageMetrics:
        """Get aggregated metrics for a user."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        user_events = []

        with self._lock:
            for event in self._events:
                if event.user_id == user_id and event.timestamp >= cutoff_time:
                    user_events.append(event)

        if not user_events:
            return UsageMetrics(
                user_id=user_id,
                time_period=timedelta(hours=hours),
                total_events=0,
                events_by_type={},
                total_size_processed=0,
                avg_response_time_ms=0,
                peak_usage_time=None,
                error_rate=0
            )

        # Aggregate metrics
        events_by_type = defaultdict(int)
        total_size = 0
        response_times = []
        hourly_counts = defaultdict(int)

        for event in user_events:
            events_by_type[event.event_type] += 1

            if event.size_bytes:
                total_size += event.size_bytes

            if event.duration_ms:
                response_times.append(event.duration_ms)

            # Track hourly usage for peak detection
            hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1

        # Find peak usage time
        peak_usage_time = max(hourly_counts.keys(), key=lambda k: hourly_counts[k]) if hourly_counts else None

        return UsageMetrics(
            user_id=user_id,
            time_period=timedelta(hours=hours),
            total_events=len(user_events),
            events_by_type=dict(events_by_type),
            total_size_processed=total_size,
            avg_response_time_ms=sum(response_times) / len(response_times) if response_times else 0,
            peak_usage_time=peak_usage_time,
            error_rate=0  # Would be calculated from error events
        )

    def get_system_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get system-wide metrics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = []

        with self._lock:
            for event in self._events:
                if event.timestamp >= cutoff_time:
                    recent_events.append(event)

        if not recent_events:
            return {
                "total_events": 0,
                "unique_users": 0,
                "events_per_hour": 0,
                "avg_response_time_ms": 0,
                "total_size_processed": 0
            }

        # Aggregate system metrics
        unique_users = set(event.user_id for event in recent_events)
        total_size = sum(event.size_bytes for event in recent_events if event.size_bytes)
        response_times = [event.duration_ms for event in recent_events if event.duration_ms]

        events_per_hour = len(recent_events) / hours

        return {
            "total_events": len(recent_events),
            "unique_users": len(unique_users),
            "events_per_hour": events_per_hour,
            "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
            "total_size_processed": total_size,
            "events_by_type": self._aggregate_events_by_type(recent_events),
            "top_users": self._get_top_users(recent_events)
        }

    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics."""
        with self._lock:
            total_events = len(self._events)
            uptime = datetime.now() - self._start_time

            # Recent activity (last 5 minutes)
            recent_cutoff = datetime.now() - timedelta(minutes=5)
            recent_events = [e for e in self._events if e.timestamp >= recent_cutoff]

            return {
                "uptime_seconds": uptime.total_seconds(),
                "total_events_tracked": total_events,
                "events_last_5_minutes": len(recent_events),
                "memory_buffer_usage": f"{total_events}/{self.max_events}",
                "active_users_last_hour": self._get_active_users_count(hours=1),
                "events_per_second": len(recent_events) / 300 if recent_events else 0  # 5 minutes = 300 seconds
            }

    def _update_user_metrics(self, event: UsageEvent):
        """Update real-time user metrics."""
        user_id = event.user_id
        self._user_metrics[user_id]["total_events"] += 1
        self._user_metrics[user_id][f"event_{event.event_type.value}"] += 1

        if event.size_bytes:
            self._user_metrics[user_id]["total_bytes"] += event.size_bytes

        if event.duration_ms:
            # Update rolling average for response time
            current_avg = self._user_metrics[user_id].get("avg_response_time", 0)
            current_count = self._user_metrics[user_id].get("response_time_count", 0)
            new_avg = (current_avg * current_count + event.duration_ms) / (current_count + 1)
            self._user_metrics[user_id]["avg_response_time"] = new_avg
            self._user_metrics[user_id]["response_time_count"] = current_count + 1

    def _cleanup_old_events(self):
        """Background thread to clean up old events."""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

                with self._lock:
                    # Remove old events
                    old_count = len(self._events)
                    self._events = deque([e for e in self._events if e.timestamp >= cutoff_time], maxlen=self.max_events)
                    removed_count = old_count - len(self._events)

                    if removed_count > 0:
                        logger.info(f"Cleaned up {removed_count} old events")

            except Exception as e:
                logger.error(f"Error during event cleanup: {e}")

    def _aggregate_events_by_type(self, events: List[UsageEvent]) -> Dict[str, int]:
        """Aggregate events by type."""
        counts = defaultdict(int)
        for event in events:
            counts[event.event_type.value] += 1
        return dict(counts)

    def _get_top_users(self, events: List[UsageEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top users by activity."""
        user_counts = defaultdict(int)
        for event in events:
            user_counts[event.user_id] += 1

        top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{"user_id": user_id, "event_count": count} for user_id, count in top_users]

    def _get_active_users_count(self, hours: int = 1) -> int:
        """Get count of active users in the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        active_users = set()

        with self._lock:
            for event in self._events:
                if event.timestamp >= cutoff_time:
                    active_users.add(event.user_id)

        return len(active_users)

    def export_events(self, user_id: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Export events for analysis."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        events_to_export = []

        with self._lock:
            for event in self._events:
                if event.timestamp >= cutoff_time:
                    if user_id is None or event.user_id == user_id:
                        events_to_export.append({
                            "event_type": event.event_type.value,
                            "user_id": event.user_id,
                            "memory_id": event.memory_id,
                            "timestamp": event.timestamp.isoformat(),
                            "metadata": event.metadata,
                            "duration_ms": event.duration_ms,
                            "size_bytes": event.size_bytes
                        })

        return events_to_export

    def clear_user_data(self, user_id: str):
        """Clear all tracking data for a specific user."""
        with self._lock:
            # Remove events
            original_count = len(self._events)
            self._events = deque([e for e in self._events if e.user_id != user_id], maxlen=self.max_events)
            removed_events = original_count - len(self._events)

            # Clear user metrics
            if user_id in self._user_metrics:
                del self._user_metrics[user_id]

        logger.info(f"Cleared {removed_events} events and metrics for user {user_id}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get tracker health status."""
        with self._lock:
            return {
                "status": "healthy",
                "total_events": len(self._events),
                "buffer_utilization": len(self._events) / self.max_events,
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
                "cleanup_thread_alive": self._cleanup_thread.is_alive(),
                "tracked_users": len(self._user_metrics)
            }