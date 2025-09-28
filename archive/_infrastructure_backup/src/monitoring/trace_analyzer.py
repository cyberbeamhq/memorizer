"""
Trace Analyzer Module
Advanced distributed tracing and request flow analysis.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)


class SpanType(Enum):
    """Types of spans in distributed tracing."""
    HTTP_REQUEST = "http_request"
    DATABASE_QUERY = "database_query"
    CACHE_OPERATION = "cache_operation"
    MEMORY_OPERATION = "memory_operation"
    LLM_CALL = "llm_call"
    COMPRESSION = "compression"
    SEARCH = "search"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    BACKGROUND_TASK = "background_task"


@dataclass
class TraceSpan:
    """Represents a single span in a distributed trace."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    span_type: SpanType

    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None

    # Context
    service_name: str = "memorizer"
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None

    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    # Status
    status: str = "ok"  # ok, error, timeout
    error_message: Optional[str] = None

    # Performance metrics
    cpu_time_ms: Optional[float] = None
    memory_bytes: Optional[int] = None
    network_bytes: Optional[int] = None


@dataclass
class Trace:
    """Represents a complete distributed trace."""
    trace_id: str
    root_span_id: str
    spans: List[TraceSpan] = field(default_factory=list)

    # Computed properties
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_ms: Optional[float] = None

    # Context
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    operation: Optional[str] = None

    # Statistics
    span_count: int = 0
    error_count: int = 0
    warning_count: int = 0


class TraceAnalyzer:
    """Analyzes distributed traces for performance and debugging."""

    def __init__(self, retention_hours: int = 24):
        """Initialize trace analyzer."""
        self.retention_hours = retention_hours

        # Storage
        self._traces: Dict[str, Trace] = {}
        self._active_spans: Dict[str, TraceSpan] = {}

        # Performance tracking
        self._span_metrics: Dict[str, List[float]] = {}
        self._error_patterns: Dict[str, int] = {}

        # Background cleanup
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        logger.info("Trace analyzer initialized")

    def start_span(
        self,
        operation_name: str,
        span_type: SpanType,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> TraceSpan:
        """Start a new span."""
        span_id = str(uuid.uuid4())
        trace_id = trace_id or str(uuid.uuid4())

        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            span_type=span_type,
            start_time=datetime.now(),
            user_id=user_id,
            tenant_id=tenant_id,
            tags=tags or {}
        )

        self._active_spans[span_id] = span

        # Create trace if it doesn't exist
        if trace_id not in self._traces:
            trace = Trace(
                trace_id=trace_id,
                root_span_id=span_id if not parent_span_id else "",
                user_id=user_id,
                tenant_id=tenant_id,
                operation=operation_name
            )
            self._traces[trace_id] = trace

        logger.debug(f"Started span {span_id} for operation {operation_name}")
        return span

    def finish_span(
        self,
        span_id: str,
        status: str = "ok",
        error_message: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        logs: Optional[List[Dict[str, Any]]] = None
    ):
        """Finish a span."""
        if span_id not in self._active_spans:
            logger.warning(f"Attempted to finish unknown span: {span_id}")
            return

        span = self._active_spans[span_id]
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        span.error_message = error_message

        if tags:
            span.tags.update(tags)
        if logs:
            span.logs.extend(logs)

        # Move to trace
        trace = self._traces.get(span.trace_id)
        if trace:
            trace.spans.append(span)
            self._update_trace_stats(trace, span)

        # Remove from active spans
        del self._active_spans[span_id]

        # Update performance metrics
        self._update_span_metrics(span)

        logger.debug(f"Finished span {span_id} with status {status}")

    def add_span_log(self, span_id: str, level: str, message: str, fields: Optional[Dict[str, Any]] = None):
        """Add a log entry to a span."""
        if span_id not in self._active_spans:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "fields": fields or {}
        }

        self._active_spans[span_id].logs.append(log_entry)

    def set_span_tag(self, span_id: str, key: str, value: str):
        """Set a tag on a span."""
        if span_id not in self._active_spans:
            return

        self._active_spans[span_id].tags[key] = value

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a complete trace."""
        return self._traces.get(trace_id)

    def find_traces(
        self,
        operation: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        has_errors: Optional[bool] = None,
        hours: int = 24,
        limit: int = 100
    ) -> List[Trace]:
        """Find traces matching criteria."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        matching_traces = []
        for trace in self._traces.values():
            # Skip traces outside time range
            if trace.start_time and trace.start_time < cutoff_time:
                continue

            # Apply filters
            if operation and trace.operation != operation:
                continue

            if user_id and trace.user_id != user_id:
                continue

            if tenant_id and trace.tenant_id != tenant_id:
                continue

            if min_duration_ms and (not trace.total_duration_ms or trace.total_duration_ms < min_duration_ms):
                continue

            if max_duration_ms and (not trace.total_duration_ms or trace.total_duration_ms > max_duration_ms):
                continue

            if has_errors is not None:
                has_trace_errors = trace.error_count > 0
                if has_errors != has_trace_errors:
                    continue

            matching_traces.append(trace)

        # Sort by start time (newest first) and limit
        matching_traces.sort(key=lambda t: t.start_time or datetime.min, reverse=True)
        return matching_traces[:limit]

    def analyze_performance_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance patterns across traces."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Collect spans for analysis
        spans_by_operation = {}
        error_patterns = {}
        slowest_operations = []

        for trace in self._traces.values():
            if not trace.start_time or trace.start_time < cutoff_time:
                continue

            for span in trace.spans:
                operation = span.operation_name

                # Group by operation
                if operation not in spans_by_operation:
                    spans_by_operation[operation] = []
                spans_by_operation[operation].append(span)

                # Track errors
                if span.status == "error":
                    error_key = f"{operation}:{span.error_message or 'unknown'}"
                    error_patterns[error_key] = error_patterns.get(error_key, 0) + 1

                # Track slow operations
                if span.duration_ms and span.duration_ms > 1000:  # > 1 second
                    slowest_operations.append({
                        "operation": operation,
                        "duration_ms": span.duration_ms,
                        "trace_id": span.trace_id,
                        "span_id": span.span_id
                    })

        # Calculate statistics per operation
        operation_stats = {}
        for operation, spans in spans_by_operation.items():
            durations = [s.duration_ms for s in spans if s.duration_ms]

            if durations:
                operation_stats[operation] = {
                    "count": len(spans),
                    "avg_duration_ms": sum(durations) / len(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations),
                    "p95_duration_ms": self._percentile(durations, 0.95),
                    "p99_duration_ms": self._percentile(durations, 0.99),
                    "error_rate": len([s for s in spans if s.status == "error"]) / len(spans)
                }

        # Sort slowest operations
        slowest_operations.sort(key=lambda x: x["duration_ms"], reverse=True)

        return {
            "time_range_hours": hours,
            "total_traces": len(self._traces),
            "operation_statistics": operation_stats,
            "error_patterns": dict(sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
            "slowest_operations": slowest_operations[:20],
            "performance_trends": self._calculate_performance_trends(hours)
        }

    def detect_performance_anomalies(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Detect performance anomalies in traces."""
        anomalies = []

        # Check for unusually slow operations
        for operation, durations in self._span_metrics.items():
            if len(durations) < 10:  # Need enough data
                continue

            avg_duration = sum(durations) / len(durations)
            recent_durations = durations[-10:]  # Last 10 measurements
            recent_avg = sum(recent_durations) / len(recent_durations)

            # Anomaly if recent average is 2x normal
            if recent_avg > avg_duration * 2:
                anomalies.append({
                    "type": "slow_operation",
                    "operation": operation,
                    "normal_avg_ms": avg_duration,
                    "recent_avg_ms": recent_avg,
                    "severity": "warning" if recent_avg < avg_duration * 3 else "critical"
                })

        # Check for error spikes
        cutoff_time = datetime.now() - timedelta(hours=1)  # Last hour
        recent_errors = {}

        for trace in self._traces.values():
            if not trace.start_time or trace.start_time < cutoff_time:
                continue

            for span in trace.spans:
                if span.status == "error":
                    operation = span.operation_name
                    recent_errors[operation] = recent_errors.get(operation, 0) + 1

        for operation, error_count in recent_errors.items():
            if error_count > 5:  # More than 5 errors in an hour
                anomalies.append({
                    "type": "error_spike",
                    "operation": operation,
                    "error_count": error_count,
                    "time_window": "1 hour",
                    "severity": "error"
                })

        return sorted(anomalies, key=lambda x: x.get("recent_avg_ms", 0), reverse=True)

    def generate_service_map(self, hours: int = 24) -> Dict[str, Any]:
        """Generate a service dependency map."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        services = set()
        connections = {}
        service_stats = {}

        for trace in self._traces.values():
            if not trace.start_time or trace.start_time < cutoff_time:
                continue

            # Track services and their connections
            span_by_service = {}
            for span in trace.spans:
                service = span.service_name
                services.add(service)

                if service not in span_by_service:
                    span_by_service[service] = []
                span_by_service[service].append(span)

                # Track service statistics
                if service not in service_stats:
                    service_stats[service] = {
                        "request_count": 0,
                        "error_count": 0,
                        "total_duration_ms": 0
                    }

                service_stats[service]["request_count"] += 1
                if span.status == "error":
                    service_stats[service]["error_count"] += 1
                if span.duration_ms:
                    service_stats[service]["total_duration_ms"] += span.duration_ms

            # Find connections between services
            for span in trace.spans:
                if span.parent_span_id:
                    # Find parent span
                    parent_span = next((s for s in trace.spans if s.span_id == span.parent_span_id), None)
                    if parent_span and parent_span.service_name != span.service_name:
                        # Connection from parent service to current service
                        conn_key = f"{parent_span.service_name}->{span.service_name}"
                        connections[conn_key] = connections.get(conn_key, 0) + 1

        # Calculate service metrics
        for service, stats in service_stats.items():
            if stats["request_count"] > 0:
                stats["error_rate"] = stats["error_count"] / stats["request_count"]
                stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["request_count"]

        return {
            "services": list(services),
            "connections": connections,
            "service_statistics": service_stats,
            "time_range_hours": hours
        }

    def export_traces(
        self,
        trace_ids: Optional[List[str]] = None,
        hours: int = 24,
        format: str = "json"
    ) -> str:
        """Export traces for external analysis."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        traces_to_export = []

        if trace_ids:
            # Export specific traces
            for trace_id in trace_ids:
                if trace_id in self._traces:
                    traces_to_export.append(self._traces[trace_id])
        else:
            # Export all recent traces
            for trace in self._traces.values():
                if trace.start_time and trace.start_time >= cutoff_time:
                    traces_to_export.append(trace)

        # Convert to export format
        exported_data = {
            "export_timestamp": datetime.now().isoformat(),
            "trace_count": len(traces_to_export),
            "traces": []
        }

        for trace in traces_to_export:
            trace_data = {
                "trace_id": trace.trace_id,
                "root_span_id": trace.root_span_id,
                "start_time": trace.start_time.isoformat() if trace.start_time else None,
                "end_time": trace.end_time.isoformat() if trace.end_time else None,
                "total_duration_ms": trace.total_duration_ms,
                "user_id": trace.user_id,
                "tenant_id": trace.tenant_id,
                "operation": trace.operation,
                "span_count": trace.span_count,
                "error_count": trace.error_count,
                "spans": []
            }

            for span in trace.spans:
                span_data = {
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    "operation_name": span.operation_name,
                    "span_type": span.span_type.value,
                    "start_time": span.start_time.isoformat(),
                    "end_time": span.end_time.isoformat() if span.end_time else None,
                    "duration_ms": span.duration_ms,
                    "service_name": span.service_name,
                    "tags": span.tags,
                    "logs": span.logs,
                    "status": span.status,
                    "error_message": span.error_message
                }
                trace_data["spans"].append(span_data)

            exported_data["traces"].append(trace_data)

        if format == "json":
            import json
            return json.dumps(exported_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get overall trace statistics."""
        active_spans_count = len(self._active_spans)
        total_traces = len(self._traces)

        # Calculate error rate
        total_spans = sum(len(trace.spans) for trace in self._traces.values())
        error_spans = sum(trace.error_count for trace in self._traces.values())
        error_rate = error_spans / total_spans if total_spans > 0 else 0

        # Calculate average trace duration
        durations = [trace.total_duration_ms for trace in self._traces.values() if trace.total_duration_ms]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_traces": total_traces,
            "active_spans": active_spans_count,
            "total_spans": total_spans,
            "error_rate": error_rate,
            "avg_trace_duration_ms": avg_duration,
            "retention_hours": self.retention_hours,
            "unique_operations": len(self._span_metrics),
            "memory_usage": {
                "traces": len(self._traces),
                "spans": sum(len(t.spans) for t in self._traces.values()),
                "active_spans": len(self._active_spans)
            }
        }

    def _update_trace_stats(self, trace: Trace, span: TraceSpan):
        """Update trace statistics when a span is added."""
        trace.span_count = len(trace.spans)

        if span.status == "error":
            trace.error_count += 1
        elif span.status == "warning":
            trace.warning_count += 1

        # Update timing
        span_times = [s.start_time for s in trace.spans if s.start_time]
        if span_times:
            trace.start_time = min(span_times)

        finished_spans = [s for s in trace.spans if s.end_time]
        if finished_spans:
            end_times = [s.end_time for s in finished_spans]
            trace.end_time = max(end_times)

            if trace.start_time:
                trace.total_duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000

    def _update_span_metrics(self, span: TraceSpan):
        """Update performance metrics for a span."""
        operation = span.operation_name

        if operation not in self._span_metrics:
            self._span_metrics[operation] = []

        if span.duration_ms:
            self._span_metrics[operation].append(span.duration_ms)

            # Keep only recent measurements (last 1000)
            if len(self._span_metrics[operation]) > 1000:
                self._span_metrics[operation] = self._span_metrics[operation][-1000:]

    def _calculate_performance_trends(self, hours: int) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Group spans by hour
        hourly_stats = {}

        for trace in self._traces.values():
            if not trace.start_time or trace.start_time < cutoff_time:
                continue

            for span in trace.spans:
                if not span.duration_ms:
                    continue

                hour_key = span.start_time.replace(minute=0, second=0, microsecond=0)

                if hour_key not in hourly_stats:
                    hourly_stats[hour_key] = {
                        "total_requests": 0,
                        "total_duration_ms": 0,
                        "error_count": 0
                    }

                hourly_stats[hour_key]["total_requests"] += 1
                hourly_stats[hour_key]["total_duration_ms"] += span.duration_ms

                if span.status == "error":
                    hourly_stats[hour_key]["error_count"] += 1

        # Calculate trends
        trends = []
        for hour, stats in sorted(hourly_stats.items()):
            avg_duration = stats["total_duration_ms"] / stats["total_requests"] if stats["total_requests"] > 0 else 0
            error_rate = stats["error_count"] / stats["total_requests"] if stats["total_requests"] > 0 else 0

            trends.append({
                "hour": hour.isoformat(),
                "request_count": stats["total_requests"],
                "avg_duration_ms": avg_duration,
                "error_rate": error_rate
            })

        return trends

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(percentile * len(sorted_values))

        if index >= len(sorted_values):
            return sorted_values[-1]

        return sorted_values[index]

    def _cleanup_loop(self):
        """Background cleanup of old traces."""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

                traces_to_remove = []
                for trace_id, trace in self._traces.items():
                    if trace.start_time and trace.start_time < cutoff_time:
                        traces_to_remove.append(trace_id)

                for trace_id in traces_to_remove:
                    del self._traces[trace_id]

                if traces_to_remove:
                    logger.info(f"Cleaned up {len(traces_to_remove)} old traces")

                time.sleep(3600)  # Run every hour

            except Exception as e:
                logger.error(f"Error in trace cleanup: {e}")
                time.sleep(3600)