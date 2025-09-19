"""
performance_monitor.py
Comprehensive performance monitoring for the Memorizer framework.
Provides metrics collection, dashboards, and alerting capabilities.
"""
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from collections import defaultdict, deque
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
from functools import wraps
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    metric_type: str  # counter, gauge, histogram

@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==, !=
    threshold: float
    severity: str  # low, medium, high, critical
    description: str
    enabled: bool = True

class PerformanceCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.lock = threading.Lock()
    
    def add_metric(self, metric: PerformanceMetric) -> None:
        """Add a performance metric."""
        with self.lock:
            self.metrics.append(metric)
    
    def get_metrics(self, name: Optional[str] = None, limit: int = 1000) -> List[PerformanceMetric]:
        """Get metrics, optionally filtered by name."""
        with self.lock:
            if name:
                return [m for m in self.metrics if m.name == name][-limit:]
            return list(self.metrics)[-limit:]
    
    def get_latest_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get the latest metric for a given name."""
        with self.lock:
            for metric in reversed(self.metrics):
                if metric.name == name:
                    return metric
        return None

class AlertManager:
    """Manages alert rules and notifications."""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, datetime] = {}
        self.alert_handlers: List[Callable] = []
        self.lock = threading.Lock()
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self.lock:
            self.rules[rule.name] = rule
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add an alert notification handler."""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Check metrics against alert rules."""
        triggered_alerts = []
        
        with self.lock:
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                # Get latest metric for this rule
                latest_metrics = [m for m in metrics if m.name == rule.metric_name]
                if not latest_metrics:
                    continue
                
                latest_metric = latest_metrics[-1]
                
                # Check condition
                if self._evaluate_condition(latest_metric.value, rule.condition, rule.threshold):
                    # Check if alert is already active
                    if rule_name not in self.active_alerts:
                        self.active_alerts[rule_name] = datetime.now(timezone.utc)
                        
                        alert_data = {
                            "rule_name": rule_name,
                            "metric_name": rule.metric_name,
                            "value": latest_metric.value,
                            "threshold": rule.threshold,
                            "condition": rule.condition,
                            "severity": rule.severity,
                            "description": rule.description,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        
                        triggered_alerts.append(alert_data)
                        
                        # Notify handlers
                        for handler in self.alert_handlers:
                            try:
                                handler(alert_data)
                            except Exception as e:
                                logger.error(f"Alert handler failed: {e}")
                else:
                    # Clear alert if condition is no longer met
                    if rule_name in self.active_alerts:
                        del self.active_alerts[rule_name]
        
        return triggered_alerts
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False

class PerformanceMonitor:
    """Main performance monitoring class."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.collector = PerformanceCollector()
        self.alert_manager = AlertManager()
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics."""
        # Request metrics
        self.request_count = Counter(
            'memorizer_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'memorizer_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Memory metrics
        self.memory_operations = Counter(
            'memorizer_memory_operations_total',
            'Total memory operations',
            ['operation', 'tier', 'status'],
            registry=self.registry
        )
        
        self.memory_duration = Histogram(
            'memorizer_memory_duration_seconds',
            'Memory operation duration',
            ['operation', 'tier'],
            registry=self.registry
        )
        
        # Database metrics
        self.database_operations = Counter(
            'memorizer_database_operations_total',
            'Total database operations',
            ['operation', 'table', 'status'],
            registry=self.registry
        )
        
        self.database_duration = Histogram(
            'memorizer_database_duration_seconds',
            'Database operation duration',
            ['operation', 'table'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations = Counter(
            'memorizer_cache_operations_total',
            'Total cache operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'memorizer_cache_hit_ratio',
            'Cache hit ratio',
            registry=self.registry
        )
        
        # Vector database metrics
        self.vector_operations = Counter(
            'memorizer_vector_operations_total',
            'Total vector database operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.vector_duration = Histogram(
            'memorizer_vector_duration_seconds',
            'Vector operation duration',
            ['operation'],
            registry=self.registry
        )
        
        # System metrics
        self.active_connections = Gauge(
            'memorizer_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memorizer_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        # Business metrics
        self.user_operations = Counter(
            'memorizer_user_operations_total',
            'Total user operations',
            ['user_id', 'operation', 'status'],
            registry=self.registry
        )
        
        self.memory_tier_distribution = Gauge(
            'memorizer_memory_tier_count',
            'Number of memories per tier',
            ['tier'],
            registry=self.registry
        )
    
    def _start_background_monitoring(self) -> None:
        """Start background monitoring tasks."""
        def monitor_loop():
            while True:
                try:
                    self._collect_system_metrics()
                    self._check_alerts()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            import psutil
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            self.memory_usage.set(memory_info.used)
            
            # Active connections (simplified)
            self.active_connections.set(len(psutil.net_connections()))
            
        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _check_alerts(self) -> None:
        """Check alert rules against current metrics."""
        try:
            metrics = self.collector.get_metrics()
            triggered_alerts = self.alert_manager.check_alerts(metrics)
            
            for alert in triggered_alerts:
                logger.warning(f"Alert triggered: {alert['rule_name']} - {alert['description']}")
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def track_request(self, method: str, endpoint: str, status_code: int, duration: float) -> None:
        """Track HTTP request metrics."""
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # Add to performance collector
        metric = PerformanceMetric(
            name="request_duration",
            value=duration,
            timestamp=datetime.now(timezone.utc),
            labels={"method": method, "endpoint": endpoint, "status_code": str(status_code)},
            metric_type="histogram"
        )
        self.collector.add_metric(metric)
    
    def track_memory_operation(self, operation: str, tier: str, duration: float, success: bool = True) -> None:
        """Track memory operation metrics."""
        status = "success" if success else "error"
        
        self.memory_operations.labels(
            operation=operation,
            tier=tier,
            status=status
        ).inc()
        
        self.memory_duration.labels(
            operation=operation,
            tier=tier
        ).observe(duration)
        
        # Track in performance collector
        metric = PerformanceMetric(
            name="memory_operation_duration",
            value=duration,
            timestamp=datetime.now(timezone.utc),
            labels={"operation": operation, "tier": tier, "status": status},
            metric_type="histogram"
        )
        self.collector.add_metric(metric)
    
    def track_database_operation(self, operation: str, table: str, duration: float, success: bool = True) -> None:
        """Track database operation metrics."""
        status = "success" if success else "error"
        
        self.database_operations.labels(
            operation=operation,
            table=table,
            status=status
        ).inc()
        
        self.database_duration.labels(
            operation=operation,
            table=table
        ).observe(duration)
    
    def track_cache_operation(self, operation: str, hit: bool = None) -> None:
        """Track cache operation metrics."""
        status = "hit" if hit is True else "miss" if hit is False else "unknown"
        
        self.cache_operations.labels(
            operation=operation,
            status=status
        ).inc()
        
        # Update hit ratio
        if hit is not None:
            self._update_cache_hit_ratio()
    
    def _update_cache_hit_ratio(self) -> None:
        """Update cache hit ratio metric."""
        try:
            # Get cache operation counts
            hit_count = self.cache_operations.labels(operation="get", status="hit")._value._value
            miss_count = self.cache_operations.labels(operation="get", status="miss")._value._value
            
            total = hit_count + miss_count
            if total > 0:
                ratio = hit_count / total
                self.cache_hit_ratio.set(ratio)
        except Exception as e:
            logger.error(f"Error updating cache hit ratio: {e}")
    
    def track_vector_operation(self, operation: str, duration: float, success: bool = True) -> None:
        """Track vector database operation metrics."""
        status = "success" if success else "error"
        
        self.vector_operations.labels(
            operation=operation,
            status=status
        ).inc()
        
        self.vector_duration.labels(
            operation=operation
        ).observe(duration)
    
    def track_user_operation(self, user_id: str, operation: str, success: bool = True) -> None:
        """Track user operation metrics."""
        status = "success" if success else "error"
        
        self.user_operations.labels(
            user_id=user_id,
            operation=operation,
            status=status
        ).inc()
    
    def update_memory_tier_distribution(self, tier_counts: Dict[str, int]) -> None:
        """Update memory tier distribution metrics."""
        for tier, count in tier_counts.items():
            self.memory_tier_distribution.labels(tier=tier).set(count)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_manager.add_rule(rule)
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add an alert notification handler."""
        self.alert_manager.add_alert_handler(handler)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_metrics": {
                "total_requests": sum(self.operation_counts.values()),
                "error_rate": sum(self.error_counts.values()) / max(sum(self.operation_counts.values()), 1),
            },
            "memory_metrics": {
                "tier_distribution": dict(self.memory_tier_distribution._metrics),
            },
            "cache_metrics": {
                "hit_ratio": self.cache_hit_ratio._value._value,
            },
            "system_metrics": {
                "active_connections": self.active_connections._value._value,
                "memory_usage": self.memory_usage._value._value,
            },
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        return generate_latest(self.registry).decode('utf-8')

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def performance_timer(operation_name: str, labels: Dict[str, str] = None):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Track performance
                if labels:
                    for key, value in labels.items():
                        if key == "operation":
                            monitor.track_memory_operation(value, labels.get("tier", "unknown"), duration, True)
                        elif key == "db_operation":
                            monitor.track_database_operation(value, labels.get("table", "unknown"), duration, True)
                        elif key == "vector_operation":
                            monitor.track_vector_operation(value, duration, True)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Track error
                if labels:
                    for key, value in labels.items():
                        if key == "operation":
                            monitor.track_memory_operation(value, labels.get("tier", "unknown"), duration, False)
                        elif key == "db_operation":
                            monitor.track_database_operation(value, labels.get("table", "unknown"), duration, False)
                        elif key == "vector_operation":
                            monitor.track_vector_operation(value, duration, False)
                
                raise
        
        return wrapper
    return decorator

# Default alert rules
def setup_default_alert_rules() -> None:
    """Setup default alert rules."""
    monitor = get_performance_monitor()
    
    # High error rate alert
    monitor.add_alert_rule(AlertRule(
        name="high_error_rate",
        metric_name="error_rate",
        condition=">",
        threshold=0.05,  # 5% error rate
        severity="high",
        description="Error rate is above 5%"
    ))
    
    # High response time alert
    monitor.add_alert_rule(AlertRule(
        name="high_response_time",
        metric_name="request_duration",
        condition=">",
        threshold=5.0,  # 5 seconds
        severity="medium",
        description="Response time is above 5 seconds"
    ))
    
    # Low cache hit ratio alert
    monitor.add_alert_rule(AlertRule(
        name="low_cache_hit_ratio",
        metric_name="cache_hit_ratio",
        condition="<",
        threshold=0.8,  # 80% hit ratio
        severity="medium",
        description="Cache hit ratio is below 80%"
    ))
    
    # High memory usage alert
    monitor.add_alert_rule(AlertRule(
        name="high_memory_usage",
        metric_name="memory_usage",
        condition=">",
        threshold=1024 * 1024 * 1024,  # 1GB
        severity="high",
        description="Memory usage is above 1GB"
    ))

# Initialize default alert rules
setup_default_alert_rules()
