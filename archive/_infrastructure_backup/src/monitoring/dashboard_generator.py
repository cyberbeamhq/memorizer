"""
Dashboard Generator Module
Generates interactive dashboards for memory system monitoring and visualization.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import base64

logger = logging.getLogger(__name__)


@dataclass
class DashboardWidget:
    """Represents a dashboard widget."""
    id: str
    title: str
    type: str  # "metric", "chart", "table", "gauge", "heatmap"
    data_source: str
    config: Dict[str, Any]
    position: Dict[str, int]  # {"x": 0, "y": 0, "width": 4, "height": 3}
    refresh_interval: int = 30  # seconds


@dataclass
class Dashboard:
    """Represents a complete dashboard."""
    id: str
    title: str
    description: str
    widgets: List[DashboardWidget]
    layout: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    tags: List[str]


class DashboardGenerator:
    """Generates monitoring dashboards for memory systems."""

    def __init__(self, metrics_collector=None, alerting_system=None, analytics=None):
        """Initialize dashboard generator."""
        self.metrics_collector = metrics_collector
        self.alerting_system = alerting_system
        self.analytics = analytics

        # Dashboard templates
        self.templates = self._create_dashboard_templates()

        # Dashboard storage
        self.dashboards: Dict[str, Dashboard] = {}

        logger.info("Dashboard generator initialized")

    def create_dashboard(self, template_name: str, title: Optional[str] = None) -> Dashboard:
        """Create a dashboard from a template."""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")

        template = self.templates[template_name]
        dashboard_id = f"{template_name}_{int(datetime.now().timestamp())}"

        dashboard = Dashboard(
            id=dashboard_id,
            title=title or template["title"],
            description=template["description"],
            widgets=template["widgets"],
            layout=template["layout"],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=template.get("tags", [])
        )

        self.dashboards[dashboard_id] = dashboard
        logger.info(f"Created dashboard: {dashboard.title}")

        return dashboard

    def generate_overview_dashboard(self) -> Dashboard:
        """Generate a comprehensive overview dashboard."""
        widgets = []

        # System health gauge
        widgets.append(DashboardWidget(
            id="system_health",
            title="System Health",
            type="gauge",
            data_source="health_monitor",
            config={
                "metric": "overall_health_score",
                "min": 0,
                "max": 100,
                "thresholds": [
                    {"value": 70, "color": "red"},
                    {"value": 85, "color": "yellow"},
                    {"value": 100, "color": "green"}
                ]
            },
            position={"x": 0, "y": 0, "width": 3, "height": 3}
        ))

        # Memory usage chart
        widgets.append(DashboardWidget(
            id="memory_usage",
            title="Memory Usage Over Time",
            type="chart",
            data_source="metrics",
            config={
                "chart_type": "line",
                "metrics": ["total_memories", "active_memories", "archived_memories"],
                "time_range": "24h",
                "y_axis": {"label": "Count"}
            },
            position={"x": 3, "y": 0, "width": 6, "height": 3}
        ))

        # Active alerts table
        widgets.append(DashboardWidget(
            id="active_alerts",
            title="Active Alerts",
            type="table",
            data_source="alerting",
            config={
                "columns": ["severity", "title", "triggered_at", "current_value"],
                "sort_by": "triggered_at",
                "sort_order": "desc",
                "max_rows": 10
            },
            position={"x": 9, "y": 0, "width": 3, "height": 3}
        ))

        # Storage efficiency metrics
        widgets.append(DashboardWidget(
            id="storage_metrics",
            title="Storage Efficiency",
            type="metric",
            data_source="analytics",
            config={
                "metrics": [
                    {"name": "total_storage_bytes", "label": "Total Storage", "format": "bytes"},
                    {"name": "compression_ratio", "label": "Avg Compression", "format": "percentage"},
                    {"name": "tier_distribution", "label": "Tier Distribution", "format": "object"}
                ]
            },
            position={"x": 0, "y": 3, "width": 4, "height": 2}
        ))

        # Performance chart
        widgets.append(DashboardWidget(
            id="performance_metrics",
            title="Response Times",
            type="chart",
            data_source="metrics",
            config={
                "chart_type": "line",
                "metrics": ["avg_response_time", "p95_response_time", "p99_response_time"],
                "time_range": "6h",
                "y_axis": {"label": "Milliseconds"}
            },
            position={"x": 4, "y": 3, "width": 4, "height": 2}
        ))

        # Top users table
        widgets.append(DashboardWidget(
            id="top_users",
            title="Most Active Users",
            type="table",
            data_source="analytics",
            config={
                "columns": ["user_id", "memory_count", "storage_usage", "last_activity"],
                "sort_by": "memory_count",
                "sort_order": "desc",
                "max_rows": 5
            },
            position={"x": 8, "y": 3, "width": 4, "height": 2}
        ))

        # Error rate heatmap
        widgets.append(DashboardWidget(
            id="error_heatmap",
            title="Error Rate Heatmap",
            type="heatmap",
            data_source="metrics",
            config={
                "metric": "error_rate",
                "time_range": "7d",
                "granularity": "hour"
            },
            position={"x": 0, "y": 5, "width": 6, "height": 3}
        ))

        # Alert timeline
        widgets.append(DashboardWidget(
            id="alert_timeline",
            title="Alert Timeline",
            type="chart",
            data_source="alerting",
            config={
                "chart_type": "bar",
                "metrics": ["alert_count"],
                "group_by": "severity",
                "time_range": "7d",
                "granularity": "day"
            },
            position={"x": 6, "y": 5, "width": 6, "height": 3}
        ))

        dashboard = Dashboard(
            id="overview_dashboard",
            title="Memory System Overview",
            description="Comprehensive overview of memory system health and performance",
            widgets=widgets,
            layout={"grid_size": 12, "row_height": 50},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["overview", "system", "monitoring"]
        )

        self.dashboards[dashboard.id] = dashboard
        return dashboard

    def generate_performance_dashboard(self) -> Dashboard:
        """Generate a performance-focused dashboard."""
        widgets = []

        # Response time distribution
        widgets.append(DashboardWidget(
            id="response_time_dist",
            title="Response Time Distribution",
            type="chart",
            data_source="metrics",
            config={
                "chart_type": "histogram",
                "metric": "response_times",
                "bins": 20,
                "time_range": "1h"
            },
            position={"x": 0, "y": 0, "width": 6, "height": 3}
        ))

        # Throughput chart
        widgets.append(DashboardWidget(
            id="throughput",
            title="Operations per Second",
            type="chart",
            data_source="metrics",
            config={
                "chart_type": "line",
                "metrics": ["operations_per_second", "successful_ops_per_second", "failed_ops_per_second"],
                "time_range": "2h",
                "y_axis": {"label": "Ops/sec"}
            },
            position={"x": 6, "y": 0, "width": 6, "height": 3}
        ))

        # Cache hit rate
        widgets.append(DashboardWidget(
            id="cache_performance",
            title="Cache Performance",
            type="gauge",
            data_source="metrics",
            config={
                "metric": "cache_hit_rate",
                "min": 0,
                "max": 100,
                "format": "percentage",
                "thresholds": [
                    {"value": 70, "color": "red"},
                    {"value": 85, "color": "yellow"},
                    {"value": 100, "color": "green"}
                ]
            },
            position={"x": 0, "y": 3, "width": 3, "height": 3}
        ))

        # Database performance
        widgets.append(DashboardWidget(
            id="db_performance",
            title="Database Query Performance",
            type="chart",
            data_source="metrics",
            config={
                "chart_type": "line",
                "metrics": ["db_query_time", "db_connection_pool"],
                "time_range": "1h",
                "y_axis": {"label": "Time (ms)"}
            },
            position={"x": 3, "y": 3, "width": 6, "height": 3}
        ))

        # Resource utilization
        widgets.append(DashboardWidget(
            id="resource_util",
            title="Resource Utilization",
            type="chart",
            data_source="metrics",
            config={
                "chart_type": "area",
                "metrics": ["cpu_usage", "memory_usage", "disk_usage"],
                "time_range": "4h",
                "y_axis": {"label": "Percentage", "max": 100}
            },
            position={"x": 9, "y": 3, "width": 3, "height": 3}
        ))

        dashboard = Dashboard(
            id="performance_dashboard",
            title="Performance Monitoring",
            description="Detailed performance metrics and system resource utilization",
            widgets=widgets,
            layout={"grid_size": 12, "row_height": 50},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["performance", "monitoring", "resources"]
        )

        self.dashboards[dashboard.id] = dashboard
        return dashboard

    def generate_tenant_dashboard(self, tenant_id: str) -> Dashboard:
        """Generate a tenant-specific dashboard."""
        widgets = []

        # Tenant memory usage
        widgets.append(DashboardWidget(
            id=f"tenant_{tenant_id}_usage",
            title=f"Memory Usage - Tenant {tenant_id}",
            type="chart",
            data_source="analytics",
            config={
                "chart_type": "area",
                "metrics": ["memory_count", "storage_bytes"],
                "filters": {"tenant_id": tenant_id},
                "time_range": "30d"
            },
            position={"x": 0, "y": 0, "width": 6, "height": 3}
        ))

        # Tenant limits gauge
        widgets.append(DashboardWidget(
            id=f"tenant_{tenant_id}_limits",
            title="Usage vs Limits",
            type="gauge",
            data_source="tenant_manager",
            config={
                "metric": "usage_percentage",
                "tenant_id": tenant_id,
                "min": 0,
                "max": 100,
                "format": "percentage"
            },
            position={"x": 6, "y": 0, "width": 3, "height": 3}
        ))

        # Recent activity
        widgets.append(DashboardWidget(
            id=f"tenant_{tenant_id}_activity",
            title="Recent Activity",
            type="table",
            data_source="usage_tracker",
            config={
                "filters": {"tenant_id": tenant_id},
                "columns": ["timestamp", "event_type", "user_id", "memory_id"],
                "max_rows": 10,
                "time_range": "24h"
            },
            position={"x": 9, "y": 0, "width": 3, "height": 3}
        ))

        dashboard = Dashboard(
            id=f"tenant_{tenant_id}_dashboard",
            title=f"Tenant {tenant_id} Dashboard",
            description=f"Monitoring dashboard for tenant {tenant_id}",
            widgets=widgets,
            layout={"grid_size": 12, "row_height": 50},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["tenant", tenant_id]
        )

        self.dashboards[dashboard.id] = dashboard
        return dashboard

    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get data for a dashboard."""
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard not found: {dashboard_id}")

        dashboard = self.dashboards[dashboard_id]
        widget_data = {}

        for widget in dashboard.widgets:
            try:
                data = self._get_widget_data(widget)
                widget_data[widget.id] = {
                    "widget": self._widget_to_dict(widget),
                    "data": data,
                    "last_updated": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting data for widget {widget.id}: {e}")
                widget_data[widget.id] = {
                    "widget": self._widget_to_dict(widget),
                    "error": str(e),
                    "last_updated": datetime.now().isoformat()
                }

        return {
            "dashboard": self._dashboard_to_dict(dashboard),
            "widgets": widget_data
        }

    def export_dashboard_config(self, dashboard_id: str) -> str:
        """Export dashboard configuration as JSON."""
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard not found: {dashboard_id}")

        dashboard = self.dashboards[dashboard_id]
        config = self._dashboard_to_dict(dashboard)

        return json.dumps(config, indent=2, default=str)

    def import_dashboard_config(self, config_json: str) -> Dashboard:
        """Import dashboard from JSON configuration."""
        config = json.loads(config_json)

        widgets = []
        for widget_config in config["widgets"]:
            widget = DashboardWidget(
                id=widget_config["id"],
                title=widget_config["title"],
                type=widget_config["type"],
                data_source=widget_config["data_source"],
                config=widget_config["config"],
                position=widget_config["position"],
                refresh_interval=widget_config.get("refresh_interval", 30)
            )
            widgets.append(widget)

        dashboard = Dashboard(
            id=config["id"],
            title=config["title"],
            description=config["description"],
            widgets=widgets,
            layout=config["layout"],
            created_at=datetime.fromisoformat(config["created_at"]),
            updated_at=datetime.now(),
            tags=config.get("tags", [])
        )

        self.dashboards[dashboard.id] = dashboard
        return dashboard

    def list_dashboards(self) -> List[Dict[str, Any]]:
        """List all available dashboards."""
        return [
            {
                "id": dashboard.id,
                "title": dashboard.title,
                "description": dashboard.description,
                "created_at": dashboard.created_at.isoformat(),
                "updated_at": dashboard.updated_at.isoformat(),
                "tags": dashboard.tags,
                "widget_count": len(dashboard.widgets)
            }
            for dashboard in self.dashboards.values()
        ]

    def _create_dashboard_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create built-in dashboard templates."""
        return {
            "overview": {
                "title": "System Overview",
                "description": "High-level overview of system health and performance",
                "widgets": [],
                "layout": {"grid_size": 12, "row_height": 50},
                "tags": ["overview", "system"]
            },
            "performance": {
                "title": "Performance Dashboard",
                "description": "Detailed performance metrics and analysis",
                "widgets": [],
                "layout": {"grid_size": 12, "row_height": 50},
                "tags": ["performance", "metrics"]
            },
            "alerts": {
                "title": "Alert Dashboard",
                "description": "Alert management and monitoring",
                "widgets": [],
                "layout": {"grid_size": 12, "row_height": 50},
                "tags": ["alerts", "monitoring"]
            }
        }

    def _get_widget_data(self, widget: DashboardWidget) -> Any:
        """Get data for a specific widget."""
        data_source = widget.data_source
        config = widget.config

        if data_source == "metrics" and self.metrics_collector:
            return self._get_metrics_data(config)
        elif data_source == "alerting" and self.alerting_system:
            return self._get_alerting_data(config)
        elif data_source == "analytics" and self.analytics:
            return self._get_analytics_data(config)
        elif data_source == "health_monitor":
            return self._get_health_data(config)
        else:
            return {"error": f"Unknown data source: {data_source}"}

    def _get_metrics_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get metrics data."""
        if not self.metrics_collector:
            return {"error": "Metrics collector not available"}

        metrics = config.get("metrics", [])
        time_range = config.get("time_range", "1h")

        # Simulate metrics data
        data = {}
        for metric in metrics:
            # Generate sample time series data
            data[metric] = self._generate_sample_timeseries(metric, time_range)

        return data

    def _get_alerting_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get alerting data."""
        if not self.alerting_system:
            return {"error": "Alerting system not available"}

        if "columns" in config:  # Table widget
            alerts = self.alerting_system.get_active_alerts()
            return {
                "alerts": [
                    {
                        "severity": alert.severity.value,
                        "title": alert.title,
                        "triggered_at": alert.triggered_at.isoformat(),
                        "current_value": alert.current_value
                    }
                    for alert in alerts[:config.get("max_rows", 10)]
                ]
            }
        else:  # Chart widget
            return self.alerting_system.create_dashboard_data()

    def _get_analytics_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get analytics data."""
        if not self.analytics:
            return {"error": "Analytics not available"}

        # Return sample analytics data
        return {
            "total_memories": 1000,
            "total_storage_bytes": 50 * 1024 * 1024,
            "compression_ratio": 0.65,
            "tier_distribution": {
                "very_new": 300,
                "mid_term": 500,
                "long_term": 200
            }
        }

    def _get_health_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get health monitoring data."""
        # Return sample health data
        return {
            "overall_health_score": 85,
            "component_health": {
                "database": 90,
                "cache": 95,
                "storage": 80,
                "api": 85
            }
        }

    def _generate_sample_timeseries(self, metric: str, time_range: str) -> List[Dict[str, Any]]:
        """Generate sample time series data."""
        import random

        # Parse time range
        if time_range.endswith('h'):
            hours = int(time_range[:-1])
        elif time_range.endswith('d'):
            hours = int(time_range[:-1]) * 24
        else:
            hours = 1

        # Generate data points
        points = []
        now = datetime.now()

        for i in range(hours * 6):  # 6 points per hour (10-minute intervals)
            timestamp = now - timedelta(minutes=i * 10)
            value = random.uniform(0, 100) if "percentage" in metric else random.uniform(0, 1000)

            points.append({
                "timestamp": timestamp.isoformat(),
                "value": value
            })

        return list(reversed(points))

    def _widget_to_dict(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Convert widget to dictionary."""
        return {
            "id": widget.id,
            "title": widget.title,
            "type": widget.type,
            "data_source": widget.data_source,
            "config": widget.config,
            "position": widget.position,
            "refresh_interval": widget.refresh_interval
        }

    def _dashboard_to_dict(self, dashboard: Dashboard) -> Dict[str, Any]:
        """Convert dashboard to dictionary."""
        return {
            "id": dashboard.id,
            "title": dashboard.title,
            "description": dashboard.description,
            "widgets": [self._widget_to_dict(w) for w in dashboard.widgets],
            "layout": dashboard.layout,
            "created_at": dashboard.created_at.isoformat(),
            "updated_at": dashboard.updated_at.isoformat(),
            "tags": dashboard.tags
        }