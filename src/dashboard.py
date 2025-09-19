"""
dashboard.py
Monitoring dashboard for the Memorizer framework.
Provides real-time metrics visualization and system status.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .performance_monitor import get_performance_monitor, PerformanceMonitor
from .health import get_health_status
from .logging_config import get_request_context

logger = logging.getLogger(__name__)

# Create dashboard router
dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@dataclass
class DashboardMetric:
    """Dashboard metric data structure."""

    name: str
    value: float
    unit: str
    trend: str  # up, down, stable
    status: str  # good, warning, critical
    description: str
    timestamp: datetime


@dataclass
class SystemStatus:
    """System status information."""

    overall_status: str  # healthy, degraded, unhealthy
    components: Dict[str, str]  # component -> status
    uptime: float
    version: str
    last_updated: datetime


class DashboardData(BaseModel):
    """Dashboard data response model."""

    timestamp: str
    system_status: Dict[str, Any]
    metrics: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    performance_summary: Dict[str, Any]


class DashboardService:
    """Service for generating dashboard data."""

    def __init__(self):
        self.performance_monitor = get_performance_monitor()

    def get_system_status(self) -> SystemStatus:
        """Get overall system status."""
        health_status = get_health_status()

        # Determine overall status
        if health_status["status"] == "healthy":
            overall_status = "healthy"
        elif health_status["status"] == "degraded":
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        # Component statuses
        components = {}
        for component, status in health_status.get("components", {}).items():
            components[component] = status["status"]

        return SystemStatus(
            overall_status=overall_status,
            components=components,
            uptime=health_status.get("uptime", 0),
            version=health_status.get("version", "unknown"),
            last_updated=datetime.now(timezone.utc),
        )

    def get_dashboard_metrics(self) -> List[DashboardMetric]:
        """Get key dashboard metrics."""
        metrics = []

        # Get performance summary
        perf_summary = self.performance_monitor.get_metrics_summary()

        # Request metrics
        request_metrics = perf_summary.get("request_metrics", {})
        total_requests = request_metrics.get("total_requests", 0)
        error_rate = request_metrics.get("error_rate", 0)

        metrics.append(
            DashboardMetric(
                name="Total Requests",
                value=total_requests,
                unit="requests",
                trend="stable",
                status="good" if total_requests > 0 else "warning",
                description="Total number of requests processed",
                timestamp=datetime.now(timezone.utc),
            )
        )

        metrics.append(
            DashboardMetric(
                name="Error Rate",
                value=error_rate * 100,
                unit="%",
                trend=(
                    "down"
                    if error_rate < 0.01
                    else "up" if error_rate > 0.05 else "stable"
                ),
                status=(
                    "good"
                    if error_rate < 0.01
                    else "warning" if error_rate < 0.05 else "critical"
                ),
                description="Percentage of requests that resulted in errors",
                timestamp=datetime.now(timezone.utc),
            )
        )

        # Cache metrics
        cache_metrics = perf_summary.get("cache_metrics", {})
        hit_ratio = cache_metrics.get("hit_ratio", 0)

        metrics.append(
            DashboardMetric(
                name="Cache Hit Ratio",
                value=hit_ratio * 100,
                unit="%",
                trend=(
                    "up" if hit_ratio > 0.8 else "down" if hit_ratio < 0.6 else "stable"
                ),
                status=(
                    "good"
                    if hit_ratio > 0.8
                    else "warning" if hit_ratio > 0.6 else "critical"
                ),
                description="Percentage of cache requests that were hits",
                timestamp=datetime.now(timezone.utc),
            )
        )

        # System metrics
        system_metrics = perf_summary.get("system_metrics", {})
        memory_usage = system_metrics.get("memory_usage", 0)
        active_connections = system_metrics.get("active_connections", 0)

        metrics.append(
            DashboardMetric(
                name="Memory Usage",
                value=memory_usage / (1024 * 1024),  # Convert to MB
                unit="MB",
                trend="up" if memory_usage > 500 * 1024 * 1024 else "stable",
                status=(
                    "good"
                    if memory_usage < 500 * 1024 * 1024
                    else "warning" if memory_usage < 1024 * 1024 * 1024 else "critical"
                ),
                description="Current memory usage",
                timestamp=datetime.now(timezone.utc),
            )
        )

        metrics.append(
            DashboardMetric(
                name="Active Connections",
                value=active_connections,
                unit="connections",
                trend="up" if active_connections > 100 else "stable",
                status=(
                    "good"
                    if active_connections < 100
                    else "warning" if active_connections < 500 else "critical"
                ),
                description="Number of active connections",
                timestamp=datetime.now(timezone.utc),
            )
        )

        return metrics

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        # This would typically come from the alert manager
        # For now, return empty list
        return []

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary data."""
        return self.performance_monitor.get_metrics_summary()

    def get_dashboard_data(self) -> DashboardData:
        """Get complete dashboard data."""
        system_status = self.get_system_status()
        metrics = self.get_dashboard_metrics()
        alerts = self.get_active_alerts()
        performance_summary = self.get_performance_summary()

        return DashboardData(
            timestamp=datetime.now(timezone.utc).isoformat(),
            system_status=asdict(system_status),
            metrics=[asdict(metric) for metric in metrics],
            alerts=alerts,
            performance_summary=performance_summary,
        )


# Global dashboard service
_dashboard_service = None


def get_dashboard_service() -> DashboardService:
    """Get global dashboard service instance."""
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service


@dashboard_router.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Serve the main dashboard page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Memorizer Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metric-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
            .metric-unit { color: #666; font-size: 0.9em; }
            .status-good { color: #28a745; }
            .status-warning { color: #ffc107; }
            .status-critical { color: #dc3545; }
            .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            .refresh-btn:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Memorizer Framework Dashboard</h1>
            <p>Real-time monitoring and metrics</p>
            <button class="refresh-btn" onclick="refreshDashboard()">Refresh</button>
        </div>
        
        <div class="metrics-grid" id="metrics-grid">
            <!-- Metrics will be populated here -->
        </div>
        
        <div class="charts">
            <div class="chart-container">
                <h3>Request Rate</h3>
                <canvas id="requestChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Response Time</h3>
                <canvas id="responseChart"></canvas>
            </div>
        </div>
        
        <script>
            let requestChart, responseChart;
            
            async function refreshDashboard() {
                try {
                    const response = await fetch('/dashboard/api/data');
                    const data = await response.json();
                    updateMetrics(data.metrics);
                    updateCharts(data);
                } catch (error) {
                    console.error('Error refreshing dashboard:', error);
                }
            }
            
            function updateMetrics(metrics) {
                const grid = document.getElementById('metrics-grid');
                grid.innerHTML = '';
                
                metrics.forEach(metric => {
                    const card = document.createElement('div');
                    card.className = 'metric-card';
                    card.innerHTML = `
                        <h4>${metric.name}</h4>
                        <div class="metric-value status-${metric.status}">${metric.value.toFixed(2)}</div>
                        <div class="metric-unit">${metric.unit}</div>
                        <p>${metric.description}</p>
                    `;
                    grid.appendChild(card);
                });
            }
            
            function updateCharts(data) {
                // Update request rate chart
                if (requestChart) {
                    requestChart.destroy();
                }
                const requestCtx = document.getElementById('requestChart').getContext('2d');
                requestChart = new Chart(requestCtx, {
                    type: 'line',
                    data: {
                        labels: ['1m ago', '2m ago', '3m ago', '4m ago', '5m ago', 'Now'],
                        datasets: [{
                            label: 'Requests/min',
                            data: [12, 19, 3, 5, 2, 3],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: { beginAtZero: true }
                        }
                    }
                });
                
                // Update response time chart
                if (responseChart) {
                    responseChart.destroy();
                }
                const responseCtx = document.getElementById('responseChart').getContext('2d');
                responseChart = new Chart(responseCtx, {
                    type: 'line',
                    data: {
                        labels: ['1m ago', '2m ago', '3m ago', '4m ago', '5m ago', 'Now'],
                        datasets: [{
                            label: 'Response Time (ms)',
                            data: [65, 59, 80, 81, 56, 55],
                            borderColor: 'rgb(255, 99, 132)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: { beginAtZero: true }
                        }
                    }
                });
            }
            
            // Initial load
            refreshDashboard();
            
            // Auto-refresh every 30 seconds
            setInterval(refreshDashboard, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@dashboard_router.get("/api/data", response_model=DashboardData)
async def get_dashboard_data():
    """Get dashboard data as JSON."""
    service = get_dashboard_service()
    return service.get_dashboard_data()


@dashboard_router.get("/api/metrics")
async def get_metrics():
    """Get raw metrics data."""
    monitor = get_performance_monitor()
    return monitor.get_metrics_summary()


@dashboard_router.get("/api/prometheus")
async def get_prometheus_metrics():
    """Get Prometheus-formatted metrics."""
    monitor = get_performance_monitor()
    metrics = monitor.get_prometheus_metrics()
    return JSONResponse(content=metrics, media_type="text/plain")


@dashboard_router.get("/api/health")
async def get_health_status():
    """Get health status for dashboard."""
    return get_health_status()


@dashboard_router.get("/api/alerts")
async def get_alerts():
    """Get active alerts."""
    service = get_dashboard_service()
    return service.get_active_alerts()


@dashboard_router.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    # This would typically update the alert status in the database
    return {"status": "acknowledged", "alert_id": alert_id}


@dashboard_router.get("/api/logs")
async def get_recent_logs(limit: int = 100):
    """Get recent log entries."""
    # This would typically fetch from a log aggregation system
    return {"logs": [], "message": "Log aggregation not configured"}


@dashboard_router.get("/api/traces")
async def get_recent_traces(limit: int = 50):
    """Get recent request traces."""
    # This would typically fetch from a distributed tracing system
    return {"traces": [], "message": "Distributed tracing not configured"}
