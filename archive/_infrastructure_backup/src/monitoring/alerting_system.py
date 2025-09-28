"""
Alerting System Module
Advanced alerting and notification system for memory management operations.
"""

import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import threading
import time

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    CONSOLE = "console"


@dataclass
class AlertRule:
    """Defines an alert rule."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str  # Condition expression
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"

    # Timing
    evaluation_interval: int = 60  # seconds
    for_duration: int = 300  # seconds (5 minutes)

    # Notification settings
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    notification_template: Optional[str] = None

    # Rate limiting
    max_alerts_per_hour: int = 10
    cooldown_period: int = 3600  # seconds (1 hour)

    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """Represents an active alert."""
    id: str
    rule_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE

    # Values
    current_value: Optional[float] = None
    threshold: Optional[float] = None

    # Timing
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    # Context
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Tracking
    notification_count: int = 0
    last_notification: Optional[datetime] = None


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    email: Optional[Dict[str, Any]] = None
    slack: Optional[Dict[str, Any]] = None
    webhook: Optional[Dict[str, Any]] = None
    sms: Optional[Dict[str, Any]] = None


class AlertingSystem:
    """Advanced alerting system for memory management."""

    def __init__(self, notification_config: Optional[NotificationConfig] = None):
        """Initialize alerting system."""
        self.notification_config = notification_config or NotificationConfig()

        # Alert management
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # Metrics tracking
        self.metrics: Dict[str, float] = {}
        self.metric_history: Dict[str, List[Dict[str, Any]]] = {}

        # Rate limiting
        self.notification_history: Dict[str, List[datetime]] = {}

        # Background processing
        self._running = False
        self._evaluation_thread: Optional[threading.Thread] = None

        logger.info("Alerting system initialized")

    def start(self):
        """Start the alerting system."""
        if self._running:
            return

        self._running = True
        self._evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self._evaluation_thread.start()

        logger.info("Alerting system started")

    def stop(self):
        """Stop the alerting system."""
        self._running = False
        if self._evaluation_thread:
            self._evaluation_thread.join(timeout=5)

        logger.info("Alerting system stopped")

    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]

            # Resolve any active alerts for this rule
            alerts_to_resolve = [
                alert for alert in self.active_alerts.values()
                if alert.rule_id == rule_id
            ]

            for alert in alerts_to_resolve:
                self.resolve_alert(alert.id, "Rule removed")

            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False

    def update_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Update a metric value."""
        self.metrics[metric_name] = value

        # Store history
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []

        self.metric_history[metric_name].append({
            "timestamp": datetime.now().isoformat(),
            "value": value,
            "labels": labels or {}
        })

        # Keep only recent history (last 1000 points)
        if len(self.metric_history[metric_name]) > 1000:
            self.metric_history[metric_name] = self.metric_history[metric_name][-1000:]

    def evaluate_rules(self):
        """Evaluate all alert rules."""
        for rule in self.rules.values():
            if not rule.enabled:
                continue

            try:
                self._evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.id}: {e}")

    def trigger_alert(self, rule: AlertRule, current_value: float, context: Optional[Dict[str, Any]] = None):
        """Trigger an alert."""
        alert_id = f"{rule.id}_{int(datetime.now().timestamp())}"

        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            title=rule.name,
            description=rule.description,
            severity=rule.severity,
            current_value=current_value,
            threshold=rule.threshold,
            labels=context.get("labels", {}) if context else {},
            annotations=context.get("annotations", {}) if context else {}
        )

        self.active_alerts[alert_id] = alert

        # Send notifications
        self._send_notifications(alert, rule)

        logger.warning(f"Alert triggered: {rule.name} (value: {current_value}, threshold: {rule.threshold})")

    def acknowledge_alert(self, alert_id: str, user_id: str, note: Optional[str] = None):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()

            if note:
                alert.annotations["acknowledgment_note"] = note
                alert.annotations["acknowledged_by"] = user_id

            logger.info(f"Alert {alert_id} acknowledged by {user_id}")

    def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()

            if resolution_note:
                alert.annotations["resolution_note"] = resolution_note

            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]

            logger.info(f"Alert {alert_id} resolved")

    def suppress_alert(self, alert_id: str, duration_hours: int, reason: str):
        """Suppress an alert for a specified duration."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.annotations["suppressed_until"] = (datetime.now() + timedelta(hours=duration_hours)).isoformat()
            alert.annotations["suppression_reason"] = reason

            logger.info(f"Alert {alert_id} suppressed for {duration_hours} hours: {reason}")

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)

    def get_alert_history(self, hours: int = 24, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        cutoff = datetime.now() - timedelta(hours=hours)

        recent_alerts = [
            alert for alert in self.alert_history
            if alert.triggered_at >= cutoff
        ]

        return sorted(recent_alerts, key=lambda a: a.triggered_at, reverse=True)[:limit]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        active_count = len(self.active_alerts)

        # Count by severity
        severity_counts = {}
        for alert in self.active_alerts.values():
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Historical stats (last 24 hours)
        last_24h = datetime.now() - timedelta(hours=24)
        recent_history = [
            alert for alert in self.alert_history
            if alert.triggered_at >= last_24h
        ]

        return {
            "active_alerts": active_count,
            "active_by_severity": severity_counts,
            "alerts_last_24h": len(recent_history),
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_metrics": len(self.metrics)
        }

    def create_dashboard_data(self) -> Dict[str, Any]:
        """Create data for alerting dashboard."""
        # Alert timeline (last 7 days)
        timeline_data = []
        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            day_alerts = [
                alert for alert in self.alert_history
                if day_start <= alert.triggered_at < day_end
            ]

            timeline_data.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "count": len(day_alerts),
                "critical": len([a for a in day_alerts if a.severity == AlertSeverity.CRITICAL]),
                "error": len([a for a in day_alerts if a.severity == AlertSeverity.ERROR]),
                "warning": len([a for a in day_alerts if a.severity == AlertSeverity.WARNING])
            })

        # Top alerting rules
        rule_stats = {}
        for alert in self.alert_history[-100:]:  # Last 100 alerts
            rule_id = alert.rule_id
            if rule_id not in rule_stats:
                rule_stats[rule_id] = {"count": 0, "rule_name": self.rules.get(rule_id, {}).get("name", "Unknown")}
            rule_stats[rule_id]["count"] += 1

        top_rules = sorted(rule_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:10]

        return {
            "alert_timeline": timeline_data,
            "top_alerting_rules": [{"rule_id": r[0], **r[1]} for r in top_rules],
            "current_metrics": self.metrics,
            "active_alerts": [self._alert_to_dict(alert) for alert in self.get_active_alerts()],
            "statistics": self.get_alert_statistics()
        }

    def _evaluation_loop(self):
        """Background loop for evaluating alert rules."""
        while self._running:
            try:
                self.evaluate_rules()
                self._cleanup_resolved_suppressions()
                time.sleep(10)  # Evaluate every 10 seconds
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                time.sleep(30)  # Back off on error

    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        metric_name = rule.condition

        if metric_name not in self.metrics:
            return  # Metric not available

        current_value = self.metrics[metric_name]

        # Check condition
        triggered = self._check_condition(current_value, rule.threshold, rule.comparison)

        # Check if we should trigger an alert
        if triggered:
            # Check rate limiting
            if not self._should_send_alert(rule):
                return

            # Check if alert already exists for this rule
            existing_alert = None
            for alert in self.active_alerts.values():
                if alert.rule_id == rule.id and alert.status == AlertStatus.ACTIVE:
                    existing_alert = alert
                    break

            if not existing_alert:
                self.trigger_alert(rule, current_value)
            else:
                # Update existing alert
                existing_alert.current_value = current_value
                existing_alert.last_notification = datetime.now()

    def _check_condition(self, current_value: float, threshold: float, comparison: str) -> bool:
        """Check if a condition is met."""
        if comparison == "gt":
            return current_value > threshold
        elif comparison == "gte":
            return current_value >= threshold
        elif comparison == "lt":
            return current_value < threshold
        elif comparison == "lte":
            return current_value <= threshold
        elif comparison == "eq":
            return current_value == threshold
        else:
            return False

    def _should_send_alert(self, rule: AlertRule) -> bool:
        """Check if we should send an alert based on rate limiting."""
        now = datetime.now()

        # Get recent notifications for this rule
        if rule.id not in self.notification_history:
            self.notification_history[rule.id] = []

        recent_notifications = self.notification_history[rule.id]

        # Remove old notifications (outside rate limit window)
        cutoff = now - timedelta(seconds=rule.cooldown_period)
        recent_notifications = [ts for ts in recent_notifications if ts >= cutoff]
        self.notification_history[rule.id] = recent_notifications

        # Check if we've exceeded the rate limit
        hour_ago = now - timedelta(hours=1)
        recent_hour_notifications = [ts for ts in recent_notifications if ts >= hour_ago]

        if len(recent_hour_notifications) >= rule.max_alerts_per_hour:
            return False

        # Check cooldown period
        if recent_notifications and (now - max(recent_notifications)).total_seconds() < rule.cooldown_period:
            return False

        # Record this notification
        self.notification_history[rule.id].append(now)
        return True

    def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for an alert."""
        for channel in rule.notification_channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    self._send_email_notification(alert, rule)
                elif channel == NotificationChannel.SLACK:
                    self._send_slack_notification(alert, rule)
                elif channel == NotificationChannel.WEBHOOK:
                    self._send_webhook_notification(alert, rule)
                elif channel == NotificationChannel.CONSOLE:
                    self._send_console_notification(alert, rule)

                alert.notification_count += 1
                alert.last_notification = datetime.now()

            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification for alert {alert.id}: {e}")

    def _send_email_notification(self, alert: Alert, rule: AlertRule):
        """Send email notification."""
        if not self.notification_config.email:
            return

        config = self.notification_config.email

        msg = MimeMultipart()
        msg['From'] = config['from_address']
        msg['To'] = ', '.join(config['to_addresses'])
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"

        body = self._format_notification_text(alert, rule)
        msg.attach(MimeText(body, 'plain'))

        # Send email
        with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as server:
            if config.get('use_tls'):
                server.starttls()
            if config.get('username') and config.get('password'):
                server.login(config['username'], config['password'])
            server.send_message(msg)

    def _send_slack_notification(self, alert: Alert, rule: AlertRule):
        """Send Slack notification."""
        if not self.notification_config.slack:
            return

        # Implementation would use Slack API
        logger.info(f"Slack notification: {alert.title}")

    def _send_webhook_notification(self, alert: Alert, rule: AlertRule):
        """Send webhook notification."""
        if not self.notification_config.webhook:
            return

        # Implementation would make HTTP POST request
        logger.info(f"Webhook notification: {alert.title}")

    def _send_console_notification(self, alert: Alert, rule: AlertRule):
        """Send console notification."""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }

        emoji = severity_emoji.get(alert.severity, "📢")
        print(f"{emoji} ALERT: {alert.title}")
        print(f"   Severity: {alert.severity.value}")
        print(f"   Description: {alert.description}")
        if alert.current_value is not None:
            print(f"   Current Value: {alert.current_value}")
        if alert.threshold is not None:
            print(f"   Threshold: {alert.threshold}")
        print(f"   Triggered: {alert.triggered_at}")

    def _format_notification_text(self, alert: Alert, rule: AlertRule) -> str:
        """Format notification text."""
        template = rule.notification_template or """
Alert: {title}
Severity: {severity}
Description: {description}
Current Value: {current_value}
Threshold: {threshold}
Triggered At: {triggered_at}

Rule: {rule_name}
Rule ID: {rule_id}
        """.strip()

        return template.format(
            title=alert.title,
            severity=alert.severity.value,
            description=alert.description,
            current_value=alert.current_value or "N/A",
            threshold=alert.threshold or "N/A",
            triggered_at=alert.triggered_at.strftime("%Y-%m-%d %H:%M:%S"),
            rule_name=rule.name,
            rule_id=rule.id
        )

    def _cleanup_resolved_suppressions(self):
        """Clean up resolved and expired suppressions."""
        now = datetime.now()

        alerts_to_resolve = []
        for alert in self.active_alerts.values():
            # Check if suppression has expired
            if alert.status == AlertStatus.SUPPRESSED:
                suppressed_until_str = alert.annotations.get("suppressed_until")
                if suppressed_until_str:
                    try:
                        suppressed_until = datetime.fromisoformat(suppressed_until_str)
                        if now >= suppressed_until:
                            # Re-activate the alert
                            alert.status = AlertStatus.ACTIVE
                            logger.info(f"Alert {alert.id} suppression expired, reactivated")
                    except:
                        pass

    def _alert_to_dict(self, alert: Alert) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": alert.id,
            "rule_id": alert.rule_id,
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "triggered_at": alert.triggered_at.isoformat(),
            "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
            "labels": alert.labels,
            "annotations": alert.annotations,
            "notification_count": alert.notification_count
        }