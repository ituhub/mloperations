# ML Model Monitoring Platform - Alert Management Module
# =============================================================================
# Comprehensive alert management, routing, and notification system
# =============================================================================

import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from datetime import datetime, timedelta
import logging
import json
import requests
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert lifecycle status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertCategory(Enum):
    """Categories of alerts"""
    DRIFT = "drift"
    PERFORMANCE = "performance"
    LATENCY = "latency"
    ERROR = "error"
    DATA_QUALITY = "data_quality"
    SYSTEM = "system"
    SLA = "sla"


@dataclass
class Alert:
    """Unified alert object"""
    alert_id: str
    model_id: str
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'model_id': self.model_id,
            'severity': self.severity.value,
            'category': self.category.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'details': self.details,
            'recommendations': self.recommendations,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_notes': self.resolution_notes,
            'tags': self.tags
        }


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    model_id: Optional[str]  # None means applies to all models
    category: AlertCategory
    condition: str  # e.g., "metric > threshold"
    threshold: float
    severity: AlertSeverity
    cooldown_minutes: int = 30
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationChannel(ABC):
    """Abstract base class for notification channels"""
    
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send alert notification"""
        pass
    
    @abstractmethod
    def get_channel_type(self) -> str:
        """Get channel type identifier"""
        pass


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send(self, alert: Alert) -> bool:
        try:
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.LOW: "📝",
                AlertSeverity.MEDIUM: "⚠️",
                AlertSeverity.HIGH: "🔴",
                AlertSeverity.CRITICAL: "🚨"
            }
            
            emoji = severity_emoji.get(alert.severity, "📢")
            
            payload = {
                "channel": self.channel,
                "username": "ML Monitor",
                "icon_emoji": ":robot_face:",
                "attachments": [{
                    "color": self._get_color(alert.severity),
                    "title": f"{emoji} {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Model", "value": alert.model_id, "short": True},
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Category", "value": alert.category.value, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                    ],
                    "footer": "ML Model Monitoring Platform"
                }]
            }
            
            if alert.metric_value is not None:
                payload["attachments"][0]["fields"].append({
                    "title": "Metric Value",
                    "value": f"{alert.metric_value:.4f}",
                    "short": True
                })
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _get_color(self, severity: AlertSeverity) -> str:
        colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.LOW: "#2196F3",
            AlertSeverity.MEDIUM: "#FF9800",
            AlertSeverity.HIGH: "#f44336",
            AlertSeverity.CRITICAL: "#9C27B0"
        }
        return colors.get(severity, "#808080")
    
    def get_channel_type(self) -> str:
        return "slack"


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel"""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
    
    def send(self, alert: Alert) -> bool:
        try:
            payload = alert.to_dict()
            response = requests.post(
                self.webhook_url, 
                json=payload, 
                headers=self.headers,
                timeout=10
            )
            return response.status_code in [200, 201, 202]
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def get_channel_type(self) -> str:
        return "webhook"


class AlertManager:
    """
    Comprehensive alert management system.
    
    Features:
    - Alert creation and lifecycle management
    - Notification routing to multiple channels
    - Alert suppression and deduplication
    - Escalation policies
    - Alert aggregation
    """
    
    def __init__(
        self,
        default_cooldown_minutes: int = 30,
        max_alerts_per_hour: int = 100,
        enable_aggregation: bool = True
    ):
        self.default_cooldown_minutes = default_cooldown_minutes
        self.max_alerts_per_hour = max_alerts_per_hour
        self.enable_aggregation = enable_aggregation
        
        # Storage
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.rules: Dict[str, AlertRule] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Tracking
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_counts: Dict[str, int] = defaultdict(int)
        
        # Alert counter
        self._alert_counter = 0
        
        logger.info("Initialized AlertManager")
    
    def register_channel(self, channel_id: str, channel: NotificationChannel):
        """Register a notification channel"""
        self.notification_channels[channel_id] = channel
        logger.info(f"Registered notification channel: {channel_id}")
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.rule_id}")
    
    def create_alert(
        self,
        model_id: str,
        severity: AlertSeverity,
        category: AlertCategory,
        title: str,
        message: str,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
        details: Optional[Dict] = None,
        recommendations: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        skip_notification: bool = False
    ) -> Optional[Alert]:
        """Create and process a new alert"""
        # Check cooldown
        cooldown_key = f"{model_id}_{category.value}_{metric_name or 'general'}"
        if not self._check_cooldown(cooldown_key):
            return None
        
        # Generate alert ID
        self._alert_counter += 1
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._alert_counter}"
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            model_id=model_id,
            severity=severity,
            category=category,
            title=title,
            message=message,
            timestamp=datetime.now(),
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            details=details or {},
            recommendations=recommendations or [],
            tags=tags or []
        )
        
        # Store alert
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_time[cooldown_key] = datetime.now()
        
        # Send notifications
        if not skip_notification:
            self._send_notifications(alert)
        
        logger.info(f"Created alert: {alert_id} ({severity.value})")
        return alert
    
    def _check_cooldown(self, cooldown_key: str) -> bool:
        """Check if cooldown period has passed"""
        last_time = self.last_alert_time.get(cooldown_key)
        if last_time is None:
            return True
        elapsed = (datetime.now() - last_time).total_seconds()
        return elapsed >= self.default_cooldown_minutes * 60
    
    def _send_notifications(self, alert: Alert):
        """Send alert to notification channels"""
        channels_to_use = list(self.notification_channels.keys())
        
        if alert.severity not in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            return
        
        for channel_id in channels_to_use:
            channel = self.notification_channels.get(channel_id)
            if channel:
                try:
                    channel.send(alert)
                except Exception as e:
                    logger.error(f"Error sending to channel {channel_id}: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()
        return True
    
    def resolve_alert(self, alert_id: str, resolution_notes: Optional[str] = None) -> bool:
        """Resolve an alert"""
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.resolution_notes = resolution_notes
        return True
    
    def get_active_alerts(self, model_id: Optional[str] = None) -> List[Alert]:
        """Get all active alerts"""
        alerts = [a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE]
        if model_id:
            alerts = [a for a in alerts if a.model_id == model_id]
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Get alert summary statistics"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [a for a in self.alert_history if a.timestamp > cutoff]
        
        by_severity = defaultdict(int)
        for a in recent:
            by_severity[a.severity.value] += 1
        
        return {
            'total_alerts': len(recent),
            'by_severity': dict(by_severity),
            'period_hours': hours
        }
