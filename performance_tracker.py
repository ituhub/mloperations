# ML Model Monitoring Platform - Performance Tracking Module
# =============================================================================
# Comprehensive model performance monitoring and metrics tracking
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import json
from scipy import stats

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked"""
    # Regression metrics
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    MAPE = "mape"
    R2 = "r2"
    
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC_ROC = "auc_roc"
    LOG_LOSS = "log_loss"
    
    # Custom metrics
    CUSTOM = "custom"
    
    # Latency metrics
    PREDICTION_LATENCY = "prediction_latency"
    THROUGHPUT = "throughput"
    
    # Business metrics
    REVENUE_IMPACT = "revenue_impact"
    COST_SAVINGS = "cost_savings"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Single metric measurement"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    model_id: str
    batch_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance degradation alert"""
    alert_level: AlertLevel
    metric_type: MetricType
    current_value: float
    threshold: float
    baseline_value: float
    deviation_percent: float
    timestamp: datetime
    model_id: str
    message: str
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'alert_level': self.alert_level.value,
            'metric_type': self.metric_type.value,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'baseline_value': self.baseline_value,
            'deviation_percent': self.deviation_percent,
            'timestamp': self.timestamp.isoformat(),
            'model_id': self.model_id,
            'message': self.message,
            'recommendations': self.recommendations
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    model_id: str
    timestamp: datetime
    report_period: str
    metrics: Dict[str, float]
    metric_trends: Dict[str, str]
    alerts: List[PerformanceAlert]
    predictions_count: int
    avg_latency_ms: float
    error_rate: float
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """
    Comprehensive model performance tracking system.
    
    Features:
    - Real-time metric tracking
    - Performance baseline establishment
    - Degradation detection
    - Trend analysis
    - Alerting system
    - SLA monitoring
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        alert_cooldown_minutes: int = 15,
        degradation_threshold: float = 0.1,  # 10% degradation triggers alert
        enable_trend_detection: bool = True
    ):
        self.window_size = window_size
        self.alert_cooldown_minutes = alert_cooldown_minutes
        self.degradation_threshold = degradation_threshold
        self.enable_trend_detection = enable_trend_detection
        
        # Metric storage
        self.metric_history: Dict[str, Dict[MetricType, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window_size))
        )
        
        # Baselines
        self.baselines: Dict[str, Dict[MetricType, Dict]] = defaultdict(dict)
        
        # Thresholds
        self.thresholds: Dict[str, Dict[MetricType, Dict]] = defaultdict(dict)
        
        # Alert history
        self.alert_history: Dict[str, List[PerformanceAlert]] = defaultdict(list)
        self.last_alert_time: Dict[str, Dict[MetricType, datetime]] = defaultdict(dict)
        
        # Prediction tracking
        self.prediction_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        logger.info("Initialized PerformanceTracker")
    
    def set_baseline(
        self,
        model_id: str,
        metric_type: MetricType,
        baseline_value: float,
        std_dev: Optional[float] = None,
        percentiles: Optional[Dict[str, float]] = None
    ):
        """
        Set performance baseline for a metric.
        
        Args:
            model_id: Model identifier
            metric_type: Type of metric
            baseline_value: Baseline (expected) value
            std_dev: Standard deviation
            percentiles: Percentile values (p25, p50, p75, p95, p99)
        """
        self.baselines[model_id][metric_type] = {
            'value': baseline_value,
            'std_dev': std_dev or 0.0,
            'percentiles': percentiles or {},
            'established_at': datetime.now().isoformat()
        }
        
        logger.info(f"Set baseline for model '{model_id}', metric '{metric_type.value}': {baseline_value}")
    
    def establish_baseline_from_data(
        self,
        model_id: str,
        metric_type: MetricType,
        values: List[float]
    ):
        """Establish baseline from historical data"""
        if not values:
            logger.warning("No values provided for baseline establishment")
            return
        
        values_array = np.array(values)
        
        self.baselines[model_id][metric_type] = {
            'value': float(np.mean(values_array)),
            'std_dev': float(np.std(values_array)),
            'percentiles': {
                'p25': float(np.percentile(values_array, 25)),
                'p50': float(np.percentile(values_array, 50)),
                'p75': float(np.percentile(values_array, 75)),
                'p95': float(np.percentile(values_array, 95)),
                'p99': float(np.percentile(values_array, 99))
            },
            'established_at': datetime.now().isoformat(),
            'sample_size': len(values)
        }
        
        logger.info(
            f"Established baseline for model '{model_id}', metric '{metric_type.value}' "
            f"from {len(values)} samples: mean={np.mean(values_array):.4f}"
        )
    
    def set_threshold(
        self,
        model_id: str,
        metric_type: MetricType,
        warning_threshold: float,
        error_threshold: float,
        critical_threshold: Optional[float] = None,
        direction: str = 'higher_is_worse'  # or 'lower_is_worse'
    ):
        """Set alerting thresholds for a metric"""
        self.thresholds[model_id][metric_type] = {
            'warning': warning_threshold,
            'error': error_threshold,
            'critical': critical_threshold or error_threshold * 1.5,
            'direction': direction
        }
        
        logger.info(
            f"Set thresholds for model '{model_id}', metric '{metric_type.value}': "
            f"warning={warning_threshold}, error={error_threshold}"
        )
    
    def log_metric(
        self,
        model_id: str,
        metric_type: MetricType,
        value: float,
        batch_size: int = 1,
        metadata: Optional[Dict] = None
    ) -> Optional[PerformanceAlert]:
        """
        Log a metric value and check for degradation.
        
        Args:
            model_id: Model identifier
            metric_type: Type of metric
            value: Metric value
            batch_size: Number of predictions in batch
            metadata: Additional metadata
            
        Returns:
            PerformanceAlert if threshold exceeded, None otherwise
        """
        timestamp = datetime.now()
        
        # Store metric
        metric_value = MetricValue(
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            model_id=model_id,
            batch_size=batch_size,
            metadata=metadata or {}
        )
        
        self.metric_history[model_id][metric_type].append({
            'value': value,
            'timestamp': timestamp.isoformat(),
            'batch_size': batch_size
        })
        
        # Check for degradation
        alert = self._check_degradation(model_id, metric_type, value, timestamp)
        
        if alert:
            self._store_alert(model_id, alert)
        
        return alert
    
    def log_prediction(
        self,
        model_id: str,
        y_true: Union[float, np.ndarray],
        y_pred: Union[float, np.ndarray],
        latency_ms: Optional[float] = None,
        is_regression: bool = True
    ) -> Dict[MetricType, float]:
        """
        Log a prediction with automatic metric calculation.
        
        Args:
            model_id: Model identifier
            y_true: True values
            y_pred: Predicted values
            latency_ms: Prediction latency in milliseconds
            is_regression: True for regression, False for classification
            
        Returns:
            Dict of calculated metrics
        """
        y_true = np.atleast_1d(y_true)
        y_pred = np.atleast_1d(y_pred)
        
        # Update counts
        self.prediction_counts[model_id] += len(y_true)
        
        # Log latency
        if latency_ms is not None:
            self.latency_history[model_id].append(latency_ms)
            self.log_metric(model_id, MetricType.PREDICTION_LATENCY, latency_ms)
        
        # Calculate metrics
        metrics = {}
        
        if is_regression:
            # Regression metrics
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            
            metrics[MetricType.MAE] = float(mae)
            metrics[MetricType.MSE] = float(mse)
            metrics[MetricType.RMSE] = float(rmse)
            
            # MAPE (avoid division by zero)
            nonzero_mask = y_true != 0
            if np.any(nonzero_mask):
                mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
                metrics[MetricType.MAPE] = float(mape)
            
            # R2
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            metrics[MetricType.R2] = float(r2)
        
        else:
            # Classification metrics
            y_true_int = y_true.astype(int)
            y_pred_int = (y_pred > 0.5).astype(int) if y_pred.max() <= 1 else y_pred.astype(int)
            
            accuracy = np.mean(y_true_int == y_pred_int)
            metrics[MetricType.ACCURACY] = float(accuracy)
            
            # Precision, Recall, F1 for binary
            tp = np.sum((y_true_int == 1) & (y_pred_int == 1))
            fp = np.sum((y_true_int == 0) & (y_pred_int == 1))
            fn = np.sum((y_true_int == 1) & (y_pred_int == 0))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics[MetricType.PRECISION] = float(precision)
            metrics[MetricType.RECALL] = float(recall)
            metrics[MetricType.F1] = float(f1)
        
        # Log all metrics
        alerts = []
        for metric_type, value in metrics.items():
            alert = self.log_metric(model_id, metric_type, value, batch_size=len(y_true))
            if alert:
                alerts.append(alert)
        
        return metrics
    
    def _check_degradation(
        self,
        model_id: str,
        metric_type: MetricType,
        value: float,
        timestamp: datetime
    ) -> Optional[PerformanceAlert]:
        """Check if metric has degraded beyond threshold"""
        # Check cooldown
        last_alert = self.last_alert_time.get(model_id, {}).get(metric_type)
        if last_alert:
            time_since_alert = (timestamp - last_alert).total_seconds()
            if time_since_alert < self.alert_cooldown_minutes * 60:
                return None
        
        # Get baseline and thresholds
        baseline = self.baselines.get(model_id, {}).get(metric_type)
        thresholds = self.thresholds.get(model_id, {}).get(metric_type)
        
        if not baseline and not thresholds:
            # No baseline or thresholds set, use dynamic detection
            return self._dynamic_degradation_check(model_id, metric_type, value, timestamp)
        
        alert_level = None
        baseline_value = baseline['value'] if baseline else 0
        
        if thresholds:
            direction = thresholds.get('direction', 'higher_is_worse')
            
            if direction == 'higher_is_worse':
                if value >= thresholds.get('critical', float('inf')):
                    alert_level = AlertLevel.CRITICAL
                elif value >= thresholds.get('error', float('inf')):
                    alert_level = AlertLevel.ERROR
                elif value >= thresholds.get('warning', float('inf')):
                    alert_level = AlertLevel.WARNING
            else:  # lower_is_worse (e.g., accuracy, R2)
                if value <= thresholds.get('critical', float('-inf')):
                    alert_level = AlertLevel.CRITICAL
                elif value <= thresholds.get('error', float('-inf')):
                    alert_level = AlertLevel.ERROR
                elif value <= thresholds.get('warning', float('-inf')):
                    alert_level = AlertLevel.WARNING
        
        elif baseline:
            # Use baseline-based detection
            deviation = abs(value - baseline_value) / (baseline_value + 1e-8)
            
            if deviation > self.degradation_threshold * 3:
                alert_level = AlertLevel.CRITICAL
            elif deviation > self.degradation_threshold * 2:
                alert_level = AlertLevel.ERROR
            elif deviation > self.degradation_threshold:
                alert_level = AlertLevel.WARNING
        
        if alert_level:
            deviation_percent = ((value - baseline_value) / (baseline_value + 1e-8)) * 100
            
            alert = PerformanceAlert(
                alert_level=alert_level,
                metric_type=metric_type,
                current_value=value,
                threshold=thresholds.get(alert_level.value, 0) if thresholds else baseline_value * (1 + self.degradation_threshold),
                baseline_value=baseline_value,
                deviation_percent=deviation_percent,
                timestamp=timestamp,
                model_id=model_id,
                message=self._generate_alert_message(
                    model_id, metric_type, value, baseline_value, alert_level
                ),
                recommendations=self._generate_recommendations(
                    metric_type, value, baseline_value, alert_level
                )
            )
            
            self.last_alert_time[model_id][metric_type] = timestamp
            return alert
        
        return None
    
    def _dynamic_degradation_check(
        self,
        model_id: str,
        metric_type: MetricType,
        value: float,
        timestamp: datetime
    ) -> Optional[PerformanceAlert]:
        """Dynamic degradation detection without preset baseline"""
        history = self.metric_history[model_id][metric_type]
        
        if len(history) < 50:
            return None
        
        # Calculate rolling statistics
        values = [h['value'] for h in list(history)[-100:]]
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        # Z-score based detection
        if std_value > 0:
            z_score = abs(value - mean_value) / std_value
            
            if z_score > 4:
                alert_level = AlertLevel.CRITICAL
            elif z_score > 3:
                alert_level = AlertLevel.ERROR
            elif z_score > 2.5:
                alert_level = AlertLevel.WARNING
            else:
                return None
            
            return PerformanceAlert(
                alert_level=alert_level,
                metric_type=metric_type,
                current_value=value,
                threshold=mean_value + 2.5 * std_value,
                baseline_value=mean_value,
                deviation_percent=((value - mean_value) / (mean_value + 1e-8)) * 100,
                timestamp=timestamp,
                model_id=model_id,
                message=f"Anomalous {metric_type.value} detected: {value:.4f} (z-score: {z_score:.2f})",
                recommendations=[
                    "Investigate recent data changes",
                    "Check for upstream data quality issues",
                    "Review recent model updates"
                ]
            )
        
        return None
    
    def _generate_alert_message(
        self,
        model_id: str,
        metric_type: MetricType,
        value: float,
        baseline: float,
        alert_level: AlertLevel
    ) -> str:
        """Generate human-readable alert message"""
        direction = "increased" if value > baseline else "decreased"
        change = abs(value - baseline)
        change_pct = (change / (baseline + 1e-8)) * 100
        
        return (
            f"Model '{model_id}' {metric_type.value} has {direction} by {change_pct:.1f}% "
            f"(current: {value:.4f}, baseline: {baseline:.4f})"
        )
    
    def _generate_recommendations(
        self,
        metric_type: MetricType,
        value: float,
        baseline: float,
        alert_level: AlertLevel
    ) -> List[str]:
        """Generate recommendations based on metric degradation"""
        recommendations = []
        
        if alert_level == AlertLevel.CRITICAL:
            recommendations.append("⚠️ CRITICAL: Immediate investigation required")
            recommendations.append("Consider rolling back to previous model version")
            recommendations.append("Check for data quality issues in production")
        
        elif alert_level == AlertLevel.ERROR:
            recommendations.append("Schedule immediate investigation")
            recommendations.append("Monitor closely for further degradation")
            recommendations.append("Prepare retraining pipeline")
        
        else:
            recommendations.append("Monitor this metric over next 24 hours")
            recommendations.append("Review recent data distribution changes")
        
        # Metric-specific recommendations
        if metric_type in [MetricType.MAE, MetricType.MSE, MetricType.RMSE]:
            recommendations.append("Check for outliers in recent input data")
            recommendations.append("Verify feature engineering pipeline")
        
        elif metric_type in [MetricType.ACCURACY, MetricType.F1]:
            recommendations.append("Review class distribution in recent data")
            recommendations.append("Check for label quality issues")
        
        elif metric_type == MetricType.PREDICTION_LATENCY:
            recommendations.append("Check infrastructure resources")
            recommendations.append("Review model complexity")
            recommendations.append("Consider model optimization or pruning")
        
        return recommendations
    
    def _store_alert(self, model_id: str, alert: PerformanceAlert):
        """Store alert in history"""
        self.alert_history[model_id].append(alert)
        
        # Keep only recent alerts
        if len(self.alert_history[model_id]) > 1000:
            self.alert_history[model_id] = self.alert_history[model_id][-1000:]
    
    def get_metrics_summary(
        self,
        model_id: str,
        lookback_minutes: int = 60
    ) -> Dict[MetricType, Dict]:
        """Get summary of recent metrics"""
        cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
        summary = {}
        
        for metric_type, history in self.metric_history[model_id].items():
            recent = [
                h for h in history 
                if datetime.fromisoformat(h['timestamp']) > cutoff
            ]
            
            if recent:
                values = [h['value'] for h in recent]
                summary[metric_type] = {
                    'current': values[-1],
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values),
                    'trend': self._calculate_trend(values)
                }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 5:
            return 'insufficient_data'
        
        recent = np.mean(values[-5:])
        older = np.mean(values[:-5]) if len(values) > 5 else values[0]
        
        if recent > older * 1.05:
            return 'increasing'
        elif recent < older * 0.95:
            return 'decreasing'
        else:
            return 'stable'
    
    def generate_report(
        self,
        model_id: str,
        report_period_hours: int = 24
    ) -> PerformanceReport:
        """Generate comprehensive performance report"""
        cutoff = datetime.now() - timedelta(hours=report_period_hours)
        
        # Collect metrics
        metrics = {}
        metric_trends = {}
        
        for metric_type, history in self.metric_history[model_id].items():
            recent = [
                h for h in history 
                if datetime.fromisoformat(h['timestamp']) > cutoff
            ]
            
            if recent:
                values = [h['value'] for h in recent]
                metrics[metric_type.value] = float(np.mean(values))
                metric_trends[metric_type.value] = self._calculate_trend(values)
        
        # Collect alerts
        recent_alerts = [
            a for a in self.alert_history[model_id]
            if a.timestamp > cutoff
        ]
        
        # Calculate statistics
        predictions_count = self.prediction_counts[model_id]
        
        latencies = list(self.latency_history[model_id])
        avg_latency = np.mean(latencies) if latencies else 0
        
        error_rate = (
            self.error_counts[model_id] / (predictions_count + 1e-8)
        ) if predictions_count > 0 else 0
        
        # Generate recommendations
        recommendations = self._generate_report_recommendations(
            metrics, metric_trends, recent_alerts
        )
        
        return PerformanceReport(
            model_id=model_id,
            timestamp=datetime.now(),
            report_period=f"{report_period_hours}h",
            metrics=metrics,
            metric_trends=metric_trends,
            alerts=recent_alerts,
            predictions_count=predictions_count,
            avg_latency_ms=float(avg_latency),
            error_rate=float(error_rate),
            recommendations=recommendations
        )
    
    def _generate_report_recommendations(
        self,
        metrics: Dict,
        trends: Dict,
        alerts: List[PerformanceAlert]
    ) -> List[str]:
        """Generate recommendations for performance report"""
        recommendations = []
        
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.alert_level == AlertLevel.CRITICAL]
        if critical_alerts:
            recommendations.append(
                f"⚠️ {len(critical_alerts)} CRITICAL alerts detected - immediate action required"
            )
        
        # Check trends
        degrading_metrics = [m for m, t in trends.items() if t == 'increasing' and 'error' in m.lower()]
        if degrading_metrics:
            recommendations.append(
                f"Degrading metrics detected: {', '.join(degrading_metrics)}"
            )
        
        # General health
        if not alerts and all(t in ['stable', 'decreasing'] for t in trends.values()):
            recommendations.append("✅ Model performance is healthy")
        
        return recommendations
    
    def export_metrics(
        self,
        model_id: str,
        format: str = 'json',
        lookback_hours: int = 24
    ) -> str:
        """Export metrics to specified format"""
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        
        data = {}
        for metric_type, history in self.metric_history[model_id].items():
            recent = [
                h for h in history 
                if datetime.fromisoformat(h['timestamp']) > cutoff
            ]
            data[metric_type.value] = recent
        
        if format == 'json':
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
