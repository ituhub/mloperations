# ML Model Monitoring Platform - Drift Detection Module
# =============================================================================
# Enhanced from trading platform's ModelDriftDetector with production features
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import json

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift that can be detected"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_DRIFT = "feature_drift"
    TARGET_DRIFT = "target_drift"
    COVARIATE_DRIFT = "covariate_drift"


class DriftSeverity(Enum):
    """Severity levels for drift alerts"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Data class for drift alerts"""
    drift_type: DriftType
    severity: DriftSeverity
    score: float
    threshold: float
    timestamp: datetime
    model_id: str
    feature_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'drift_type': self.drift_type.value,
            'severity': self.severity.value,
            'score': self.score,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'model_id': self.model_id,
            'feature_name': self.feature_name,
            'details': self.details,
            'recommendations': self.recommendations
        }


@dataclass
class DriftReport:
    """Comprehensive drift analysis report"""
    model_id: str
    timestamp: datetime
    overall_drift_score: float
    drift_detected: bool
    drift_type: Optional[DriftType]
    feature_drift_scores: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Dict[str, float]]
    alerts: List[DriftAlert]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DriftDetector:
    """
    Comprehensive drift detection system for ML models.
    
    Supports multiple drift detection methods:
    - Statistical tests (KS, Chi-squared, PSI)
    - Distribution comparison
    - Feature-level drift analysis
    - Concept drift detection
    - Adaptive thresholds
    """
    
    def __init__(
        self,
        reference_window: int = 1000,
        detection_window: int = 100,
        psi_threshold: float = 0.1,
        ks_threshold: float = 0.05,
        enable_adaptive_threshold: bool = True,
        alert_cooldown_minutes: int = 30
    ):
        self.reference_window = reference_window
        self.detection_window = detection_window
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.enable_adaptive_threshold = enable_adaptive_threshold
        self.alert_cooldown_minutes = alert_cooldown_minutes
        
        # Reference data storage
        self.reference_data: Dict[str, np.ndarray] = {}
        self.reference_stats: Dict[str, Dict] = {}
        self.feature_names: Dict[str, List[str]] = {}
        
        # Drift history
        self.drift_history: Dict[str, List[Dict]] = defaultdict(list)
        self.alert_history: Dict[str, List[DriftAlert]] = defaultdict(list)
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Adaptive threshold tracking
        self.baseline_scores: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"Initialized DriftDetector with PSI threshold={psi_threshold}, KS threshold={ks_threshold}")
    
    def set_reference(
        self,
        model_id: str,
        X_reference: np.ndarray,
        y_reference: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        predictions_reference: Optional[np.ndarray] = None
    ) -> bool:
        """
        Set reference distribution for drift detection.
        
        Args:
            model_id: Unique identifier for the model
            X_reference: Reference feature data
            y_reference: Reference target data (for concept drift)
            feature_names: Names of features
            predictions_reference: Reference predictions (for prediction drift)
            
        Returns:
            bool: Success status
        """
        try:
            # Validate input
            if X_reference is None or len(X_reference) == 0:
                logger.error("Empty reference data provided")
                return False
            
            # Handle multi-dimensional data
            if len(X_reference.shape) > 2:
                X_flat = X_reference.reshape(X_reference.shape[0], -1)
            else:
                X_flat = X_reference
            
            # Store reference data (use most recent samples)
            self.reference_data[model_id] = {
                'X': X_flat[-self.reference_window:],
                'y': y_reference[-self.reference_window:] if y_reference is not None else None,
                'predictions': predictions_reference[-self.reference_window:] if predictions_reference is not None else None
            }
            
            # Generate feature names if not provided
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X_flat.shape[1])]
            self.feature_names[model_id] = feature_names
            
            # Calculate reference statistics
            self.reference_stats[model_id] = self._calculate_statistics(X_flat)
            
            # Calculate baseline drift scores for adaptive thresholding
            if self.enable_adaptive_threshold:
                self._establish_baseline(model_id, X_flat)
            
            logger.info(f"Set reference distribution for model '{model_id}' with {len(X_flat)} samples, {X_flat.shape[1]} features")
            return True
            
        except Exception as e:
            logger.error(f"Error setting reference distribution: {e}")
            return False
    
    def _calculate_statistics(self, data: np.ndarray) -> Dict:
        """Calculate comprehensive statistics for reference data"""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
            'quantiles': {
                'q1': np.percentile(data, 25, axis=0),
                'q2': np.percentile(data, 50, axis=0),
                'q3': np.percentile(data, 75, axis=0)
            },
            'skewness': stats.skew(data, axis=0),
            'kurtosis': stats.kurtosis(data, axis=0)
        }
    
    def _establish_baseline(self, model_id: str, X_reference: np.ndarray):
        """Establish baseline drift scores using cross-validation style splits"""
        try:
            n_splits = 5
            split_size = len(X_reference) // n_splits
            
            for i in range(n_splits - 1):
                ref_split = X_reference[i * split_size:(i + 1) * split_size]
                curr_split = X_reference[(i + 1) * split_size:(i + 2) * split_size]
                
                # Calculate PSI between splits
                psi_scores = []
                for j in range(ref_split.shape[1]):
                    psi = self._calculate_psi(ref_split[:, j], curr_split[:, j])
                    psi_scores.append(psi)
                
                self.baseline_scores[model_id].append(np.mean(psi_scores))
            
        except Exception as e:
            logger.warning(f"Error establishing baseline: {e}")
    
    def detect_drift(
        self,
        model_id: str,
        X_current: np.ndarray,
        y_current: Optional[np.ndarray] = None,
        predictions_current: Optional[np.ndarray] = None
    ) -> DriftReport:
        """
        Detect drift in current data compared to reference.
        
        Args:
            model_id: Model identifier
            X_current: Current feature data
            y_current: Current target data
            predictions_current: Current model predictions
            
        Returns:
            DriftReport: Comprehensive drift analysis report
        """
        if model_id not in self.reference_data:
            logger.error(f"No reference data set for model '{model_id}'")
            return self._create_empty_report(model_id)
        
        try:
            # Prepare data
            if len(X_current.shape) > 2:
                X_flat = X_current.reshape(X_current.shape[0], -1)
            else:
                X_flat = X_current
            
            current_data = X_flat[-self.detection_window:]
            reference_data = self.reference_data[model_id]['X']
            
            # Detect various types of drift
            feature_drift = self._detect_feature_drift(model_id, reference_data, current_data)
            overall_score, drift_detected, drift_type = self._aggregate_drift(feature_drift)
            
            # Statistical tests
            statistical_tests = self._run_statistical_tests(reference_data, current_data)
            
            # Concept drift detection (if targets available)
            concept_drift_result = None
            if y_current is not None and self.reference_data[model_id]['y'] is not None:
                concept_drift_result = self._detect_concept_drift(
                    model_id, 
                    self.reference_data[model_id]['y'],
                    y_current[-self.detection_window:]
                )
            
            # Prediction drift detection
            prediction_drift_result = None
            if predictions_current is not None and self.reference_data[model_id]['predictions'] is not None:
                prediction_drift_result = self._detect_prediction_drift(
                    model_id,
                    self.reference_data[model_id]['predictions'],
                    predictions_current[-self.detection_window:]
                )
            
            # Generate alerts
            alerts = self._generate_alerts(
                model_id, 
                overall_score, 
                feature_drift, 
                concept_drift_result,
                prediction_drift_result
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                drift_detected, 
                feature_drift, 
                overall_score
            )
            
            # Create report
            report = DriftReport(
                model_id=model_id,
                timestamp=datetime.now(),
                overall_drift_score=overall_score,
                drift_detected=drift_detected,
                drift_type=drift_type,
                feature_drift_scores=feature_drift,
                statistical_tests=statistical_tests,
                alerts=alerts,
                recommendations=recommendations,
                metadata={
                    'reference_samples': len(reference_data),
                    'current_samples': len(current_data),
                    'n_features': current_data.shape[1],
                    'concept_drift': concept_drift_result,
                    'prediction_drift': prediction_drift_result
                }
            )
            
            # Store in history
            self._store_drift_record(model_id, report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return self._create_empty_report(model_id)
    
    def _detect_feature_drift(
        self, 
        model_id: str,
        reference: np.ndarray, 
        current: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Detect drift for each feature"""
        feature_drift = {}
        feature_names = self.feature_names.get(model_id, [])
        
        n_features = min(reference.shape[1], current.shape[1])
        
        for i in range(n_features):
            ref_feature = reference[:, i]
            curr_feature = current[:, i]
            
            feature_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
            
            # KS test
            ks_stat, ks_p = stats.ks_2samp(ref_feature, curr_feature)
            
            # PSI
            psi_score = self._calculate_psi(ref_feature, curr_feature)
            
            # Wasserstein distance (Earth Mover's Distance)
            try:
                wasserstein = stats.wasserstein_distance(ref_feature, curr_feature)
            except:
                wasserstein = 0.0
            
            # Jensen-Shannon divergence
            js_divergence = self._calculate_js_divergence(ref_feature, curr_feature)
            
            # Mean shift
            mean_shift = abs(np.mean(curr_feature) - np.mean(ref_feature)) / (np.std(ref_feature) + 1e-8)
            
            # Variance ratio
            variance_ratio = np.var(curr_feature) / (np.var(ref_feature) + 1e-8)
            
            feature_drift[feature_name] = {
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'psi_score': float(psi_score),
                'wasserstein_distance': float(wasserstein),
                'js_divergence': float(js_divergence),
                'mean_shift': float(mean_shift),
                'variance_ratio': float(variance_ratio),
                'drift_detected': psi_score > self.psi_threshold or ks_p < self.ks_threshold
            }
        
        return feature_drift
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        """
        try:
            # Handle edge cases
            if len(reference) == 0 or len(current) == 0:
                return 0.0
            
            # Create buckets based on reference distribution
            min_val = min(np.min(reference), np.min(current))
            max_val = max(np.max(reference), np.max(current))
            
            if min_val == max_val:
                return 0.0
            
            bin_edges = np.linspace(min_val, max_val, buckets + 1)
            
            # Calculate distributions
            ref_dist, _ = np.histogram(reference, bins=bin_edges)
            curr_dist, _ = np.histogram(current, bins=bin_edges)
            
            # Normalize to probabilities
            ref_prob = (ref_dist + 0.001) / (len(reference) + 0.001 * buckets)
            curr_prob = (curr_dist + 0.001) / (len(current) + 0.001 * buckets)
            
            # Calculate PSI
            psi = np.sum((curr_prob - ref_prob) * np.log(curr_prob / ref_prob))
            
            return max(0.0, psi)
            
        except Exception as e:
            logger.warning(f"Error calculating PSI: {e}")
            return 0.0
    
    def _calculate_js_divergence(self, p: np.ndarray, q: np.ndarray, bins: int = 50) -> float:
        """Calculate Jensen-Shannon divergence between two distributions"""
        try:
            min_val = min(np.min(p), np.min(q))
            max_val = max(np.max(p), np.max(q))
            
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
            q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
            
            # Add small value to avoid log(0)
            p_hist = p_hist + 1e-10
            q_hist = q_hist + 1e-10
            
            # Normalize
            p_hist = p_hist / p_hist.sum()
            q_hist = q_hist / q_hist.sum()
            
            # Calculate JS divergence
            m = 0.5 * (p_hist + q_hist)
            js = 0.5 * (stats.entropy(p_hist, m) + stats.entropy(q_hist, m))
            
            return float(js)
            
        except Exception as e:
            logger.warning(f"Error calculating JS divergence: {e}")
            return 0.0
    
    def _run_statistical_tests(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        """Run comprehensive statistical tests"""
        tests = {}
        
        try:
            # Overall KS test
            overall_ks_stat, overall_ks_p = stats.ks_2samp(
                reference.flatten(), 
                current.flatten()
            )
            tests['ks_test'] = {
                'statistic': float(overall_ks_stat),
                'p_value': float(overall_ks_p),
                'significant': overall_ks_p < self.ks_threshold
            }
            
            # Mann-Whitney U test
            try:
                mw_stat, mw_p = stats.mannwhitneyu(
                    reference.flatten()[:1000],
                    current.flatten()[:1000],
                    alternative='two-sided'
                )
                tests['mann_whitney'] = {
                    'statistic': float(mw_stat),
                    'p_value': float(mw_p),
                    'significant': mw_p < 0.05
                }
            except:
                pass
            
            # Levene's test for variance equality
            try:
                levene_stat, levene_p = stats.levene(
                    reference.flatten()[:1000],
                    current.flatten()[:1000]
                )
                tests['levene_test'] = {
                    'statistic': float(levene_stat),
                    'p_value': float(levene_p),
                    'significant': levene_p < 0.05
                }
            except:
                pass
            
        except Exception as e:
            logger.warning(f"Error running statistical tests: {e}")
        
        return tests
    
    def _detect_concept_drift(
        self, 
        model_id: str,
        y_reference: np.ndarray, 
        y_current: np.ndarray
    ) -> Dict:
        """Detect concept drift (change in P(y|X))"""
        try:
            # Distribution comparison
            ks_stat, ks_p = stats.ks_2samp(y_reference.flatten(), y_current.flatten())
            
            # Mean shift
            mean_shift = abs(np.mean(y_current) - np.mean(y_reference))
            
            # Variance change
            var_ratio = np.var(y_current) / (np.var(y_reference) + 1e-8)
            
            return {
                'detected': ks_p < self.ks_threshold,
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'mean_shift': float(mean_shift),
                'variance_ratio': float(var_ratio)
            }
        except Exception as e:
            logger.warning(f"Error detecting concept drift: {e}")
            return {'detected': False}
    
    def _detect_prediction_drift(
        self,
        model_id: str,
        pred_reference: np.ndarray,
        pred_current: np.ndarray
    ) -> Dict:
        """Detect drift in model predictions"""
        try:
            ks_stat, ks_p = stats.ks_2samp(pred_reference.flatten(), pred_current.flatten())
            psi = self._calculate_psi(pred_reference.flatten(), pred_current.flatten())
            
            return {
                'detected': psi > self.psi_threshold,
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'psi_score': float(psi),
                'mean_shift': float(abs(np.mean(pred_current) - np.mean(pred_reference)))
            }
        except Exception as e:
            logger.warning(f"Error detecting prediction drift: {e}")
            return {'detected': False}
    
    def _aggregate_drift(
        self, 
        feature_drift: Dict[str, Dict]
    ) -> Tuple[float, bool, Optional[DriftType]]:
        """Aggregate feature-level drift into overall score"""
        if not feature_drift:
            return 0.0, False, None
        
        psi_scores = [f['psi_score'] for f in feature_drift.values()]
        ks_stats = [f['ks_statistic'] for f in feature_drift.values()]
        
        avg_psi = np.mean(psi_scores)
        max_psi = np.max(psi_scores)
        avg_ks = np.mean(ks_stats)
        
        # Combined score (weighted average)
        overall_score = 0.6 * max_psi + 0.3 * avg_psi + 0.1 * avg_ks
        
        # Adaptive threshold
        if self.enable_adaptive_threshold and self.baseline_scores:
            baseline_mean = np.mean([np.mean(s) for s in self.baseline_scores.values()])
            baseline_std = np.std([np.mean(s) for s in self.baseline_scores.values()])
            adaptive_threshold = baseline_mean + 2 * baseline_std + self.psi_threshold
        else:
            adaptive_threshold = self.psi_threshold
        
        drift_detected = max_psi > adaptive_threshold
        drift_type = DriftType.DATA_DRIFT if drift_detected else None
        
        return float(overall_score), drift_detected, drift_type
    
    def _generate_alerts(
        self,
        model_id: str,
        overall_score: float,
        feature_drift: Dict,
        concept_drift: Optional[Dict],
        prediction_drift: Optional[Dict]
    ) -> List[DriftAlert]:
        """Generate alerts based on drift analysis"""
        alerts = []
        
        # Check cooldown
        last_alert = self.last_alert_time.get(model_id)
        if last_alert and (datetime.now() - last_alert).total_seconds() < self.alert_cooldown_minutes * 60:
            return alerts
        
        # Feature drift alerts
        for feature_name, metrics in feature_drift.items():
            if metrics['drift_detected']:
                severity = self._determine_severity(metrics['psi_score'])
                
                if severity != DriftSeverity.NONE:
                    alert = DriftAlert(
                        drift_type=DriftType.FEATURE_DRIFT,
                        severity=severity,
                        score=metrics['psi_score'],
                        threshold=self.psi_threshold,
                        timestamp=datetime.now(),
                        model_id=model_id,
                        feature_name=feature_name,
                        details=metrics,
                        recommendations=self._get_feature_recommendations(feature_name, metrics)
                    )
                    alerts.append(alert)
        
        # Concept drift alert
        if concept_drift and concept_drift.get('detected'):
            alert = DriftAlert(
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=DriftSeverity.HIGH,
                score=concept_drift.get('ks_statistic', 0),
                threshold=self.ks_threshold,
                timestamp=datetime.now(),
                model_id=model_id,
                details=concept_drift,
                recommendations=[
                    "Target distribution has changed significantly",
                    "Consider retraining the model with recent data",
                    "Investigate business or environmental changes"
                ]
            )
            alerts.append(alert)
        
        # Prediction drift alert
        if prediction_drift and prediction_drift.get('detected'):
            alert = DriftAlert(
                drift_type=DriftType.PREDICTION_DRIFT,
                severity=DriftSeverity.MEDIUM,
                score=prediction_drift.get('psi_score', 0),
                threshold=self.psi_threshold,
                timestamp=datetime.now(),
                model_id=model_id,
                details=prediction_drift,
                recommendations=[
                    "Model predictions distribution has shifted",
                    "May indicate concept drift or data quality issues",
                    "Monitor model performance metrics closely"
                ]
            )
            alerts.append(alert)
        
        # Update last alert time
        if alerts:
            self.last_alert_time[model_id] = datetime.now()
            self.alert_history[model_id].extend(alerts)
        
        return alerts
    
    def _determine_severity(self, psi_score: float) -> DriftSeverity:
        """Determine severity based on PSI score"""
        if psi_score < 0.1:
            return DriftSeverity.NONE
        elif psi_score < 0.15:
            return DriftSeverity.LOW
        elif psi_score < 0.25:
            return DriftSeverity.MEDIUM
        elif psi_score < 0.4:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def _get_feature_recommendations(self, feature_name: str, metrics: Dict) -> List[str]:
        """Generate feature-specific recommendations"""
        recommendations = []
        
        if metrics['mean_shift'] > 2.0:
            recommendations.append(f"Significant mean shift detected in '{feature_name}'")
            recommendations.append("Check for data collection or preprocessing changes")
        
        if metrics['variance_ratio'] > 2.0 or metrics['variance_ratio'] < 0.5:
            recommendations.append(f"Variance change detected in '{feature_name}'")
            recommendations.append("Investigate potential data quality issues")
        
        if metrics['psi_score'] > 0.25:
            recommendations.append(f"High PSI ({metrics['psi_score']:.3f}) for '{feature_name}'")
            recommendations.append("Consider updating reference distribution")
        
        return recommendations
    
    def _generate_recommendations(
        self, 
        drift_detected: bool,
        feature_drift: Dict,
        overall_score: float
    ) -> List[str]:
        """Generate overall recommendations"""
        recommendations = []
        
        if not drift_detected:
            recommendations.append("No significant drift detected. Continue monitoring.")
            return recommendations
        
        # Count drifted features
        n_drifted = sum(1 for f in feature_drift.values() if f.get('drift_detected', False))
        total_features = len(feature_drift)
        
        if n_drifted > total_features * 0.5:
            recommendations.append("⚠️ CRITICAL: More than 50% of features show drift")
            recommendations.append("Immediate model retraining recommended")
        elif n_drifted > total_features * 0.2:
            recommendations.append("⚠️ WARNING: Significant feature drift detected")
            recommendations.append("Schedule model retraining within 1 week")
        else:
            recommendations.append("ℹ️ Minor drift detected in some features")
            recommendations.append("Continue monitoring, no immediate action required")
        
        # Top drifted features
        sorted_features = sorted(
            feature_drift.items(), 
            key=lambda x: x[1].get('psi_score', 0), 
            reverse=True
        )[:5]
        
        if sorted_features:
            recommendations.append("Top drifted features to investigate:")
            for feature_name, metrics in sorted_features:
                if metrics.get('drift_detected'):
                    recommendations.append(f"  - {feature_name}: PSI={metrics['psi_score']:.4f}")
        
        return recommendations
    
    def _store_drift_record(self, model_id: str, report: DriftReport):
        """Store drift record in history"""
        record = {
            'timestamp': report.timestamp.isoformat(),
            'overall_score': report.overall_drift_score,
            'drift_detected': report.drift_detected,
            'n_drifted_features': sum(
                1 for f in report.feature_drift_scores.values() 
                if f.get('drift_detected', False)
            )
        }
        
        self.drift_history[model_id].append(record)
        
        # Keep only recent history (last 1000 records)
        if len(self.drift_history[model_id]) > 1000:
            self.drift_history[model_id] = self.drift_history[model_id][-1000:]
    
    def _create_empty_report(self, model_id: str) -> DriftReport:
        """Create an empty drift report"""
        return DriftReport(
            model_id=model_id,
            timestamp=datetime.now(),
            overall_drift_score=0.0,
            drift_detected=False,
            drift_type=None,
            feature_drift_scores={},
            statistical_tests={},
            alerts=[],
            recommendations=["No reference data available. Set reference first."]
        )
    
    def get_drift_trend(self, model_id: str, lookback_hours: int = 24) -> Dict:
        """Get drift trend over time"""
        if model_id not in self.drift_history:
            return {'trend': 'unknown', 'data_points': 0}
        
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        
        recent_records = [
            r for r in self.drift_history[model_id]
            if datetime.fromisoformat(r['timestamp']) > cutoff
        ]
        
        if len(recent_records) < 2:
            return {'trend': 'insufficient_data', 'data_points': len(recent_records)}
        
        scores = [r['overall_score'] for r in recent_records]
        
        # Calculate trend
        if len(scores) >= 3:
            recent_avg = np.mean(scores[-3:])
            older_avg = np.mean(scores[:-3]) if len(scores) > 3 else scores[0]
            
            if recent_avg > older_avg * 1.2:
                trend = 'increasing'
            elif recent_avg < older_avg * 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'data_points': len(recent_records),
            'scores': scores,
            'timestamps': [r['timestamp'] for r in recent_records]
        }
    
    def export_report(self, report: DriftReport, format: str = 'json') -> str:
        """Export drift report to specified format"""
        if format == 'json':
            return json.dumps({
                'model_id': report.model_id,
                'timestamp': report.timestamp.isoformat(),
                'overall_drift_score': report.overall_drift_score,
                'drift_detected': report.drift_detected,
                'drift_type': report.drift_type.value if report.drift_type else None,
                'feature_drift_scores': report.feature_drift_scores,
                'statistical_tests': report.statistical_tests,
                'alerts': [a.to_dict() for a in report.alerts],
                'recommendations': report.recommendations,
                'metadata': report.metadata
            }, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
