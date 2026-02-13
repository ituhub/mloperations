# ML Model Monitoring Platform - Model Explainability Module
# =============================================================================
# Comprehensive model explanation and interpretation system
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from datetime import datetime
import logging
import json

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Types of explanations available"""
    SHAP = "shap"
    FEATURE_IMPORTANCE = "feature_importance"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    GRADIENT_IMPORTANCE = "gradient_importance"
    LIME = "lime"
    COUNTERFACTUAL = "counterfactual"
    PARTIAL_DEPENDENCE = "partial_dependence"


@dataclass
class FeatureContribution:
    """Individual feature contribution to prediction"""
    feature_name: str
    contribution: float
    baseline_value: float
    feature_value: float
    direction: str  # 'positive' or 'negative'
    importance_rank: int


@dataclass
class PredictionExplanation:
    """Complete explanation for a single prediction"""
    model_id: str
    prediction_id: str
    timestamp: datetime
    predicted_value: float
    base_value: float
    feature_contributions: List[FeatureContribution]
    top_positive_features: List[str]
    top_negative_features: List[str]
    confidence: Optional[float] = None
    explanation_type: ExplanationType = ExplanationType.SHAP
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'model_id': self.model_id,
            'prediction_id': self.prediction_id,
            'timestamp': self.timestamp.isoformat(),
            'predicted_value': self.predicted_value,
            'base_value': self.base_value,
            'feature_contributions': [
                {
                    'feature_name': fc.feature_name,
                    'contribution': fc.contribution,
                    'baseline_value': fc.baseline_value,
                    'feature_value': fc.feature_value,
                    'direction': fc.direction,
                    'importance_rank': fc.importance_rank
                }
                for fc in self.feature_contributions
            ],
            'top_positive_features': self.top_positive_features,
            'top_negative_features': self.top_negative_features,
            'confidence': self.confidence,
            'explanation_type': self.explanation_type.value,
            'metadata': self.metadata
        }


@dataclass
class GlobalExplanation:
    """Global model explanation"""
    model_id: str
    timestamp: datetime
    feature_importance: Dict[str, float]
    top_features: List[str]
    feature_interactions: Dict[str, float]
    explanation_type: ExplanationType
    n_samples_used: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelExplainer:
    """
    Comprehensive model explainability system.
    
    Supports:
    - SHAP explanations (for various model types)
    - Permutation importance
    - Gradient-based explanations (for neural networks)
    - Feature importance tracking over time
    - Natural language explanations
    """
    
    def __init__(
        self,
        background_samples: int = 100,
        n_permutations: int = 10,
        enable_caching: bool = True,
        cache_size: int = 1000
    ):
        self.background_samples = background_samples
        self.n_permutations = n_permutations
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Explainer cache
        self.shap_explainers: Dict[str, Any] = {}
        self.explanation_cache: Dict[str, PredictionExplanation] = {}
        
        # Feature importance history
        self.feature_importance_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Background data
        self.background_data: Dict[str, np.ndarray] = {}
        
        logger.info(f"Initialized ModelExplainer (SHAP available: {SHAP_AVAILABLE})")
    
    def set_background_data(
        self, 
        model_id: str, 
        X_background: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """Set background data for explanations"""
        # Use random sample if too large
        if len(X_background) > self.background_samples:
            indices = np.random.choice(
                len(X_background), 
                self.background_samples, 
                replace=False
            )
            X_background = X_background[indices]
        
        self.background_data[model_id] = {
            'X': X_background,
            'feature_names': feature_names or [f'feature_{i}' for i in range(X_background.shape[-1])]
        }
        
        # Clear cached explainer for this model
        if model_id in self.shap_explainers:
            del self.shap_explainers[model_id]
        
        logger.info(f"Set background data for model '{model_id}' with {len(X_background)} samples")
    
    def explain_prediction(
        self,
        model_id: str,
        model: Any,
        X: np.ndarray,
        prediction_id: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        methods: Optional[List[ExplanationType]] = None
    ) -> PredictionExplanation:
        """
        Generate explanation for a prediction.
        
        Args:
            model_id: Model identifier
            model: The model object
            X: Input features (single sample or batch)
            prediction_id: Optional prediction identifier
            feature_names: Optional feature names
            methods: Explanation methods to use
            
        Returns:
            PredictionExplanation object
        """
        if methods is None:
            methods = [ExplanationType.PERMUTATION_IMPORTANCE]
            if SHAP_AVAILABLE:
                methods.append(ExplanationType.SHAP)
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Handle 3D data (sequences)
        original_shape = X.shape
        if len(X.shape) == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        # Get feature names
        if feature_names is None:
            if model_id in self.background_data:
                feature_names = self.background_data[model_id]['feature_names']
            else:
                feature_names = [f'feature_{i}' for i in range(X_flat.shape[1])]
        
        # Get prediction
        try:
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                model.eval()
                with torch.no_grad():
                    pred_tensor = model(torch.tensor(X, dtype=torch.float32))
                    predicted_value = float(pred_tensor.numpy().flatten()[0])
            elif hasattr(model, 'predict'):
                predicted_value = float(model.predict(X_flat)[0])
            else:
                predicted_value = 0.0
        except Exception as e:
            logger.warning(f"Error getting prediction: {e}")
            predicted_value = 0.0
        
        # Calculate explanations
        contributions = []
        
        if ExplanationType.SHAP in methods and SHAP_AVAILABLE:
            shap_contributions = self._get_shap_explanation(
                model_id, model, X, X_flat, feature_names, original_shape
            )
            if shap_contributions:
                contributions.extend(shap_contributions)
        
        if ExplanationType.PERMUTATION_IMPORTANCE in methods:
            perm_contributions = self._get_permutation_importance(
                model_id, model, X, X_flat, feature_names, original_shape
            )
            if perm_contributions:
                contributions.extend(perm_contributions)
        
        if ExplanationType.GRADIENT_IMPORTANCE in methods and TORCH_AVAILABLE:
            if isinstance(model, torch.nn.Module):
                grad_contributions = self._get_gradient_importance(
                    model, X, feature_names
                )
                if grad_contributions:
                    contributions.extend(grad_contributions)
        
        # Aggregate contributions (average if multiple methods)
        aggregated = self._aggregate_contributions(contributions, feature_names)
        
        # Sort by absolute contribution
        aggregated.sort(key=lambda x: abs(x.contribution), reverse=True)
        
        # Update ranks
        for i, fc in enumerate(aggregated):
            fc.importance_rank = i + 1
        
        # Get top positive and negative features
        positive = [fc.feature_name for fc in aggregated if fc.contribution > 0][:5]
        negative = [fc.feature_name for fc in aggregated if fc.contribution < 0][:5]
        
        # Calculate base value
        base_value = predicted_value - sum(fc.contribution for fc in aggregated)
        
        explanation = PredictionExplanation(
            model_id=model_id,
            prediction_id=prediction_id or f"pred_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            predicted_value=predicted_value,
            base_value=base_value,
            feature_contributions=aggregated,
            top_positive_features=positive,
            top_negative_features=negative,
            explanation_type=methods[0] if methods else ExplanationType.PERMUTATION_IMPORTANCE
        )
        
        # Cache explanation
        if self.enable_caching:
            self._cache_explanation(explanation)
        
        return explanation
    
    def _get_shap_explanation(
        self,
        model_id: str,
        model: Any,
        X: np.ndarray,
        X_flat: np.ndarray,
        feature_names: List[str],
        original_shape: tuple
    ) -> List[FeatureContribution]:
        """Get SHAP-based explanations"""
        try:
            # Get or create explainer
            if model_id not in self.shap_explainers:
                background = self.background_data.get(model_id, {}).get('X')
                
                if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                    if background is not None:
                        # For neural networks, use DeepExplainer
                        if len(original_shape) == 3:
                            bg_tensor = torch.tensor(
                                background[:min(50, len(background))], 
                                dtype=torch.float32
                            )
                        else:
                            bg_tensor = torch.tensor(
                                background[:min(50, len(background))], 
                                dtype=torch.float32
                            )
                        self.shap_explainers[model_id] = shap.DeepExplainer(model, bg_tensor)
                    else:
                        return []
                elif hasattr(model, 'feature_importances_'):
                    # Tree-based model
                    self.shap_explainers[model_id] = shap.TreeExplainer(model)
                elif background is not None:
                    # Kernel SHAP for other models
                    self.shap_explainers[model_id] = shap.KernelExplainer(
                        model.predict, 
                        background[:min(100, len(background))]
                    )
                else:
                    return []
            
            explainer = self.shap_explainers[model_id]
            
            # Calculate SHAP values
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                X_tensor = torch.tensor(X, dtype=torch.float32)
                shap_values = explainer.shap_values(X_tensor)
            else:
                shap_values = explainer.shap_values(X_flat[:1])
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            if len(shap_values.shape) > 2:
                # Aggregate over time dimension for sequences
                shap_values = np.mean(shap_values, axis=1)
            
            shap_values = shap_values.flatten()
            
            # Create contributions
            contributions = []
            for i, (fname, sv) in enumerate(zip(feature_names[:len(shap_values)], shap_values)):
                contributions.append(FeatureContribution(
                    feature_name=fname,
                    contribution=float(sv),
                    baseline_value=0.0,
                    feature_value=float(X_flat[0, i]) if i < X_flat.shape[1] else 0.0,
                    direction='positive' if sv > 0 else 'negative',
                    importance_rank=0
                ))
            
            return contributions
            
        except Exception as e:
            logger.warning(f"Error computing SHAP values: {e}")
            return []
    
    def _get_permutation_importance(
        self,
        model_id: str,
        model: Any,
        X: np.ndarray,
        X_flat: np.ndarray,
        feature_names: List[str],
        original_shape: tuple
    ) -> List[FeatureContribution]:
        """Calculate permutation importance"""
        try:
            # Get baseline prediction
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                model.eval()
                with torch.no_grad():
                    baseline = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()[0]
            elif hasattr(model, 'predict'):
                baseline = model.predict(X_flat)[0]
            else:
                return []
            
            contributions = []
            n_features = min(len(feature_names), X_flat.shape[1])
            
            for i in range(n_features):
                importance_scores = []
                
                for _ in range(self.n_permutations):
                    X_permuted = X_flat.copy()
                    np.random.shuffle(X_permuted[:, i])
                    
                    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                        if len(original_shape) == 3:
                            X_perm_reshaped = X_permuted.reshape(original_shape)
                            X_tensor = torch.tensor(X_perm_reshaped, dtype=torch.float32)
                        else:
                            X_tensor = torch.tensor(X_permuted, dtype=torch.float32)
                        
                        with torch.no_grad():
                            permuted_pred = model(X_tensor).numpy().flatten()[0]
                    else:
                        permuted_pred = model.predict(X_permuted)[0]
                    
                    importance_scores.append(abs(baseline - permuted_pred))
                
                avg_importance = np.mean(importance_scores)
                
                contributions.append(FeatureContribution(
                    feature_name=feature_names[i],
                    contribution=float(avg_importance),
                    baseline_value=float(baseline),
                    feature_value=float(X_flat[0, i]),
                    direction='positive' if avg_importance > 0 else 'negative',
                    importance_rank=0
                ))
            
            return contributions
            
        except Exception as e:
            logger.warning(f"Error calculating permutation importance: {e}")
            return []
    
    def _get_gradient_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureContribution]:
        """Calculate gradient-based importance for neural networks"""
        if not TORCH_AVAILABLE:
            return []
        
        try:
            model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
            
            output = model(X_tensor)
            output.backward(torch.ones_like(output))
            
            gradients = X_tensor.grad.abs()
            
            # Aggregate gradients
            if len(gradients.shape) == 3:
                # For sequences, average over time
                gradients = gradients.mean(dim=1)
            
            gradients = gradients.mean(dim=0).numpy()
            
            contributions = []
            for i, (fname, grad) in enumerate(zip(feature_names[:len(gradients)], gradients)):
                contributions.append(FeatureContribution(
                    feature_name=fname,
                    contribution=float(grad),
                    baseline_value=0.0,
                    feature_value=float(X.flatten()[i]) if i < len(X.flatten()) else 0.0,
                    direction='positive',
                    importance_rank=0
                ))
            
            return contributions
            
        except Exception as e:
            logger.warning(f"Error calculating gradient importance: {e}")
            return []
    
    def _aggregate_contributions(
        self,
        contributions: List[FeatureContribution],
        feature_names: List[str]
    ) -> List[FeatureContribution]:
        """Aggregate contributions from multiple methods"""
        if not contributions:
            return []
        
        # Group by feature name
        feature_scores: Dict[str, List[float]] = defaultdict(list)
        feature_values: Dict[str, float] = {}
        
        for fc in contributions:
            feature_scores[fc.feature_name].append(fc.contribution)
            feature_values[fc.feature_name] = fc.feature_value
        
        # Average contributions
        aggregated = []
        for fname in feature_names:
            if fname in feature_scores:
                avg_contribution = np.mean(feature_scores[fname])
                aggregated.append(FeatureContribution(
                    feature_name=fname,
                    contribution=float(avg_contribution),
                    baseline_value=0.0,
                    feature_value=feature_values.get(fname, 0.0),
                    direction='positive' if avg_contribution > 0 else 'negative',
                    importance_rank=0
                ))
        
        return aggregated
    
    def _cache_explanation(self, explanation: PredictionExplanation):
        """Cache explanation for future use"""
        cache_key = f"{explanation.model_id}_{explanation.prediction_id}"
        self.explanation_cache[cache_key] = explanation
        
        # Enforce cache size limit
        if len(self.explanation_cache) > self.cache_size:
            # Remove oldest entries
            keys_to_remove = list(self.explanation_cache.keys())[:-self.cache_size]
            for key in keys_to_remove:
                del self.explanation_cache[key]
    
    def get_global_explanation(
        self,
        model_id: str,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 100
    ) -> GlobalExplanation:
        """
        Generate global model explanation.
        
        Args:
            model_id: Model identifier
            model: The model object
            X: Sample data
            feature_names: Feature names
            n_samples: Number of samples to use
            
        Returns:
            GlobalExplanation object
        """
        # Sample data if needed
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Flatten if needed
        if len(X_sample.shape) == 3:
            X_flat = X_sample.reshape(X_sample.shape[0], -1)
        else:
            X_flat = X_sample
        
        # Get feature names
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_flat.shape[1])]
        
        # Calculate global feature importance
        feature_importance = {}
        
        # Try tree-based importance first
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, imp in enumerate(importances):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = float(imp)
        
        # Permutation importance
        else:
            try:
                if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                    model.eval()
                    with torch.no_grad():
                        baseline_preds = model(torch.tensor(X_sample, dtype=torch.float32)).numpy()
                else:
                    baseline_preds = model.predict(X_flat)
                
                baseline_variance = np.var(baseline_preds)
                
                for i in range(min(len(feature_names), X_flat.shape[1])):
                    X_permuted = X_flat.copy()
                    np.random.shuffle(X_permuted[:, i])
                    
                    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                        permuted_preds = model(torch.tensor(X_permuted, dtype=torch.float32)).numpy()
                    else:
                        permuted_preds = model.predict(X_permuted)
                    
                    importance = np.mean(np.abs(baseline_preds.flatten() - permuted_preds.flatten()))
                    feature_importance[feature_names[i]] = float(importance)
                    
            except Exception as e:
                logger.warning(f"Error calculating global importance: {e}")
        
        # Get top features
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        top_features = [f[0] for f in sorted_features[:10]]
        
        # Store in history
        self.feature_importance_history[model_id].append({
            'timestamp': datetime.now().isoformat(),
            'importance': feature_importance
        })
        
        return GlobalExplanation(
            model_id=model_id,
            timestamp=datetime.now(),
            feature_importance=feature_importance,
            top_features=top_features,
            feature_interactions={},  # Can be extended
            explanation_type=ExplanationType.PERMUTATION_IMPORTANCE,
            n_samples_used=len(X_sample)
        )
    
    def generate_text_explanation(
        self,
        explanation: PredictionExplanation,
        include_details: bool = True
    ) -> str:
        """Generate human-readable explanation text"""
        lines = []
        
        lines.append(f"=== Prediction Explanation ===")
        lines.append(f"Model: {explanation.model_id}")
        lines.append(f"Predicted Value: {explanation.predicted_value:.4f}")
        lines.append(f"Base Value: {explanation.base_value:.4f}")
        lines.append("")
        
        # Top contributing features
        lines.append("Top Positive Contributors:")
        for fc in explanation.feature_contributions[:5]:
            if fc.contribution > 0:
                lines.append(f"  + {fc.feature_name}: {fc.contribution:+.4f}")
        
        lines.append("")
        lines.append("Top Negative Contributors:")
        for fc in explanation.feature_contributions:
            if fc.contribution < 0:
                lines.append(f"  - {fc.feature_name}: {fc.contribution:+.4f}")
                if len([f for f in explanation.feature_contributions if f.contribution < 0]) >= 5:
                    break
        
        if include_details:
            lines.append("")
            lines.append("All Feature Contributions:")
            for fc in explanation.feature_contributions:
                lines.append(
                    f"  {fc.importance_rank}. {fc.feature_name}: "
                    f"{fc.contribution:+.4f} (value: {fc.feature_value:.4f})"
                )
        
        return "\n".join(lines)
    
    def get_feature_importance_trend(
        self,
        model_id: str,
        feature_name: str,
        lookback: int = 100
    ) -> Dict:
        """Get importance trend for a specific feature over time"""
        history = self.feature_importance_history.get(model_id, [])
        
        if not history:
            return {'trend': 'no_data', 'values': []}
        
        recent = history[-lookback:]
        values = [
            h['importance'].get(feature_name, 0) 
            for h in recent
        ]
        timestamps = [h['timestamp'] for h in recent]
        
        if len(values) < 2:
            return {'trend': 'insufficient_data', 'values': values}
        
        # Calculate trend
        recent_avg = np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
        older_avg = np.mean(values[:-10]) if len(values) > 10 else values[0]
        
        if recent_avg > older_avg * 1.2:
            trend = 'increasing'
        elif recent_avg < older_avg * 0.8:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'values': values,
            'timestamps': timestamps,
            'current_importance': values[-1] if values else 0
        }
