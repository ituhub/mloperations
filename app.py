# ML Model Monitoring Platform - Enhanced Version
# =============================================================================
# Professional ML monitoring dashboard with advanced features
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional
import sys
import os

# Page configuration
st.set_page_config(
    page_title="ML Model Monitor Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ENHANCED CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Main Theme */
    :root {
        --primary-color: #4F46E5;
        --secondary-color: #7C3AED;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --info-color: #3B82F6;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #E5E7EB;
        height: 100%;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #4F46E5;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.15);
        transform: translateX(5px);
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
    }
    
    .stat-card-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2937;
    }
    
    .stat-card-label {
        font-size: 0.875rem;
        color: #6B7280;
        margin-top: 0.25rem;
    }
    
    /* Status Cards */
    .status-card-healthy {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border-left: 4px solid #10B981;
        border-radius: 12px;
        padding: 1.25rem;
    }
    
    .status-card-warning {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 4px solid #F59E0B;
        border-radius: 12px;
        padding: 1.25rem;
    }
    
    .status-card-critical {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left: 4px solid #EF4444;
        border-radius: 12px;
        padding: 1.25rem;
    }
    
    /* Alert Styles */
    .alert-critical {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left: 5px solid #EF4444;
        padding: 1.25rem;
        margin: 0.75rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2);
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 5px solid #F59E0B;
        padding: 1.25rem;
        margin: 0.75rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);
    }
    
    .alert-info {
        background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
        border-left: 5px solid #3B82F6;
        padding: 1.25rem;
        margin: 0.75rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
    }
    
    /* Model Card */
    .model-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #E5E7EB;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .model-card:hover {
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        transform: translateY(-3px);
    }
    
    .model-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .model-card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1F2937;
    }
    
    /* Status Badges */
    .badge-healthy {
        background: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-warning {
        background: #FEF3C7;
        color: #92400E;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-critical {
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Tabs Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #F3F4F6;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Benefit Card */
    .benefit-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .benefit-card:hover {
        box-shadow: 0 12px 40px rgba(79, 70, 229, 0.15);
        transform: translateY(-5px);
        border-color: #4F46E5;
    }
    
    .benefit-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .benefit-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 0.5rem;
    }
    
    .benefit-description {
        color: #6B7280;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Step Card */
    .step-card {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border-radius: 16px;
        padding: 1.5rem;
        position: relative;
        margin-left: 2rem;
    }
    
    .step-number {
        position: absolute;
        left: -2rem;
        top: 50%;
        transform: translateY(-50%);
        width: 3rem;
        height: 3rem;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1.25rem;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
    }
    
    /* Anomaly Indicator */
    .anomaly-high {
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .anomaly-normal {
        background: #D1FAE5;
        color: #065F46;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F3F4F6;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #D1D5DB;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #9CA3AF;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DEMO DATA GENERATION
# =============================================================================

def generate_demo_models():
    """Generate demo model configurations"""
    return {
        'model_001': {
            'name': 'Customer Churn Predictor',
            'type': 'Classification',
            'framework': 'XGBoost',
            'version': 'v2.3.1',
            'deployed_at': datetime.now() - timedelta(days=45),
            'last_prediction': datetime.now() - timedelta(minutes=2),
            'status': 'healthy',
            'predictions_today': 15420,
            'predictions_total': 2450000,
            'avg_latency_ms': 23.5,
            'accuracy': 0.94,
            'f1_score': 0.91,
            'data_size_mb': 245
        },
        'model_002': {
            'name': 'Revenue Forecaster',
            'type': 'Regression',
            'framework': 'LightGBM',
            'version': 'v1.8.0',
            'deployed_at': datetime.now() - timedelta(days=120),
            'last_prediction': datetime.now() - timedelta(minutes=1),
            'status': 'warning',
            'predictions_today': 8930,
            'predictions_total': 1890000,
            'avg_latency_ms': 45.2,
            'mae': 0.0234,
            'r2_score': 0.89,
            'data_size_mb': 512
        },
        'model_003': {
            'name': 'Fraud Detection',
            'type': 'Classification',
            'framework': 'PyTorch',
            'version': 'v3.1.2',
            'deployed_at': datetime.now() - timedelta(days=30),
            'last_prediction': datetime.now() - timedelta(seconds=30),
            'status': 'healthy',
            'predictions_today': 125000,
            'predictions_total': 8900000,
            'avg_latency_ms': 12.1,
            'accuracy': 0.987,
            'precision': 0.95,
            'data_size_mb': 890
        },
        'model_004': {
            'name': 'Demand Predictor',
            'type': 'Regression',
            'framework': 'TensorFlow',
            'version': 'v2.0.0',
            'deployed_at': datetime.now() - timedelta(days=15),
            'last_prediction': datetime.now() - timedelta(minutes=5),
            'status': 'critical',
            'predictions_today': 4520,
            'predictions_total': 340000,
            'avg_latency_ms': 67.8,
            'mae': 0.0891,
            'r2_score': 0.72,
            'data_size_mb': 156
        }
    }


def generate_demo_metrics(model_id: str, hours: int = 24) -> pd.DataFrame:
    """Generate demo metrics for a model"""
    np.random.seed(hash(model_id) % 2**32)
    
    timestamps = pd.date_range(
        end=datetime.now(),
        periods=hours * 6,
        freq='10min'
    )
    
    base_mae = 0.05 + np.random.random() * 0.02
    base_accuracy = 0.92 + np.random.random() * 0.05
    
    drift_factor = np.linspace(0, 0.02 if 'model_004' in model_id else 0.005, len(timestamps))
    
    data = {
        'timestamp': timestamps,
        'mae': base_mae + np.random.normal(0, 0.005, len(timestamps)) + drift_factor,
        'rmse': (base_mae + np.random.normal(0, 0.008, len(timestamps)) + drift_factor) * 1.2,
        'accuracy': np.clip(base_accuracy - drift_factor + np.random.normal(0, 0.01, len(timestamps)), 0.8, 1.0),
        'latency_ms': 20 + np.random.exponential(10, len(timestamps)),
        'predictions_count': np.random.poisson(100, len(timestamps)),
        'error_rate': np.clip(0.001 + np.random.exponential(0.002, len(timestamps)), 0, 0.1),
        'memory_usage_mb': 200 + np.random.normal(0, 20, len(timestamps)),
        'cpu_usage_percent': 30 + np.random.normal(0, 10, len(timestamps))
    }
    
    return pd.DataFrame(data)


def generate_demo_drift_data(model_id: str) -> Dict:
    """Generate demo drift detection data"""
    np.random.seed(hash(model_id) % 2**32)
    
    features = ['age', 'income', 'tenure', 'usage_frequency', 'support_tickets', 
                'last_purchase_days', 'account_balance', 'num_products']
    
    is_critical = 'model_004' in model_id
    
    drift_scores = {}
    for feature in features:
        if is_critical and feature in ['income', 'tenure']:
            psi = np.random.uniform(0.15, 0.35)
        else:
            psi = np.random.exponential(0.04)
        
        drift_scores[feature] = {
            'psi_score': psi,
            'ks_statistic': np.random.random() * 0.3,
            'ks_p_value': np.random.random(),
            'drift_detected': psi > 0.1,
            'mean_shift': np.random.normal(0, 0.5),
            'variance_ratio': 0.8 + np.random.random() * 0.4,
            'wasserstein_distance': np.random.exponential(0.1),
            'js_divergence': np.random.exponential(0.05)
        }
    
    return {
        'overall_score': np.mean([d['psi_score'] for d in drift_scores.values()]),
        'drift_detected': any(d['drift_detected'] for d in drift_scores.values()),
        'feature_drift': drift_scores,
        'timestamp': datetime.now().isoformat(),
        'n_features_drifted': sum(1 for d in drift_scores.values() if d['drift_detected'])
    }


def generate_demo_alerts(model_id: str) -> List[Dict]:
    """Generate demo alerts"""
    alerts = []
    
    if 'model_002' in model_id:
        alerts.append({
            'alert_id': 'alert_001',
            'severity': 'medium',
            'category': 'drift',
            'title': 'Minor Data Drift Detected',
            'message': 'Feature "usage_frequency" showing slight distribution shift (PSI: 0.12)',
            'timestamp': datetime.now() - timedelta(hours=3),
            'status': 'active',
            'recommendations': ['Monitor for next 24 hours', 'Review data pipeline']
        })
    
    if 'model_004' in model_id:
        alerts.extend([
            {
                'alert_id': 'alert_002',
                'severity': 'critical',
                'category': 'performance',
                'title': 'Severe Model Performance Degradation',
                'message': 'MAE increased by 45% over the last 24 hours. Model predictions may be unreliable.',
                'timestamp': datetime.now() - timedelta(hours=1),
                'status': 'active',
                'recommendations': ['Immediate investigation required', 'Consider rollback to v1.9.0', 'Check for data quality issues']
            },
            {
                'alert_id': 'alert_003',
                'severity': 'high',
                'category': 'drift',
                'title': 'Significant Data Drift Detected',
                'message': 'Multiple features showing drift: income (PSI: 0.28), tenure (PSI: 0.19)',
                'timestamp': datetime.now() - timedelta(hours=2),
                'status': 'active',
                'recommendations': ['Retrain model with recent data', 'Investigate upstream data changes']
            },
            {
                'alert_id': 'alert_004',
                'severity': 'high',
                'category': 'latency',
                'title': 'SLA Violation - High Latency',
                'message': 'P95 latency (89ms) exceeded SLA threshold (50ms) for the past 30 minutes',
                'timestamp': datetime.now() - timedelta(minutes=30),
                'status': 'active',
                'recommendations': ['Check infrastructure resources', 'Consider model optimization']
            }
        ])
    
    return alerts


def generate_demo_explanations() -> Dict:
    """Generate demo feature importance"""
    features = ['income', 'tenure', 'usage_frequency', 'age', 'support_tickets',
                'last_purchase_days', 'account_balance', 'num_products']
    
    importances = np.random.dirichlet(np.ones(len(features)) * 2)
    
    return {
        feature: {
            'importance': float(imp),
            'direction': 'positive' if np.random.random() > 0.3 else 'negative',
            'stability': 'stable' if np.random.random() > 0.2 else 'volatile'
        }
        for feature, imp in zip(features, importances)
    }


def generate_anomaly_data(model_id: str) -> Dict:
    """Generate anomaly detection results"""
    np.random.seed(hash(model_id) % 2**32)
    
    is_critical = 'model_004' in model_id
    
    anomalies = {
        'total_predictions': np.random.randint(5000, 15000),
        'anomalies_detected': np.random.randint(50, 200) if is_critical else np.random.randint(5, 30),
        'anomaly_rate': 0.025 if is_critical else 0.003,
        'top_anomalies': [
            {
                'timestamp': (datetime.now() - timedelta(hours=np.random.randint(1, 24))).isoformat(),
                'prediction_id': f'pred_{np.random.randint(10000, 99999)}',
                'anomaly_score': np.random.uniform(0.85, 0.99),
                'reason': np.random.choice([
                    'Unusual feature combination',
                    'Out-of-distribution input',
                    'Extreme prediction value',
                    'High uncertainty score'
                ])
            }
            for _ in range(5)
        ],
        'anomaly_trend': 'increasing' if is_critical else 'stable'
    }
    
    return anomalies


def generate_data_profile(model_id: str) -> Dict:
    """Generate data quality profile"""
    features = ['age', 'income', 'tenure', 'usage_frequency', 'support_tickets']
    
    profile = {
        'total_records': np.random.randint(50000, 200000),
        'time_range': '24 hours',
        'features': {}
    }
    
    for feature in features:
        profile['features'][feature] = {
            'missing_rate': np.random.uniform(0, 0.05),
            'mean': np.random.uniform(10, 100),
            'std': np.random.uniform(5, 30),
            'min': np.random.uniform(0, 10),
            'max': np.random.uniform(100, 500),
            'unique_count': np.random.randint(50, 1000),
            'outlier_rate': np.random.uniform(0, 0.03)
        }
    
    profile['overall_quality_score'] = np.random.uniform(0.85, 0.98)
    
    return profile


# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

def render_header():
    """Render dashboard header"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<p class="main-header">🔍 ML Model Monitor Pro</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Enterprise-grade ML monitoring with drift detection, explainability & intelligent alerting</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: right; padding-top: 1rem;">
            <span style="background: #E0E7FF; color: #4338CA; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.85rem;">
                🟢 System Online
            </span>
            <p style="color: #6B7280; font-size: 0.85rem; margin-top: 0.5rem;">
                Last refresh: {datetime.now().strftime('%H:%M:%S')}
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_overview_cards(models: Dict):
    """Render overview metric cards"""
    total_models = len(models)
    healthy_models = sum(1 for m in models.values() if m['status'] == 'healthy')
    warning_models = sum(1 for m in models.values() if m['status'] == 'warning')
    critical_models = sum(1 for m in models.values() if m['status'] == 'critical')
    total_predictions = sum(m['predictions_today'] for m in models.values())
    avg_latency = np.mean([m['avg_latency_ms'] for m in models.values()])
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{total_models}</div>
            <div class="stat-card-label">📊 Total Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="border-left: 4px solid #10B981;">
            <div class="stat-card-value" style="color: #10B981;">{healthy_models}</div>
            <div class="stat-card-label">✅ Healthy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card" style="border-left: 4px solid #F59E0B;">
            <div class="stat-card-value" style="color: #F59E0B;">{warning_models}</div>
            <div class="stat-card-label">⚠️ Warning</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card" style="border-left: 4px solid #EF4444;">
            <div class="stat-card-value" style="color: #EF4444;">{critical_models}</div>
            <div class="stat-card-label">🚨 Critical</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{total_predictions:,}</div>
            <div class="stat-card-label">🎯 Predictions Today</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{avg_latency:.1f}ms</div>
            <div class="stat-card-label">⚡ Avg Latency</div>
        </div>
        """, unsafe_allow_html=True)


def render_model_cards(models: Dict):
    """Render model status cards"""
    st.markdown("### 📋 Model Fleet Overview")
    
    cols = st.columns(len(models))
    
    for idx, (model_id, model_info) in enumerate(models.items()):
        with cols[idx]:
            status_class = f"status-card-{model_info['status']}"
            badge_class = f"badge-{model_info['status']}"
            
            status_emoji = {'healthy': '🟢', 'warning': '🟡', 'critical': '🔴'}
            
            # Calculate uptime (demo)
            uptime_days = (datetime.now() - model_info['deployed_at']).days
            
            st.markdown(f"""
            <div class="model-card">
                <div class="model-card-header">
                    <span class="model-card-title">{model_info['name']}</span>
                    <span class="{badge_class}">{status_emoji[model_info['status']]} {model_info['status'].upper()}</span>
                </div>
                <hr style="margin: 0.75rem 0; border-color: #E5E7EB;">
                <div style="font-size: 0.9rem; color: #4B5563;">
                    <p style="margin: 0.4rem 0;"><strong>🏷️ Type:</strong> {model_info['type']}</p>
                    <p style="margin: 0.4rem 0;"><strong>🔧 Framework:</strong> {model_info['framework']}</p>
                    <p style="margin: 0.4rem 0;"><strong>📦 Version:</strong> {model_info['version']}</p>
                    <p style="margin: 0.4rem 0;"><strong>📈 Today:</strong> {model_info['predictions_today']:,} predictions</p>
                    <p style="margin: 0.4rem 0;"><strong>⚡ Latency:</strong> {model_info['avg_latency_ms']:.1f}ms</p>
                    <p style="margin: 0.4rem 0;"><strong>📅 Uptime:</strong> {uptime_days} days</p>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_performance_charts(model_id: str, metrics_df: pd.DataFrame):
    """Render performance metrics charts with cards"""
    st.markdown("### 📈 Performance Metrics")
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    latest = metrics_df.iloc[-1]
    prev_24h = metrics_df.iloc[0]
    
    with col1:
        mae_change = ((latest['mae'] - prev_24h['mae']) / prev_24h['mae']) * 100
        st.metric("Mean Absolute Error", f"{latest['mae']:.4f}", f"{mae_change:+.1f}%", delta_color="inverse")
    
    with col2:
        acc_change = ((latest['accuracy'] - prev_24h['accuracy']) / prev_24h['accuracy']) * 100
        st.metric("Accuracy", f"{latest['accuracy']:.2%}", f"{acc_change:+.1f}%")
    
    with col3:
        lat_change = ((latest['latency_ms'] - prev_24h['latency_ms']) / prev_24h['latency_ms']) * 100
        st.metric("P50 Latency", f"{latest['latency_ms']:.1f}ms", f"{lat_change:+.1f}%", delta_color="inverse")
    
    with col4:
        st.metric("Error Rate", f"{latest['error_rate']:.2%}", "Stable")
    
    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Error Metrics", "⚡ Latency", "📊 Throughput", "💻 Resources"])
    
    with tab1:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('MAE Over Time', 'Accuracy Over Time'),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df['mae'],
                mode='lines',
                name='MAE',
                line=dict(color='#4F46E5', width=2),
                fill='tozeroy',
                fillcolor='rgba(79, 70, 229, 0.1)'
            ),
            row=1, col=1
        )
        
        fig.add_hline(y=0.07, line_dash="dash", line_color="#EF4444", 
                      annotation_text="Threshold", row=1, col=1)
        
        fig.add_trace(
            go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df['accuracy'],
                mode='lines',
                name='Accuracy',
                line=dict(color='#10B981', width=2),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=True, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df['latency_ms'],
                mode='lines',
                name='Latency',
                line=dict(color='#8B5CF6', width=2),
                fill='tozeroy',
                fillcolor='rgba(139, 92, 246, 0.1)'
            ))
            
            fig.add_hline(y=50, line_dash="dash", line_color="#EF4444",
                          annotation_text="SLA Limit (50ms)")
            
            fig.update_layout(
                title='Prediction Latency (ms)',
                height=350,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Latency Percentiles")
            percentiles = metrics_df['latency_ms'].quantile([0.5, 0.95, 0.99])
            
            for p, label in [(0.5, 'P50'), (0.95, 'P95'), (0.99, 'P99')]:
                val = percentiles[p]
                color = '#10B981' if val < 50 else ('#F59E0B' if val < 100 else '#EF4444')
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, {color}20 0%, white 100%); 
                            padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
                            border-left: 4px solid {color};">
                    <span style="font-weight: 600;">{label}</span>: {val:.1f}ms
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics_df['timestamp'],
            y=metrics_df['predictions_count'],
            name='Predictions',
            marker_color='#6366F1'
        ))
        
        fig.update_layout(
            title='Predictions Per Interval',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df['memory_usage_mb'],
                mode='lines',
                name='Memory',
                line=dict(color='#F59E0B', width=2),
                fill='tozeroy',
                fillcolor='rgba(245, 158, 11, 0.1)'
            ))
            fig.update_layout(title='Memory Usage (MB)', height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df['cpu_usage_percent'],
                mode='lines',
                name='CPU',
                line=dict(color='#EC4899', width=2),
                fill='tozeroy',
                fillcolor='rgba(236, 72, 153, 0.1)'
            ))
            fig.update_layout(title='CPU Usage (%)', height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)


def render_drift_analysis(model_id: str, drift_data: Dict):
    """Render drift detection analysis with cards"""
    st.markdown("### 🔄 Data Drift Analysis")
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "🚨 DRIFT DETECTED" if drift_data['drift_detected'] else "✅ NO DRIFT"
        status_color = "#EF4444" if drift_data['drift_detected'] else "#10B981"
        st.markdown(f"""
        <div class="stat-card" style="border-left: 4px solid {status_color};">
            <div style="color: {status_color}; font-weight: 700; font-size: 1.1rem;">{status}</div>
            <div class="stat-card-label">Overall Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{drift_data['overall_score']:.4f}</div>
            <div class="stat-card-label">Overall Drift Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{drift_data['n_features_drifted']}</div>
            <div class="stat-card-label">Features Drifted</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{len(drift_data['feature_drift'])}</div>
            <div class="stat-card-label">Features Monitored</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        feature_drift = drift_data['feature_drift']
        features = list(feature_drift.keys())
        psi_scores = [feature_drift[f]['psi_score'] for f in features]
        colors = ['#EF4444' if feature_drift[f]['drift_detected'] else '#10B981' for f in features]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=features,
            y=psi_scores,
            marker_color=colors,
            text=[f"{s:.3f}" for s in psi_scores],
            textposition='outside'
        ))
        
        fig.add_hline(y=0.1, line_dash="dash", line_color="#F59E0B",
                      annotation_text="Warning (0.1)")
        fig.add_hline(y=0.2, line_dash="dash", line_color="#EF4444",
                      annotation_text="Critical (0.2)")
        
        fig.update_layout(
            title='Feature PSI Scores',
            height=400,
            template='plotly_white',
            yaxis_title='PSI Score'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Drifted Features")
        
        drifted = [(f, d) for f, d in feature_drift.items() if d['drift_detected']]
        
        if drifted:
            for f, d in sorted(drifted, key=lambda x: x[1]['psi_score'], reverse=True):
                severity_color = '#EF4444' if d['psi_score'] > 0.2 else '#F59E0B'
                st.markdown(f"""
                <div style="background: {severity_color}15; padding: 1rem; border-radius: 8px; 
                            margin: 0.5rem 0; border-left: 4px solid {severity_color};">
                    <strong>{f}</strong><br>
                    <span style="color: #6B7280; font-size: 0.9rem;">
                        PSI: {d['psi_score']:.4f}<br>
                        KS: {d['ks_statistic']:.4f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("All features within normal range! 🎉")
    
    # Detailed Table
    with st.expander("📊 Detailed Drift Metrics"):
        drift_df = pd.DataFrame([
            {
                'Feature': f,
                'PSI Score': d['psi_score'],
                'KS Statistic': d['ks_statistic'],
                'KS P-Value': d['ks_p_value'],
                'Mean Shift': d['mean_shift'],
                'Variance Ratio': d['variance_ratio'],
                'Drift Detected': '⚠️ Yes' if d['drift_detected'] else '✅ No'
            }
            for f, d in drift_data['feature_drift'].items()
        ])
        st.dataframe(drift_df, use_container_width=True, hide_index=True)


def render_anomaly_detection(model_id: str, anomaly_data: Dict):
    """Render anomaly detection section"""
    st.markdown("### 🔎 Anomaly Detection")
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    anomaly_rate = anomaly_data['anomaly_rate']
    rate_color = '#EF4444' if anomaly_rate > 0.01 else ('#F59E0B' if anomaly_rate > 0.005 else '#10B981')
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{anomaly_data['total_predictions']:,}</div>
            <div class="stat-card-label">Predictions Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="border-left: 4px solid {rate_color};">
            <div class="stat-card-value" style="color: {rate_color};">{anomaly_data['anomalies_detected']}</div>
            <div class="stat-card-label">Anomalies Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{anomaly_rate:.2%}</div>
            <div class="stat-card-label">Anomaly Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        trend = anomaly_data['anomaly_trend']
        trend_icon = '📈' if trend == 'increasing' else ('📉' if trend == 'decreasing' else '➡️')
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{trend_icon}</div>
            <div class="stat-card-label">Trend: {trend.capitalize()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top Anomalies Table
    st.markdown("#### 🎯 Recent Anomalies")
    
    anomaly_df = pd.DataFrame(anomaly_data['top_anomalies'])
    anomaly_df['anomaly_score'] = anomaly_df['anomaly_score'].apply(lambda x: f"{x:.2%}")
    anomaly_df = anomaly_df.rename(columns={
        'timestamp': 'Timestamp',
        'prediction_id': 'Prediction ID',
        'anomaly_score': 'Anomaly Score',
        'reason': 'Reason'
    })
    
    st.dataframe(anomaly_df, use_container_width=True, hide_index=True)


def render_data_quality(model_id: str, profile_data: Dict):
    """Render data quality profiling section"""
    st.markdown("### 📋 Data Quality Profile")
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    quality_score = profile_data['overall_quality_score']
    quality_color = '#10B981' if quality_score > 0.9 else ('#F59E0B' if quality_score > 0.8 else '#EF4444')
    
    with col1:
        st.markdown(f"""
        <div class="stat-card" style="border-left: 4px solid {quality_color};">
            <div class="stat-card-value" style="color: {quality_color};">{quality_score:.1%}</div>
            <div class="stat-card-label">Quality Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{profile_data['total_records']:,}</div>
            <div class="stat-card-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{len(profile_data['features'])}</div>
            <div class="stat-card-label">Features Profiled</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{profile_data['time_range']}</div>
            <div class="stat-card-label">Time Range</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Quality Table
    st.markdown("#### Feature Statistics")
    
    quality_df = pd.DataFrame([
        {
            'Feature': f,
            'Missing Rate': f"{d['missing_rate']:.2%}",
            'Mean': f"{d['mean']:.2f}",
            'Std Dev': f"{d['std']:.2f}",
            'Min': f"{d['min']:.2f}",
            'Max': f"{d['max']:.2f}",
            'Outlier Rate': f"{d['outlier_rate']:.2%}"
        }
        for f, d in profile_data['features'].items()
    ])
    
    st.dataframe(quality_df, use_container_width=True, hide_index=True)


def render_alerts_panel(alerts: List[Dict]):
    """Render alerts panel with cards"""
    st.markdown("### 🚨 Active Alerts")
    
    if not alerts:
        st.markdown("""
        <div style="background: #D1FAE5; border-radius: 12px; padding: 2rem; text-align: center;">
            <span style="font-size: 3rem;">🎉</span>
            <h3 style="color: #065F46; margin: 1rem 0;">All Clear!</h3>
            <p style="color: #047857;">No active alerts. Your models are running smoothly.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Alert Summary Cards
    critical_count = sum(1 for a in alerts if a['severity'] == 'critical')
    high_count = sum(1 for a in alerts if a['severity'] == 'high')
    medium_count = sum(1 for a in alerts if a['severity'] == 'medium')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", len(alerts))
    with col2:
        st.metric("🚨 Critical", critical_count)
    with col3:
        st.metric("⚠️ High", high_count)
    with col4:
        st.metric("📝 Medium", medium_count)
    
    st.markdown("")
    
    # Alert Cards
    for alert in sorted(alerts, key=lambda x: {'critical': 0, 'high': 1, 'medium': 2}.get(x['severity'], 3)):
        severity = alert['severity']
        
        if severity == 'critical':
            alert_class = 'alert-critical'
            icon = '🚨'
        elif severity == 'high':
            alert_class = 'alert-warning'
            icon = '⚠️'
        else:
            alert_class = 'alert-info'
            icon = 'ℹ️'
        
        recommendations_html = ""
        if alert.get('recommendations'):
            recommendations_html = "<br><strong>Recommendations:</strong><ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>"
            for rec in alert['recommendations']:
                recommendations_html += f"<li style='margin: 0.25rem 0;'>{rec}</li>"
            recommendations_html += "</ul>"
        
        st.markdown(f"""
        <div class="{alert_class}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <strong style="font-size: 1.1rem;">{icon} {alert['title']}</strong>
                    <p style="margin: 0.5rem 0; color: #374151;">{alert['message']}</p>
                    {recommendations_html}
                </div>
                <div style="text-align: right; min-width: 120px;">
                    <span style="background: rgba(0,0,0,0.1); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem;">
                        {alert['category'].upper()}
                    </span>
                    <p style="font-size: 0.8rem; color: #6B7280; margin-top: 0.5rem;">
                        {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_explainability(model_id: str, importance_data: Dict):
    """Render model explainability section with cards"""
    st.markdown("### 🧠 Model Explainability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance bar chart
        features = list(importance_data.keys())
        importances = [importance_data[f]['importance'] for f in features]
        
        sorted_idx = np.argsort(importances)[::-1]
        features = [features[i] for i in sorted_idx]
        importances = [importances[i] for i in sorted_idx]
        directions = [importance_data[features[i]]['direction'] for i in range(len(features))]
        
        colors = ['#10B981' if d == 'positive' else '#EF4444' for d in directions]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker_color=colors,
            text=[f"{imp:.1%}" for imp in importances],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Global Feature Importance',
            height=400,
            template='plotly_white',
            xaxis_title='Importance'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Top Feature Insights")
        
        for i, feature in enumerate(features[:5]):
            data = importance_data[feature]
            direction = data['direction']
            stability = data['stability']
            
            direction_icon = '📈' if direction == 'positive' else '📉'
            stability_color = '#10B981' if stability == 'stable' else '#F59E0B'
            
            st.markdown(f"""
            <div class="feature-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="font-size: 1rem;">#{i+1} {feature}</strong>
                        <p style="margin: 0.25rem 0; color: #6B7280;">
                            {direction_icon} {direction.capitalize()} impact
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 1.5rem; font-weight: 700; color: #4F46E5;">
                            {data['importance']:.1%}
                        </span>
                        <p style="margin: 0; color: {stability_color}; font-size: 0.8rem;">
                            {stability.capitalize()}
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_about_section():
    """Render comprehensive About/How-To section"""
    st.markdown("### 📖 About ML Model Monitor Pro")


def render_data_upload_section():
    """Render the data upload section"""
    st.markdown("### 📤 Upload Your Data")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%); 
                border-radius: 16px; padding: 1.5rem; margin-bottom: 2rem;">
        <h4 style="margin: 0 0 0.5rem 0; color: #4338CA;">📁 Supported File Formats</h4>
        <p style="margin: 0; color: #4B5563;">
            <strong>CSV</strong> (.csv) • <strong>Excel</strong> (.xlsx, .xls) • 
            <strong>JSON</strong> (.json) • <strong>Parquet</strong> (.parquet)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data type tabs
    upload_tab1, upload_tab2, upload_tab3, upload_tab4, upload_tab5 = st.tabs([
        "📊 Models", "📈 Metrics", "🎯 Predictions", "📋 Reference Data", "🎲 Generate Sample"
    ])
    
    with upload_tab1:
        render_model_upload()
    
    with upload_tab2:
        render_metrics_upload()
    
    with upload_tab3:
        render_predictions_upload()
    
    with upload_tab4:
        render_reference_upload()
    
    with upload_tab5:
        render_sample_generator()
    
    # Show current data status
    st.markdown("---")
    render_data_status()


def render_model_upload():
    """Render model registry upload"""
    st.markdown("#### Upload Model Registry")
    
    st.markdown("""
    <div class="feature-card">
        <strong>Required Columns:</strong> model_id, name, type<br>
        <strong>Optional:</strong> framework, version, status, deployed_at, accuracy, mae, predictions_today
    </div>
    """, unsafe_allow_html=True)
    
    # Show example
    with st.expander("📋 View Example Format"):
        example_df = pd.DataFrame({
            'model_id': ['model_001', 'model_002', 'model_003'],
            'name': ['Churn Predictor', 'Revenue Forecast', 'Fraud Detection'],
            'type': ['Classification', 'Regression', 'Classification'],
            'framework': ['XGBoost', 'LightGBM', 'PyTorch'],
            'version': ['v1.0', 'v2.1', 'v3.0'],
            'status': ['healthy', 'warning', 'healthy'],
            'accuracy': [0.94, None, 0.987],
            'mae': [None, 0.023, None]
        })
        st.dataframe(example_df, use_container_width=True)
        
        csv = example_df.to_csv(index=False)
        st.download_button("📥 Download Template", csv, "models_template.csv", "text/csv")
    
    uploaded_file = st.file_uploader("Upload Models File", type=['csv', 'xlsx', 'json'], key="models_upload")
    
    if uploaded_file:
        df = read_uploaded_file(uploaded_file)
        if df is not None:
            st.markdown("**Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("✅ Upload Models", type="primary", key="btn_models"):
                if 'uploaded_data' not in st.session_state:
                    st.session_state.uploaded_data = {}
                st.session_state.uploaded_data['models'] = df
                st.success(f"✅ Uploaded {len(df)} models!")
                st.balloons()


def render_metrics_upload():
    """Render metrics upload"""
    st.markdown("#### Upload Performance Metrics")
    
    st.markdown("""
    <div class="feature-card">
        <strong>Required Columns:</strong> model_id, timestamp, metric_name, metric_value<br>
        <strong>Metric Names:</strong> mae, rmse, accuracy, latency_ms, predictions_count, error_rate
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📋 View Example Format"):
        timestamps = pd.date_range(end=datetime.now(), periods=5, freq='10min')
        example_df = pd.DataFrame({
            'model_id': ['model_001'] * 5,
            'timestamp': timestamps,
            'metric_name': ['mae', 'mae', 'mae', 'mae', 'mae'],
            'metric_value': [0.052, 0.048, 0.055, 0.051, 0.049]
        })
        st.dataframe(example_df, use_container_width=True)
        
        # Alternative wide format
        st.markdown("**Alternative Wide Format:**")
        wide_df = pd.DataFrame({
            'model_id': ['model_001'] * 5,
            'timestamp': timestamps,
            'mae': [0.052, 0.048, 0.055, 0.051, 0.049],
            'rmse': [0.065, 0.062, 0.068, 0.064, 0.061],
            'latency_ms': [23.5, 22.1, 25.8, 24.2, 21.9],
            'accuracy': [0.94, 0.945, 0.938, 0.942, 0.947]
        })
        st.dataframe(wide_df, use_container_width=True)
        
        csv = wide_df.to_csv(index=False)
        st.download_button("📥 Download Template", csv, "metrics_template.csv", "text/csv")
    
    uploaded_file = st.file_uploader("Upload Metrics File", type=['csv', 'xlsx', 'json', 'parquet'], key="metrics_upload")
    
    if uploaded_file:
        df = read_uploaded_file(uploaded_file)
        if df is not None:
            st.markdown("**Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Total: {len(df)} rows")
            
            if st.button("✅ Upload Metrics", type="primary", key="btn_metrics"):
                if 'uploaded_data' not in st.session_state:
                    st.session_state.uploaded_data = {}
                st.session_state.uploaded_data['metrics'] = df
                st.success(f"✅ Uploaded {len(df)} metric records!")
                st.balloons()


def render_predictions_upload():
    """Render predictions upload"""
    st.markdown("#### Upload Prediction Logs")
    
    st.markdown("""
    <div class="feature-card">
        <strong>Required Columns:</strong> model_id, timestamp, prediction<br>
        <strong>Optional:</strong> actual (ground truth), latency_ms, confidence, feature columns (feature_1, feature_2, etc.)
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📋 View Example Format"):
        timestamps = pd.date_range(end=datetime.now(), periods=5, freq='1min')
        example_df = pd.DataFrame({
            'model_id': ['model_001'] * 5,
            'timestamp': timestamps,
            'prediction': [0.85, 0.23, 0.67, 0.91, 0.45],
            'actual': [1, 0, 1, 1, 0],
            'latency_ms': [23.5, 21.2, 28.1, 22.8, 24.5],
            'age': [35, 42, 28, 55, 31],
            'income': [75000, 52000, 89000, 120000, 45000],
            'tenure': [24, 12, 36, 48, 6]
        })
        st.dataframe(example_df, use_container_width=True)
        
        csv = example_df.to_csv(index=False)
        st.download_button("📥 Download Template", csv, "predictions_template.csv", "text/csv")
    
    uploaded_file = st.file_uploader("Upload Predictions File", type=['csv', 'xlsx', 'json', 'parquet'], key="predictions_upload")
    
    if uploaded_file:
        df = read_uploaded_file(uploaded_file)
        if df is not None:
            st.markdown("**Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Total: {len(df)} predictions")
            
            # Auto-detect feature columns
            feature_cols = [c for c in df.columns if c not in ['model_id', 'timestamp', 'prediction', 'actual', 'latency_ms', 'confidence']]
            if feature_cols:
                st.info(f"🔍 Detected {len(feature_cols)} feature columns: {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}")
            
            if st.button("✅ Upload Predictions", type="primary", key="btn_predictions"):
                if 'uploaded_data' not in st.session_state:
                    st.session_state.uploaded_data = {}
                st.session_state.uploaded_data['predictions'] = df
                st.success(f"✅ Uploaded {len(df)} predictions!")
                st.balloons()


def render_reference_upload():
    """Render reference data upload for drift detection"""
    st.markdown("#### Upload Reference Data (Training Distribution)")
    
    st.markdown("""
    <div class="feature-card">
        <strong>Purpose:</strong> Reference data is used to detect drift by comparing production data against training data distribution.<br>
        <strong>Format:</strong> Upload your training data or feature statistics.
    </div>
    """, unsafe_allow_html=True)
    
    # Model selector
    model_id = st.text_input("Model ID", value="model_001", key="ref_model_id")
    
    upload_type = st.radio(
        "Upload Type",
        ["Raw Training Data", "Feature Statistics"],
        horizontal=True
    )
    
    if upload_type == "Raw Training Data":
        with st.expander("📋 View Example Format"):
            example_df = pd.DataFrame({
                'age': np.random.normal(35, 10, 100).astype(int),
                'income': np.random.normal(65000, 25000, 100).astype(int),
                'tenure': np.random.normal(24, 12, 100).astype(int),
                'usage': np.random.normal(50, 20, 100),
                'target': np.random.binomial(1, 0.3, 100)
            })
            st.dataframe(example_df.head(), use_container_width=True)
            st.caption("Upload your actual training data - statistics will be computed automatically")
    else:
        with st.expander("📋 View Example Format"):
            example_df = pd.DataFrame({
                'feature_name': ['age', 'income', 'tenure', 'usage'],
                'mean_value': [35.5, 65000, 24, 50],
                'std_value': [10.2, 25000, 12, 20],
                'min_value': [18, 20000, 1, 0],
                'max_value': [75, 200000, 60, 100]
            })
            st.dataframe(example_df, use_container_width=True)
            
            csv = example_df.to_csv(index=False)
            st.download_button("📥 Download Template", csv, "reference_template.csv", "text/csv")
    
    uploaded_file = st.file_uploader("Upload Reference Data", type=['csv', 'xlsx', 'json', 'parquet'], key="reference_upload")
    
    if uploaded_file:
        df = read_uploaded_file(uploaded_file)
        if df is not None:
            st.markdown("**Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            if upload_type == "Raw Training Data":
                # Show computed statistics
                st.markdown("**Computed Statistics:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                stats_df = df[numeric_cols].describe().T
                st.dataframe(stats_df, use_container_width=True)
            
            if st.button("✅ Upload Reference Data", type="primary", key="btn_reference"):
                if 'uploaded_data' not in st.session_state:
                    st.session_state.uploaded_data = {}
                if 'reference' not in st.session_state.uploaded_data:
                    st.session_state.uploaded_data['reference'] = {}
                st.session_state.uploaded_data['reference'][model_id] = df
                st.success(f"✅ Uploaded reference data for {model_id}!")
                st.balloons()


def render_sample_generator():
    """Render sample data generator"""
    st.markdown("#### 🎲 Generate Sample Data")
    
    st.markdown("""
    <div class="feature-card">
        <strong>Purpose:</strong> Generate realistic sample data to test the platform without uploading real data.<br>
        Download the generated files and upload them to explore all features.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_models = st.slider("Number of Models", 1, 10, 4)
        n_days = st.slider("Days of History", 1, 30, 7)
    
    with col2:
        n_predictions = st.slider("Predictions per Model", 100, 5000, 1000)
        add_drift = st.checkbox("Include Data Drift", value=True)
        add_anomalies = st.checkbox("Include Anomalies", value=True)
    
    if st.button("🎲 Generate Sample Data", type="primary"):
        with st.spinner("Generating data..."):
            # Generate Models
            frameworks = ['XGBoost', 'LightGBM', 'PyTorch', 'TensorFlow', 'Scikit-learn']
            statuses = ['healthy', 'healthy', 'healthy', 'warning', 'critical']
            
            models_data = []
            for i in range(n_models):
                models_data.append({
                    'model_id': f'model_{i+1:03d}',
                    'name': f'Sample Model {i+1}',
                    'type': np.random.choice(['Classification', 'Regression']),
                    'framework': np.random.choice(frameworks),
                    'version': f'v{np.random.randint(1,4)}.{np.random.randint(0,10)}.{np.random.randint(0,10)}',
                    'status': np.random.choice(statuses),
                    'deployed_at': (datetime.now() - timedelta(days=np.random.randint(10, 100))).isoformat(),
                    'predictions_today': np.random.randint(1000, 50000),
                    'avg_latency_ms': round(np.random.uniform(10, 80), 1),
                    'accuracy': round(np.random.uniform(0.85, 0.98), 3) if np.random.random() > 0.5 else None,
                    'mae': round(np.random.uniform(0.02, 0.1), 4) if np.random.random() > 0.5 else None
                })
            
            models_df = pd.DataFrame(models_data)
            
            # Generate Metrics (wide format)
            metrics_list = []
            for model in models_data:
                model_id = model['model_id']
                timestamps = pd.date_range(end=datetime.now(), periods=n_days * 24 * 6, freq='10min')
                
                # Add drift trend if enabled
                drift_factor = np.linspace(0, 0.03 if add_drift else 0, len(timestamps))
                
                for j, ts in enumerate(timestamps):
                    metrics_list.append({
                        'model_id': model_id,
                        'timestamp': ts,
                        'mae': round(0.05 + np.random.normal(0, 0.005) + drift_factor[j], 5),
                        'rmse': round(0.065 + np.random.normal(0, 0.008) + drift_factor[j], 5),
                        'accuracy': round(np.clip(0.92 + np.random.normal(0, 0.01) - drift_factor[j], 0.7, 1.0), 4),
                        'latency_ms': round(20 + np.random.exponential(15), 2),
                        'predictions_count': np.random.poisson(100),
                        'error_rate': round(np.clip(0.002 + np.random.exponential(0.002), 0, 0.1), 5)
                    })
            
            metrics_df = pd.DataFrame(metrics_list)
            
            # Generate Predictions with features
            predictions_list = []
            feature_names = ['age', 'income', 'tenure', 'usage_freq', 'support_tickets']
            
            for model in models_data:
                model_id = model['model_id']
                
                for i in range(n_predictions):
                    pred = {
                        'model_id': model_id,
                        'timestamp': datetime.now() - timedelta(minutes=i),
                        'prediction': round(np.random.random(), 4),
                        'actual': int(np.random.binomial(1, 0.5)) if np.random.random() > 0.3 else None,
                        'latency_ms': round(np.random.exponential(25), 2)
                    }
                    
                    # Add features
                    for feat in feature_names:
                        if feat == 'age':
                            pred[feat] = int(np.random.normal(40, 12))
                        elif feat == 'income':
                            pred[feat] = int(np.random.normal(70000, 30000))
                        elif feat == 'tenure':
                            pred[feat] = int(np.random.exponential(24))
                        else:
                            pred[feat] = round(np.random.normal(50, 20), 2)
                    
                    # Add anomalies
                    if add_anomalies and np.random.random() < 0.02:
                        pred['is_anomaly'] = True
                        pred['anomaly_score'] = round(np.random.uniform(0.8, 0.99), 3)
                    
                    predictions_list.append(pred)
            
            predictions_df = pd.DataFrame(predictions_list)
            
            # Generate Reference Data
            reference_list = []
            for model in models_data:
                for feat in feature_names:
                    if feat == 'age':
                        reference_list.append({
                            'model_id': model['model_id'],
                            'feature_name': feat,
                            'mean_value': 40,
                            'std_value': 12,
                            'min_value': 18,
                            'max_value': 80
                        })
                    elif feat == 'income':
                        reference_list.append({
                            'model_id': model['model_id'],
                            'feature_name': feat,
                            'mean_value': 70000,
                            'std_value': 30000,
                            'min_value': 15000,
                            'max_value': 250000
                        })
                    else:
                        reference_list.append({
                            'model_id': model['model_id'],
                            'feature_name': feat,
                            'mean_value': 50,
                            'std_value': 20,
                            'min_value': 0,
                            'max_value': 100
                        })
            
            reference_df = pd.DataFrame(reference_list)
        
        st.success("✅ Sample data generated!")
        
        # Download buttons
        st.markdown("#### 📥 Download Generated Files")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                "📊 Models CSV",
                models_df.to_csv(index=False),
                "sample_models.csv",
                "text/csv",
                use_container_width=True
            )
            st.caption(f"{len(models_df)} models")
        
        with col2:
            st.download_button(
                "📈 Metrics CSV",
                metrics_df.to_csv(index=False),
                "sample_metrics.csv",
                "text/csv",
                use_container_width=True
            )
            st.caption(f"{len(metrics_df):,} records")
        
        with col3:
            st.download_button(
                "🎯 Predictions CSV",
                predictions_df.to_csv(index=False),
                "sample_predictions.csv",
                "text/csv",
                use_container_width=True
            )
            st.caption(f"{len(predictions_df):,} predictions")
        
        with col4:
            st.download_button(
                "📋 Reference CSV",
                reference_df.to_csv(index=False),
                "sample_reference.csv",
                "text/csv",
                use_container_width=True
            )
            st.caption(f"{len(reference_df)} records")


def render_data_status():
    """Render current data status"""
    st.markdown("### 📊 Current Data Status")
    
    uploaded_data = st.session_state.get('uploaded_data', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        models_count = len(uploaded_data.get('models', [])) if 'models' in uploaded_data else 0
        source = "✅ uploaded" if models_count > 0 else "📌 demo"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{models_count if models_count > 0 else 4}</div>
            <div class="stat-card-label">📊 Models ({source})</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        metrics_count = len(uploaded_data.get('metrics', [])) if 'metrics' in uploaded_data else 0
        source = "✅ uploaded" if metrics_count > 0 else "📌 demo"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{metrics_count:,}</div>
            <div class="stat-card-label">📈 Metrics ({source})</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pred_count = len(uploaded_data.get('predictions', [])) if 'predictions' in uploaded_data else 0
        source = "✅ uploaded" if pred_count > 0 else "📌 demo"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{pred_count:,}</div>
            <div class="stat-card-label">🎯 Predictions ({source})</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ref_count = len(uploaded_data.get('reference', {})) if 'reference' in uploaded_data else 0
        source = "✅ uploaded" if ref_count > 0 else "📌 demo"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{ref_count}</div>
            <div class="stat-card-label">📋 Reference Sets ({source})</div>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_data:
        if st.button("🗑️ Clear All Uploaded Data"):
            st.session_state.uploaded_data = {}
            st.success("All uploaded data cleared!")
            st.rerun()


def read_uploaded_file(uploaded_file):
    """Read uploaded file and return DataFrame"""
    try:
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        elif file_name.endswith('.json'):
            return pd.read_json(uploaded_file)
        elif file_name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_name}")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None


def render_about_section():
    """Render comprehensive About/How-To section"""
    st.markdown("### 📖 About ML Model Monitor Pro")
    
    # Purpose Section
    st.markdown("""
    <div class="info-card" style="margin-bottom: 2rem;">
        <h4 style="color: #4F46E5; margin-top: 0;">🎯 Purpose</h4>
        <p style="font-size: 1.05rem; color: #374151; line-height: 1.8;">
            ML Model Monitor Pro is an <strong>enterprise-grade machine learning monitoring platform</strong> 
            designed to help data science teams maintain healthy, reliable, and explainable ML models in production. 
            It provides real-time insights into model performance, data quality, and prediction reliability.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Benefits
    st.markdown("#### 💎 Key Benefits")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="benefit-card">
            <div class="benefit-icon">🚨</div>
            <div class="benefit-title">Early Detection</div>
            <div class="benefit-description">
                Catch model degradation and data drift before they impact business outcomes. 
                Reduce incident response time by up to 80%.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="benefit-card">
            <div class="benefit-icon">🧠</div>
            <div class="benefit-title">Explainability</div>
            <div class="benefit-description">
                Understand why your models make specific predictions. Build trust with stakeholders 
                through transparent AI decision-making.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="benefit-card">
            <div class="benefit-icon">⚡</div>
            <div class="benefit-title">Faster Response</div>
            <div class="benefit-description">
                Intelligent alerting with actionable recommendations. Know exactly what to fix 
                and how to fix it when issues arise.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="benefit-card">
            <div class="benefit-icon">📊</div>
            <div class="benefit-title">Comprehensive Insights</div>
            <div class="benefit-description">
                360-degree view of your ML operations including performance, latency, 
                data quality, and resource utilization.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # How to Use Section
    st.markdown("#### 🚀 How to Use This Platform")
    
    steps = [
        {
            'title': 'Monitor Model Health',
            'description': 'Start at the Overview tab to see the health status of all your deployed models. Green indicates healthy, yellow means warning, and red signals critical issues requiring attention.',
            'tip': 'Focus on critical models first - they need immediate attention!'
        },
        {
            'title': 'Analyze Performance Trends',
            'description': 'Use the Performance tab to track metrics like MAE, accuracy, latency, and throughput over time. Look for upward trends in error metrics or latency that could indicate problems.',
            'tip': 'Set up baseline metrics when you first deploy a model to make comparisons easier.'
        },
        {
            'title': 'Detect Data Drift',
            'description': 'The Drift Detection tab uses statistical tests (PSI, KS-test) to identify when your production data has changed from your training data. Drift can cause model performance to degrade.',
            'tip': 'PSI > 0.1 indicates moderate drift, PSI > 0.2 is significant drift requiring action.'
        },
        {
            'title': 'Investigate Anomalies',
            'description': 'The Anomaly Detection feature identifies unusual predictions that may indicate model issues or interesting edge cases. Review flagged predictions to understand model behavior.',
            'tip': 'High anomaly rates often correlate with data quality issues or concept drift.'
        },
        {
            'title': 'Understand Predictions',
            'description': 'Use the Explainability tab to understand which features drive your model\'s predictions. This is crucial for debugging issues and building stakeholder trust.',
            'tip': 'Monitor feature importance over time - sudden changes may indicate data issues.'
        },
        {
            'title': 'Respond to Alerts',
            'description': 'When alerts are triggered, follow the recommendations provided. Critical alerts need immediate attention, while warnings should be investigated within 24-48 hours.',
            'tip': 'Set up notification channels (Slack, email) to never miss critical alerts.'
        }
    ]
    
    for i, step in enumerate(steps, 1):
        st.markdown(f"""
        <div style="display: flex; margin: 1.5rem 0; align-items: flex-start;">
            <div style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); 
                        color: white; width: 3rem; height: 3rem; border-radius: 50%; 
                        display: flex; align-items: center; justify-content: center; 
                        font-weight: 700; font-size: 1.25rem; flex-shrink: 0;
                        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);">
                {i}
            </div>
            <div style="margin-left: 1.5rem; flex-grow: 1;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1F2937;">{step['title']}</h4>
                <p style="color: #4B5563; margin: 0 0 0.5rem 0;">{step['description']}</p>
                <p style="background: #EEF2FF; padding: 0.75rem 1rem; border-radius: 8px; 
                          color: #4338CA; margin: 0; font-size: 0.9rem;">
                    💡 <strong>Pro Tip:</strong> {step['tip']}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Features Overview
    st.markdown("#### ✨ Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        features_left = [
            ("📈 Performance Monitoring", "Track MAE, RMSE, accuracy, F1, and custom metrics in real-time"),
            ("🔄 Data Drift Detection", "PSI, KS-test, Wasserstein distance, JS divergence analysis"),
            ("🔎 Anomaly Detection", "Identify unusual predictions and out-of-distribution inputs"),
            ("📋 Data Profiling", "Monitor data quality, missing values, and feature distributions"),
            ("⚡ Latency Tracking", "P50, P95, P99 latency with SLA violation alerts"),
        ]
        
        for title, desc in features_left:
            st.markdown(f"""
            <div class="feature-card">
                <strong>{title}</strong>
                <p style="margin: 0.25rem 0 0 0; color: #6B7280; font-size: 0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        features_right = [
            ("🧠 Model Explainability", "SHAP values, permutation importance, feature contributions"),
            ("🚨 Intelligent Alerting", "Multi-channel notifications with actionable recommendations"),
            ("📊 Resource Monitoring", "CPU, memory usage, and infrastructure health"),
            ("🔗 Model Comparison", "Compare multiple models side-by-side"),
            ("📤 Export & Integration", "API access, webhook support, report generation"),
        ]
        
        for title, desc in features_right:
            st.markdown(f"""
            <div class="feature-card">
                <strong>{title}</strong>
                <p style="margin: 0.25rem 0 0 0; color: #6B7280; font-size: 0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Quick Reference
    with st.expander("📚 Quick Reference - Metric Thresholds"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **PSI (Population Stability Index)**
            - < 0.1: No significant change ✅
            - 0.1 - 0.2: Moderate shift ⚠️
            - > 0.2: Significant shift 🚨
            """)
        
        with col2:
            st.markdown("""
            **Latency SLA**
            - < 50ms: Good ✅
            - 50-100ms: Acceptable ⚠️
            - > 100ms: Degraded 🚨
            """)
        
        with col3:
            st.markdown("""
            **Anomaly Rate**
            - < 0.5%: Normal ✅
            - 0.5-1%: Elevated ⚠️
            - > 1%: High 🚨
            """)


def render_sidebar(models: Dict) -> str:
    """Render sidebar and return selected model"""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 2rem;">🔍</span>
        <h2 style="margin: 0.5rem 0;">ML Monitor Pro</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Model selector
    st.sidebar.markdown("#### 🎯 Select Model")
    model_options = {m_id: f"{m_info['name']}" for m_id, m_info in models.items()}
    selected_model = st.sidebar.selectbox(
        "Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        label_visibility="collapsed"
    )
    
    # Show selected model status
    model_info = models[selected_model]
    status_colors = {'healthy': '#10B981', 'warning': '#F59E0B', 'critical': '#EF4444'}
    st.sidebar.markdown(f"""
    <div style="background: {status_colors[model_info['status']]}20; 
                padding: 0.75rem; border-radius: 8px; margin-top: 0.5rem;
                border-left: 4px solid {status_colors[model_info['status']]};">
        <strong>Status:</strong> {model_info['status'].upper()}<br>
        <small>Version: {model_info['version']}</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Time range selector
    st.sidebar.markdown("#### ⏱️ Time Range")
    time_range = st.sidebar.select_slider(
        "Range",
        options=['6h', '12h', '24h', '7d', '30d'],
        value='24h',
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Quick Actions
    st.sidebar.markdown("#### ⚡ Quick Actions")
    
    if st.sidebar.button("📥 Export Report", use_container_width=True):
        st.sidebar.success("Report exported!")
    
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    if st.sidebar.button("⚙️ Settings", use_container_width=True):
        st.sidebar.info("Settings panel coming soon!")
    
    st.sidebar.markdown("---")
    
    # System Status
    st.sidebar.markdown("#### 💻 System Status")
    st.sidebar.markdown("""
    <div style="font-size: 0.9rem;">
        🟢 API Gateway: Online<br>
        🟢 Database: Connected<br>
        🟢 Alert Service: Active<br>
        🟢 Metrics Collector: Running
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Footer
    st.sidebar.markdown("""
    <div style="text-align: center; color: #9CA3AF; font-size: 0.8rem;">
        ML Monitor Pro v1.0.0<br>
        © 2025 Your Company
    </div>
    """, unsafe_allow_html=True)
    
    return selected_model


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    
    # Generate demo data
    models = generate_demo_models()
    
    # Render sidebar and get selected model
    selected_model = render_sidebar(models)
    
    # Render header
    render_header()
    
    # Overview metrics
    render_overview_cards(models)
    
    st.markdown("---")
    
    # Model cards
    render_model_cards(models)
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Performance", 
        "🔄 Drift Detection",
        "🔎 Anomalies",
        "📋 Data Quality",
        "🧠 Explainability",
        "🚨 Alerts",
        "📤 Upload Data",
        "📖 About & Help"
    ])
    
    with tab1:
        metrics_df = generate_demo_metrics(selected_model)
        render_performance_charts(selected_model, metrics_df)
    
    with tab2:
        drift_data = generate_demo_drift_data(selected_model)
        render_drift_analysis(selected_model, drift_data)
    
    with tab3:
        anomaly_data = generate_anomaly_data(selected_model)
        render_anomaly_detection(selected_model, anomaly_data)
    
    with tab4:
        profile_data = generate_data_profile(selected_model)
        render_data_quality(selected_model, profile_data)
    
    with tab5:
        importance_data = generate_demo_explanations()
        render_explainability(selected_model, importance_data)
    
    with tab6:
        alerts = generate_demo_alerts(selected_model)
        render_alerts_panel(alerts)
    
    with tab7:
        render_data_upload_section()
    
    with tab8:
        render_about_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9CA3AF; font-size: 0.9rem; padding: 2rem 0;">
        <strong>ML Model Monitor Pro</strong> v1.0.0 | 
        Built with ❤️ using Streamlit | 
        <a href="#" style="color: #4F46E5;">Documentation</a> | 
        <a href="#" style="color: #4F46E5;">Support</a> | 
        <a href="#" style="color: #4F46E5;">API Reference</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
