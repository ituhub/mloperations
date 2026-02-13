# ML Model Monitor Pro - Integration Examples
# =============================================================================
# Examples showing how to integrate monitoring into your ML pipeline
# =============================================================================

"""
This file contains examples of how to:
1. Log predictions from your ML model
2. Track performance metrics
3. Set up reference data for drift detection
4. Configure alerting

Choose the integration pattern that matches your infrastructure.
"""

# =============================================================================
# EXAMPLE 1: Basic Integration with Any ML Model
# =============================================================================

def example_basic_integration():
    """
    Basic example: Monitor any scikit-learn compatible model
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import time
    
    # Import monitoring components
    import sys
    sys.path.append('..')
    from backend.data_integration import DataProviderFactory, FileConfig
    from backend.core.drift_detector import DriftDetector
    from backend.core.performance_tracker import PerformanceTracker, MetricType
    
    # --- Setup ---
    # Use file-based storage for this example
    provider = DataProviderFactory.create('file', FileConfig(data_directory='./monitoring_data'))
    drift_detector = DriftDetector()
    performance_tracker = PerformanceTracker()
    
    # --- Train a model ---
    X, y = make_classification(n_samples=10000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # --- Set reference data for drift detection ---
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    drift_detector.set_reference(
        model_id='my_model',
        X_reference=X_train,
        feature_names=feature_names
    )
    
    # --- Simulate production predictions ---
    print("Simulating production predictions...")
    
    for i in range(100):
        # Get a batch of data (simulating production traffic)
        batch_idx = np.random.choice(len(X_test), size=10)
        X_batch = X_test[batch_idx]
        y_true = y_test[batch_idx]
        
        # Make predictions and measure latency
        start_time = time.time()
        y_pred = model.predict(X_batch)
        y_proba = model.predict_proba(X_batch)[:, 1]
        latency_ms = (time.time() - start_time) * 1000
        
        # --- Log predictions ---
        for j in range(len(X_batch)):
            provider.log_prediction(
                model_id='my_model',
                features={f'feature_{k}': float(X_batch[j, k]) for k in range(X_batch.shape[1])},
                prediction=float(y_proba[j]),
                actual=float(y_true[j])
            )
        
        # --- Log performance metrics ---
        accuracy = np.mean(y_pred == y_true)
        performance_tracker.log_prediction(
            model_id='my_model',
            y_true=y_true,
            y_pred=y_pred,
            latency_ms=latency_ms,
            is_regression=False
        )
        
        # --- Check for drift (every 10 batches) ---
        if i % 10 == 0:
            drift_report = drift_detector.detect_drift(
                model_id='my_model',
                X_current=X_batch
            )
            
            if drift_report.drift_detected:
                print(f"⚠️ Drift detected at batch {i}! Score: {drift_report.overall_drift_score:.4f}")
    
    print("✅ Monitoring integration complete!")
    
    # --- Generate reports ---
    perf_report = performance_tracker.generate_report('my_model', report_period_hours=1)
    print(f"\nPerformance Summary:")
    print(f"  Predictions: {perf_report.predictions_count}")
    print(f"  Avg Latency: {perf_report.avg_latency_ms:.2f}ms")
    print(f"  Alerts: {len(perf_report.alerts)}")


# =============================================================================
# EXAMPLE 2: Integration with FastAPI Serving
# =============================================================================

def example_fastapi_integration():
    """
    Example: Integrate monitoring with FastAPI model serving
    
    Save this as a separate file (e.g., api_server.py) and run with:
    uvicorn api_server:app --reload
    """
    
    code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import time
import joblib

# Import monitoring
from backend.data_integration import DataProviderFactory, DatabaseConfig
from backend.core.drift_detector import DriftDetector
from backend.core.performance_tracker import PerformanceTracker, MetricType
from backend.core.alert_manager import AlertManager, AlertSeverity, AlertCategory

app = FastAPI(title="ML Model API with Monitoring")

# --- Initialize monitoring ---
# Use database for production
provider = DataProviderFactory.create('database', DatabaseConfig(
    host='localhost',
    database='ml_monitoring',
    username='admin',
    password='secret'
))

drift_detector = DriftDetector()
performance_tracker = PerformanceTracker()
alert_manager = AlertManager()

# Load your model
model = joblib.load('model.pkl')
MODEL_ID = 'production_model_v1'

# Request/Response models
class PredictionRequest(BaseModel):
    features: Dict[str, float]
    
class PredictionResponse(BaseModel):
    prediction: float
    confidence: Optional[float]
    prediction_id: str
    latency_ms: float

class BatchPredictionRequest(BaseModel):
    instances: List[Dict[str, float]]


@app.on_event("startup")
async def startup():
    """Initialize reference data on startup"""
    # Load reference data from database or file
    ref_data = provider.get_reference_data(MODEL_ID)
    if not ref_data.empty:
        # Convert to numpy array for drift detection
        X_ref = ref_data[['mean_value']].values  # Simplified
        drift_detector.set_reference(MODEL_ID, X_ref)
    
    # Set performance baselines
    historical_metrics = provider.get_metrics(MODEL_ID, hours=168)  # Last week
    if not historical_metrics.empty and 'mae' in historical_metrics.columns:
        performance_tracker.establish_baseline_from_data(
            MODEL_ID,
            MetricType.MAE,
            historical_metrics['mae'].tolist()
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint with monitoring"""
    start_time = time.time()
    
    try:
        # Convert features to model input
        feature_values = np.array(list(request.features.values())).reshape(1, -1)
        
        # Make prediction
        prediction = float(model.predict(feature_values)[0])
        
        # Get confidence if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_values)[0]
            confidence = float(max(proba))
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Generate prediction ID
        prediction_id = f"pred_{int(time.time() * 1000)}"
        
        # --- MONITORING ---
        
        # Log prediction
        provider.log_prediction(
            model_id=MODEL_ID,
            features=request.features,
            prediction=prediction
        )
        
        # Log latency
        alert = performance_tracker.log_metric(
            MODEL_ID, 
            MetricType.PREDICTION_LATENCY, 
            latency_ms
        )
        
        # Check if alert was triggered
        if alert:
            alert_manager.create_alert(
                model_id=MODEL_ID,
                severity=AlertSeverity.HIGH,
                category=AlertCategory.LATENCY,
                title="High Prediction Latency",
                message=f"Latency {latency_ms:.1f}ms exceeded threshold",
                metric_value=latency_ms
            )
        
        # Check for drift (async in production)
        drift_report = drift_detector.detect_drift(MODEL_ID, feature_values)
        if drift_report.drift_detected:
            alert_manager.create_alert(
                model_id=MODEL_ID,
                severity=AlertSeverity.MEDIUM,
                category=AlertCategory.DRIFT,
                title="Data Drift Detected",
                message=f"Drift score: {drift_report.overall_drift_score:.4f}",
                metric_value=drift_report.overall_drift_score
            )
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            prediction_id=prediction_id,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        # Log error
        alert_manager.create_alert(
            model_id=MODEL_ID,
            severity=AlertSeverity.HIGH,
            category=AlertCategory.ERROR,
            title="Prediction Error",
            message=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def log_feedback(prediction_id: str, actual_value: float):
    """Log ground truth for a prediction"""
    # In production, update the prediction record with actual value
    # This enables accuracy tracking
    pass


@app.get("/health")
async def health_check():
    """Health check with monitoring status"""
    return {
        "status": "healthy",
        "model_id": MODEL_ID,
        "drift_detected": drift_detector.get_drift_trend(MODEL_ID).get('trend', 'unknown'),
        "active_alerts": len(alert_manager.get_active_alerts(MODEL_ID))
    }
'''
    
    print("FastAPI Integration Example:")
    print("=" * 50)
    print(code)


# =============================================================================
# EXAMPLE 3: Integration with Batch Processing (Airflow/Spark)
# =============================================================================

def example_batch_processing():
    """
    Example: Monitor batch prediction jobs
    
    Use this pattern with Airflow, Spark, or any batch processing framework.
    """
    
    code = '''
# airflow_dag.py - Example Airflow DAG with monitoring

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Monitoring imports
from backend.data_integration import DataProviderFactory, DatabaseConfig
from backend.core.drift_detector import DriftDetector
from backend.core.performance_tracker import PerformanceTracker, MetricType
from backend.core.alert_manager import AlertManager, AlertSeverity, AlertCategory

# Initialize monitoring (shared across tasks)
provider = DataProviderFactory.create('database', DatabaseConfig(...))
drift_detector = DriftDetector()
performance_tracker = PerformanceTracker()
alert_manager = AlertManager()

MODEL_ID = 'batch_model_v1'


def run_batch_predictions(**context):
    """Task: Run batch predictions"""
    import joblib
    
    # Load model
    model = joblib.load('/path/to/model.pkl')
    
    # Load batch data
    df = pd.read_parquet('/path/to/batch_data.parquet')
    X = df.drop('target', axis=1).values
    y_true = df['target'].values if 'target' in df.columns else None
    
    # Make predictions
    import time
    start_time = time.time()
    y_pred = model.predict(X)
    total_time = time.time() - start_time
    
    # Save predictions
    df['prediction'] = y_pred
    df.to_parquet('/path/to/predictions.parquet')
    
    # Store metrics for downstream tasks
    context['ti'].xcom_push('batch_size', len(X))
    context['ti'].xcom_push('total_time', total_time)
    context['ti'].xcom_push('predictions_path', '/path/to/predictions.parquet')
    
    return y_pred


def monitor_drift(**context):
    """Task: Check for data drift"""
    
    # Load batch data
    df = pd.read_parquet('/path/to/batch_data.parquet')
    X = df.drop('target', axis=1, errors='ignore').values
    feature_names = df.drop('target', axis=1, errors='ignore').columns.tolist()
    
    # Run drift detection
    drift_report = drift_detector.detect_drift(
        model_id=MODEL_ID,
        X_current=X
    )
    
    # Log drift metrics
    provider.log_metric(MODEL_ID, 'drift_score', drift_report.overall_drift_score)
    
    # Alert if drift detected
    if drift_report.drift_detected:
        alert_manager.create_alert(
            model_id=MODEL_ID,
            severity=AlertSeverity.HIGH,
            category=AlertCategory.DRIFT,
            title="Batch Data Drift Detected",
            message=f"Drift score: {drift_report.overall_drift_score:.4f}",
            details={
                'drifted_features': [
                    f for f, d in drift_report.feature_drift_scores.items()
                    if d.get('drift_detected')
                ]
            }
        )
        
        # Optionally fail the DAG
        # raise AirflowException("Data drift detected!")
    
    return drift_report.drift_detected


def log_performance(**context):
    """Task: Log batch performance metrics"""
    
    batch_size = context['ti'].xcom_pull(task_ids='run_predictions', key='batch_size')
    total_time = context['ti'].xcom_pull(task_ids='run_predictions', key='total_time')
    
    # Load predictions with actuals
    df = pd.read_parquet('/path/to/predictions.parquet')
    
    if 'target' in df.columns:
        y_true = df['target'].values
        y_pred = df['prediction'].values
        
        # Log performance
        metrics = performance_tracker.log_prediction(
            model_id=MODEL_ID,
            y_true=y_true,
            y_pred=y_pred,
            is_regression=True  # or False for classification
        )
        
        # Log to database
        for metric_type, value in metrics.items():
            provider.log_metric(MODEL_ID, metric_type.value, value)
    
    # Log batch metrics
    provider.log_metric(MODEL_ID, 'batch_size', batch_size)
    provider.log_metric(MODEL_ID, 'batch_processing_time_seconds', total_time)
    provider.log_metric(MODEL_ID, 'throughput_predictions_per_second', batch_size / total_time)
    
    print(f"Logged metrics for batch of {batch_size} predictions")


# Define DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ml_batch_predictions_with_monitoring',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
) as dag:
    
    predict_task = PythonOperator(
        task_id='run_predictions',
        python_callable=run_batch_predictions,
    )
    
    drift_task = PythonOperator(
        task_id='monitor_drift',
        python_callable=monitor_drift,
    )
    
    performance_task = PythonOperator(
        task_id='log_performance',
        python_callable=log_performance,
    )
    
    # Task dependencies
    predict_task >> [drift_task, performance_task]
'''
    
    print("Batch Processing Integration Example:")
    print("=" * 50)
    print(code)


# =============================================================================
# EXAMPLE 4: Integration with Your Trading Backend (enhprog.py)
# =============================================================================

def example_trading_backend_integration():
    """
    Example: Integrate monitoring with your existing trading backend
    """
    
    code = '''
# Integration with your enhprog.py trading backend
# Add this to your existing code

import sys
sys.path.append('/path/to/ml_monitor_platform')

from backend.data_integration import DataProviderFactory
from backend.core.drift_detector import DriftDetector
from backend.core.performance_tracker import PerformanceTracker, MetricType
from backend.core.model_explainer import ModelExplainer

# Initialize monitoring
provider = DataProviderFactory.create('file')  # or 'database'
drift_detector = DriftDetector()
performance_tracker = PerformanceTracker()
model_explainer = ModelExplainer()


# In your train_enhanced_models function, add:
def train_enhanced_models_with_monitoring(df, feature_cols, ticker, ...):
    """Enhanced training with monitoring integration"""
    
    # ... your existing training code ...
    
    # After training, set reference data for drift detection
    X_seq, y_seq, data_scaler = prepare_sequence_data(df, feature_cols, time_step)
    
    drift_detector.set_reference(
        model_id=f"{ticker}_ensemble",
        X_reference=X_seq,
        feature_names=feature_cols
    )
    
    # Set performance baselines
    if cv_results:
        for model_name, results in cv_results.items():
            if results.get('mean_score'):
                performance_tracker.establish_baseline_from_data(
                    model_id=f"{ticker}_{model_name}",
                    metric_type=MetricType.MSE,
                    values=[results['mean_score']]
                )
    
    return trained_models, data_scaler, config


# In your prediction function, add monitoring:
def make_prediction_with_monitoring(ticker, models, X_current, feature_cols):
    """Make prediction with drift detection and monitoring"""
    import time
    
    start_time = time.time()
    
    # ... your existing prediction code ...
    prediction = ensemble_predict(models, X_current)
    
    latency_ms = (time.time() - start_time) * 1000
    
    # --- MONITORING ---
    model_id = f"{ticker}_ensemble"
    
    # Log prediction
    provider.log_prediction(
        model_id=model_id,
        features={f: float(X_current[-1, -1, i]) for i, f in enumerate(feature_cols)},
        prediction=float(prediction)
    )
    
    # Log latency
    performance_tracker.log_metric(model_id, MetricType.PREDICTION_LATENCY, latency_ms)
    
    # Check drift
    drift_report = drift_detector.detect_drift(model_id, X_current)
    
    if drift_report.drift_detected:
        print(f"⚠️ Data drift detected for {ticker}!")
        print(f"   Drifted features: {[f for f, d in drift_report.feature_drift_scores.items() if d.get('drift_detected')]}")
    
    # Get explanation
    if models.get('xgboost'):
        explanation = model_explainer.explain_prediction(
            model_id=model_id,
            model=models['xgboost'],
            X=X_current.reshape(X_current.shape[0], -1),
            feature_names=feature_cols
        )
        print(f"Top features: {explanation.top_positive_features[:3]}")
    
    return prediction, drift_report


# In your Streamlit app, add monitoring dashboard link:
def render_monitoring_link():
    """Add link to monitoring dashboard in your trading app"""
    import streamlit as st
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Model Monitoring")
    
    if st.sidebar.button("Open Monitoring Dashboard"):
        # This assumes monitoring dashboard runs on port 8502
        st.sidebar.markdown("[Open Dashboard](http://localhost:8502)")
    
    # Show quick status
    provider = DataProviderFactory.create('file')
    alerts = provider.get_alerts()
    
    if alerts:
        critical = sum(1 for a in alerts if a['severity'] == 'critical')
        if critical > 0:
            st.sidebar.error(f"🚨 {critical} Critical Alerts!")
        else:
            st.sidebar.warning(f"⚠️ {len(alerts)} Active Alerts")
    else:
        st.sidebar.success("✅ All Models Healthy")
'''
    
    print("Trading Backend Integration Example:")
    print("=" * 50)
    print(code)


# =============================================================================
# RUN EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ML Model Monitor Pro - Integration Examples")
    print("=" * 60)
    
    print("\n1. Basic Integration (sklearn):")
    print("-" * 40)
    # example_basic_integration()  # Uncomment to run
    
    print("\n2. FastAPI Integration:")
    print("-" * 40)
    example_fastapi_integration()
    
    print("\n3. Batch Processing Integration:")
    print("-" * 40)
    example_batch_processing()
    
    print("\n4. Trading Backend Integration:")
    print("-" * 40)
    example_trading_backend_integration()
    
    print("\n" + "=" * 60)
    print("Choose the integration pattern that matches your use case!")
    print("=" * 60)
