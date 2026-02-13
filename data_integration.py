# ML Model Monitoring Platform - Data Integration Module
# =============================================================================
# Supports multiple data sources: Database, API, Files, Real-time streams
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import json
import os
from pathlib import Path

# Optional imports for various data sources
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DATA SOURCE CONFIGURATIONS
# =============================================================================

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "ml_monitoring"
    username: str = ""
    password: str = ""
    driver: str = "postgresql"  # postgresql, mysql, sqlite
    
    @property
    def connection_string(self) -> str:
        if self.driver == "sqlite":
            return f"sqlite:///{self.database}"
        return f"{self.driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class APIConfig:
    """API endpoint configuration"""
    base_url: str = "http://localhost:8000"
    api_key: str = ""
    timeout: int = 30
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class FileConfig:
    """File-based data source configuration"""
    data_directory: str = "./data"
    metrics_file: str = "metrics.csv"
    predictions_file: str = "predictions.csv"
    models_file: str = "models.json"


# =============================================================================
# ABSTRACT DATA PROVIDER
# =============================================================================

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def get_models(self) -> Dict[str, Dict]:
        """Get all registered models"""
        pass
    
    @abstractmethod
    def get_metrics(self, model_id: str, hours: int = 24) -> pd.DataFrame:
        """Get performance metrics for a model"""
        pass
    
    @abstractmethod
    def get_predictions(self, model_id: str, limit: int = 1000) -> pd.DataFrame:
        """Get recent predictions for drift analysis"""
        pass
    
    @abstractmethod
    def get_reference_data(self, model_id: str) -> pd.DataFrame:
        """Get reference/training data distribution"""
        pass
    
    @abstractmethod
    def get_alerts(self, model_id: str = None) -> List[Dict]:
        """Get active alerts"""
        pass
    
    @abstractmethod
    def log_metric(self, model_id: str, metric_name: str, value: float):
        """Log a new metric value"""
        pass
    
    @abstractmethod
    def log_prediction(self, model_id: str, features: Dict, prediction: float, 
                       actual: Optional[float] = None):
        """Log a prediction"""
        pass


# =============================================================================
# DATABASE DATA PROVIDER
# =============================================================================

class DatabaseDataProvider(DataProvider):
    """
    Database-backed data provider.
    Supports PostgreSQL, MySQL, SQLite.
    
    Required Tables:
    - models: Model registry
    - metrics: Time-series metrics
    - predictions: Prediction logs
    - reference_data: Training data statistics
    - alerts: Alert history
    """
    
    def __init__(self, config: DatabaseConfig):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("sqlalchemy is required for database provider")
        
        self.config = config
        self.engine = create_engine(config.connection_string)
        logger.info(f"Connected to database: {config.database}")
    
    def get_models(self) -> Dict[str, Dict]:
        query = """
        SELECT 
            model_id, name, type, framework, version,
            deployed_at, status, predictions_today,
            avg_latency_ms, accuracy, mae
        FROM models
        WHERE is_active = true
        """
        
        df = pd.read_sql(query, self.engine)
        
        models = {}
        for _, row in df.iterrows():
            models[row['model_id']] = row.to_dict()
        
        return models
    
    def get_metrics(self, model_id: str, hours: int = 24) -> pd.DataFrame:
        query = """
        SELECT 
            timestamp, metric_name, metric_value, batch_size
        FROM metrics
        WHERE model_id = :model_id
          AND timestamp > :cutoff
        ORDER BY timestamp
        """
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        df = pd.read_sql(
            text(query), 
            self.engine,
            params={'model_id': model_id, 'cutoff': cutoff}
        )
        
        # Pivot to wide format
        if not df.empty:
            df = df.pivot(index='timestamp', columns='metric_name', values='metric_value')
            df = df.reset_index()
        
        return df
    
    def get_predictions(self, model_id: str, limit: int = 1000) -> pd.DataFrame:
        query = """
        SELECT 
            prediction_id, timestamp, features, prediction,
            actual, latency_ms, is_anomaly
        FROM predictions
        WHERE model_id = :model_id
        ORDER BY timestamp DESC
        LIMIT :limit
        """
        
        df = pd.read_sql(
            text(query),
            self.engine,
            params={'model_id': model_id, 'limit': limit}
        )
        
        # Parse JSON features column
        if 'features' in df.columns:
            df['features'] = df['features'].apply(json.loads)
        
        return df
    
    def get_reference_data(self, model_id: str) -> pd.DataFrame:
        query = """
        SELECT 
            feature_name, mean_value, std_value, min_value, max_value,
            percentile_25, percentile_50, percentile_75
        FROM reference_data
        WHERE model_id = :model_id
        """
        
        return pd.read_sql(text(query), self.engine, params={'model_id': model_id})
    
    def get_alerts(self, model_id: str = None) -> List[Dict]:
        query = """
        SELECT 
            alert_id, model_id, severity, category, title, message,
            timestamp, status, recommendations
        FROM alerts
        WHERE status = 'active'
        """
        
        if model_id:
            query += " AND model_id = :model_id"
        
        query += " ORDER BY timestamp DESC"
        
        params = {'model_id': model_id} if model_id else {}
        df = pd.read_sql(text(query), self.engine, params=params)
        
        alerts = []
        for _, row in df.iterrows():
            alert = row.to_dict()
            if isinstance(alert.get('recommendations'), str):
                alert['recommendations'] = json.loads(alert['recommendations'])
            alerts.append(alert)
        
        return alerts
    
    def log_metric(self, model_id: str, metric_name: str, value: float):
        query = """
        INSERT INTO metrics (model_id, metric_name, metric_value, timestamp)
        VALUES (:model_id, :metric_name, :value, :timestamp)
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(query), {
                'model_id': model_id,
                'metric_name': metric_name,
                'value': value,
                'timestamp': datetime.now()
            })
            conn.commit()
    
    def log_prediction(self, model_id: str, features: Dict, prediction: float,
                       actual: Optional[float] = None):
        query = """
        INSERT INTO predictions (model_id, features, prediction, actual, timestamp)
        VALUES (:model_id, :features, :prediction, :actual, :timestamp)
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(query), {
                'model_id': model_id,
                'features': json.dumps(features),
                'prediction': prediction,
                'actual': actual,
                'timestamp': datetime.now()
            })
            conn.commit()


# =============================================================================
# API DATA PROVIDER
# =============================================================================

class APIDataProvider(DataProvider):
    """
    API-backed data provider.
    Connects to your ML platform's REST API.
    """
    
    def __init__(self, config: APIConfig):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is required for API provider")
        
        self.config = config
        self.session = requests.Session()
        
        # Set headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {config.api_key}',
            **config.headers
        })
        
        logger.info(f"Connected to API: {config.base_url}")
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        url = f"{self.config.base_url}{endpoint}"
        response = self.session.request(method, url, timeout=self.config.timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def get_models(self) -> Dict[str, Dict]:
        data = self._request('GET', '/api/v1/models')
        return {m['model_id']: m for m in data.get('models', [])}
    
    def get_metrics(self, model_id: str, hours: int = 24) -> pd.DataFrame:
        data = self._request('GET', f'/api/v1/models/{model_id}/metrics', 
                            params={'hours': hours})
        return pd.DataFrame(data.get('metrics', []))
    
    def get_predictions(self, model_id: str, limit: int = 1000) -> pd.DataFrame:
        data = self._request('GET', f'/api/v1/models/{model_id}/predictions',
                            params={'limit': limit})
        return pd.DataFrame(data.get('predictions', []))
    
    def get_reference_data(self, model_id: str) -> pd.DataFrame:
        data = self._request('GET', f'/api/v1/models/{model_id}/reference')
        return pd.DataFrame(data.get('reference', []))
    
    def get_alerts(self, model_id: str = None) -> List[Dict]:
        endpoint = f'/api/v1/models/{model_id}/alerts' if model_id else '/api/v1/alerts'
        data = self._request('GET', endpoint)
        return data.get('alerts', [])
    
    def log_metric(self, model_id: str, metric_name: str, value: float):
        self._request('POST', f'/api/v1/models/{model_id}/metrics', json={
            'metric_name': metric_name,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_prediction(self, model_id: str, features: Dict, prediction: float,
                       actual: Optional[float] = None):
        self._request('POST', f'/api/v1/models/{model_id}/predictions', json={
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'timestamp': datetime.now().isoformat()
        })


# =============================================================================
# FILE-BASED DATA PROVIDER
# =============================================================================

class FileDataProvider(DataProvider):
    """
    File-based data provider for local/testing use.
    Reads from CSV/JSON files.
    """
    
    def __init__(self, config: FileConfig):
        self.config = config
        self.data_dir = Path(config.data_directory)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using file data directory: {self.data_dir}")
    
    def get_models(self) -> Dict[str, Dict]:
        models_file = self.data_dir / self.config.models_file
        
        if models_file.exists():
            with open(models_file) as f:
                return json.load(f)
        
        return {}
    
    def get_metrics(self, model_id: str, hours: int = 24) -> pd.DataFrame:
        metrics_file = self.data_dir / f"{model_id}_{self.config.metrics_file}"
        
        if metrics_file.exists():
            df = pd.read_csv(metrics_file, parse_dates=['timestamp'])
            cutoff = datetime.now() - timedelta(hours=hours)
            return df[df['timestamp'] > cutoff]
        
        return pd.DataFrame()
    
    def get_predictions(self, model_id: str, limit: int = 1000) -> pd.DataFrame:
        predictions_file = self.data_dir / f"{model_id}_{self.config.predictions_file}"
        
        if predictions_file.exists():
            df = pd.read_csv(predictions_file, parse_dates=['timestamp'])
            return df.tail(limit)
        
        return pd.DataFrame()
    
    def get_reference_data(self, model_id: str) -> pd.DataFrame:
        ref_file = self.data_dir / f"{model_id}_reference.csv"
        
        if ref_file.exists():
            return pd.read_csv(ref_file)
        
        return pd.DataFrame()
    
    def get_alerts(self, model_id: str = None) -> List[Dict]:
        alerts_file = self.data_dir / "alerts.json"
        
        if alerts_file.exists():
            with open(alerts_file) as f:
                alerts = json.load(f)
            
            if model_id:
                alerts = [a for a in alerts if a.get('model_id') == model_id]
            
            return alerts
        
        return []
    
    def log_metric(self, model_id: str, metric_name: str, value: float):
        metrics_file = self.data_dir / f"{model_id}_{self.config.metrics_file}"
        
        new_row = pd.DataFrame([{
            'timestamp': datetime.now(),
            'metric_name': metric_name,
            'value': value
        }])
        
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
        
        df.to_csv(metrics_file, index=False)
    
    def log_prediction(self, model_id: str, features: Dict, prediction: float,
                       actual: Optional[float] = None):
        predictions_file = self.data_dir / f"{model_id}_{self.config.predictions_file}"
        
        new_row = pd.DataFrame([{
            'timestamp': datetime.now(),
            'features': json.dumps(features),
            'prediction': prediction,
            'actual': actual
        }])
        
        if predictions_file.exists():
            df = pd.read_csv(predictions_file)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
        
        df.to_csv(predictions_file, index=False)


# =============================================================================
# DEMO DATA PROVIDER (FOR TESTING)
# =============================================================================

class DemoDataProvider(DataProvider):
    """
    Demo data provider that generates synthetic data.
    Use this for testing and demonstrations.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        logger.info("Using demo data provider (synthetic data)")
    
    def get_models(self) -> Dict[str, Dict]:
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
    
    def get_metrics(self, model_id: str, hours: int = 24) -> pd.DataFrame:
        np.random.seed(hash(model_id) % 2**32)
        
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=hours * 6,
            freq='10min'
        )
        
        is_critical = 'model_004' in model_id
        base_mae = 0.05 + np.random.random() * 0.02
        drift_factor = np.linspace(0, 0.03 if is_critical else 0.005, len(timestamps))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'mae': base_mae + np.random.normal(0, 0.005, len(timestamps)) + drift_factor,
            'rmse': (base_mae + np.random.normal(0, 0.008, len(timestamps)) + drift_factor) * 1.2,
            'accuracy': np.clip(0.92 - drift_factor + np.random.normal(0, 0.01, len(timestamps)), 0.7, 1.0),
            'latency_ms': 20 + np.random.exponential(10 if not is_critical else 30, len(timestamps)),
            'predictions_count': np.random.poisson(100, len(timestamps)),
            'error_rate': np.clip(0.001 + np.random.exponential(0.002, len(timestamps)), 0, 0.1),
            'memory_usage_mb': 200 + np.random.normal(0, 20, len(timestamps)),
            'cpu_usage_percent': 30 + np.random.normal(0, 10, len(timestamps))
        })
    
    def get_predictions(self, model_id: str, limit: int = 1000) -> pd.DataFrame:
        np.random.seed(hash(model_id) % 2**32)
        
        features = ['age', 'income', 'tenure', 'usage_frequency', 'support_tickets']
        
        data = []
        for i in range(limit):
            row = {
                'prediction_id': f'pred_{i}',
                'timestamp': datetime.now() - timedelta(minutes=i),
                'prediction': np.random.random(),
                'actual': np.random.random() if np.random.random() > 0.3 else None,
                'latency_ms': np.random.exponential(20)
            }
            
            for f in features:
                row[f] = np.random.normal(50, 20)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_reference_data(self, model_id: str) -> pd.DataFrame:
        features = ['age', 'income', 'tenure', 'usage_frequency', 'support_tickets']
        
        return pd.DataFrame([
            {
                'feature_name': f,
                'mean_value': np.random.uniform(30, 70),
                'std_value': np.random.uniform(10, 30),
                'min_value': np.random.uniform(0, 20),
                'max_value': np.random.uniform(80, 150)
            }
            for f in features
        ])
    
    def get_alerts(self, model_id: str = None) -> List[Dict]:
        all_alerts = [
            {
                'alert_id': 'alert_001',
                'model_id': 'model_002',
                'severity': 'medium',
                'category': 'drift',
                'title': 'Minor Data Drift Detected',
                'message': 'Feature "usage_frequency" showing slight distribution shift (PSI: 0.12)',
                'timestamp': datetime.now() - timedelta(hours=3),
                'status': 'active',
                'recommendations': ['Monitor for next 24 hours', 'Review data pipeline']
            },
            {
                'alert_id': 'alert_002',
                'model_id': 'model_004',
                'severity': 'critical',
                'category': 'performance',
                'title': 'Severe Model Performance Degradation',
                'message': 'MAE increased by 45% over the last 24 hours',
                'timestamp': datetime.now() - timedelta(hours=1),
                'status': 'active',
                'recommendations': ['Immediate investigation required', 'Consider rollback']
            },
            {
                'alert_id': 'alert_003',
                'model_id': 'model_004',
                'severity': 'high',
                'category': 'drift',
                'title': 'Significant Data Drift Detected',
                'message': 'Multiple features showing drift: income (PSI: 0.28), tenure (PSI: 0.19)',
                'timestamp': datetime.now() - timedelta(hours=2),
                'status': 'active',
                'recommendations': ['Retrain model with recent data']
            }
        ]
        
        if model_id:
            return [a for a in all_alerts if a['model_id'] == model_id]
        
        return all_alerts
    
    def log_metric(self, model_id: str, metric_name: str, value: float):
        logger.info(f"[DEMO] Logged metric: {model_id}/{metric_name} = {value}")
    
    def log_prediction(self, model_id: str, features: Dict, prediction: float,
                       actual: Optional[float] = None):
        logger.info(f"[DEMO] Logged prediction: {model_id} = {prediction}")


# =============================================================================
# DATA PROVIDER FACTORY
# =============================================================================

class DataProviderFactory:
    """Factory to create data providers based on configuration"""
    
    @staticmethod
    def create(provider_type: str, config: Any = None) -> DataProvider:
        """
        Create a data provider.
        
        Args:
            provider_type: One of 'demo', 'database', 'api', 'file'
            config: Provider-specific configuration
            
        Returns:
            DataProvider instance
        """
        if provider_type == 'demo':
            return DemoDataProvider()
        
        elif provider_type == 'database':
            if config is None:
                config = DatabaseConfig()
            return DatabaseDataProvider(config)
        
        elif provider_type == 'api':
            if config is None:
                config = APIConfig()
            return APIDataProvider(config)
        
        elif provider_type == 'file':
            if config is None:
                config = FileConfig()
            return FileDataProvider(config)
        
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    @staticmethod
    def from_env() -> DataProvider:
        """
        Create data provider from environment variables.
        
        Environment variables:
            ML_MONITOR_PROVIDER: demo, database, api, file
            ML_MONITOR_DB_HOST: Database host
            ML_MONITOR_DB_PORT: Database port
            ML_MONITOR_DB_NAME: Database name
            ML_MONITOR_DB_USER: Database username
            ML_MONITOR_DB_PASS: Database password
            ML_MONITOR_API_URL: API base URL
            ML_MONITOR_API_KEY: API key
            ML_MONITOR_DATA_DIR: File data directory
        """
        provider_type = os.getenv('ML_MONITOR_PROVIDER', 'demo')
        
        if provider_type == 'database':
            config = DatabaseConfig(
                host=os.getenv('ML_MONITOR_DB_HOST', 'localhost'),
                port=int(os.getenv('ML_MONITOR_DB_PORT', '5432')),
                database=os.getenv('ML_MONITOR_DB_NAME', 'ml_monitoring'),
                username=os.getenv('ML_MONITOR_DB_USER', ''),
                password=os.getenv('ML_MONITOR_DB_PASS', '')
            )
            return DatabaseDataProvider(config)
        
        elif provider_type == 'api':
            config = APIConfig(
                base_url=os.getenv('ML_MONITOR_API_URL', 'http://localhost:8000'),
                api_key=os.getenv('ML_MONITOR_API_KEY', '')
            )
            return APIDataProvider(config)
        
        elif provider_type == 'file':
            config = FileConfig(
                data_directory=os.getenv('ML_MONITOR_DATA_DIR', './data')
            )
            return FileDataProvider(config)
        
        else:
            return DemoDataProvider()


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
# Example 1: Using Demo Provider (default)
from data_integration import DataProviderFactory

provider = DataProviderFactory.create('demo')
models = provider.get_models()
metrics = provider.get_metrics('model_001', hours=24)


# Example 2: Using Database Provider
from data_integration import DataProviderFactory, DatabaseConfig

config = DatabaseConfig(
    host='localhost',
    port=5432,
    database='ml_monitoring',
    username='admin',
    password='secret'
)
provider = DataProviderFactory.create('database', config)


# Example 3: Using API Provider
from data_integration import DataProviderFactory, APIConfig

config = APIConfig(
    base_url='https://api.mymlplatform.com',
    api_key='your-api-key'
)
provider = DataProviderFactory.create('api', config)


# Example 4: Using File Provider
from data_integration import DataProviderFactory, FileConfig

config = FileConfig(
    data_directory='./monitoring_data'
)
provider = DataProviderFactory.create('file', config)


# Example 5: Using Environment Variables
# Set: ML_MONITOR_PROVIDER=database
# Set: ML_MONITOR_DB_HOST=localhost
# etc.

provider = DataProviderFactory.from_env()


# Example 6: Logging predictions from your ML pipeline
provider.log_prediction(
    model_id='model_001',
    features={'age': 35, 'income': 75000, 'tenure': 24},
    prediction=0.85,
    actual=1.0  # Optional: set when ground truth is available
)

provider.log_metric('model_001', 'latency_ms', 23.5)
"""
