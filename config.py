# ML Model Monitor Pro - Configuration
# =============================================================================
# Configure your data source here
# =============================================================================

import os

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# Choose your data provider: 'demo', 'database', 'api', 'file'
DATA_PROVIDER = os.getenv('ML_MONITOR_PROVIDER', 'demo')

# -----------------------------------------------------------------------------
# DATABASE CONFIGURATION (PostgreSQL, MySQL, SQLite)
# -----------------------------------------------------------------------------
DATABASE_CONFIG = {
    'host': os.getenv('ML_MONITOR_DB_HOST', 'localhost'),
    'port': int(os.getenv('ML_MONITOR_DB_PORT', '5432')),
    'database': os.getenv('ML_MONITOR_DB_NAME', 'ml_monitoring'),
    'username': os.getenv('ML_MONITOR_DB_USER', 'admin'),
    'password': os.getenv('ML_MONITOR_DB_PASS', ''),
    'driver': os.getenv('ML_MONITOR_DB_DRIVER', 'postgresql')  # postgresql, mysql, sqlite
}

# -----------------------------------------------------------------------------
# API CONFIGURATION
# -----------------------------------------------------------------------------
API_CONFIG = {
    'base_url': os.getenv('ML_MONITOR_API_URL', 'http://localhost:8000'),
    'api_key': os.getenv('ML_MONITOR_API_KEY', ''),
    'timeout': int(os.getenv('ML_MONITOR_API_TIMEOUT', '30'))
}

# -----------------------------------------------------------------------------
# FILE-BASED CONFIGURATION
# -----------------------------------------------------------------------------
FILE_CONFIG = {
    'data_directory': os.getenv('ML_MONITOR_DATA_DIR', './data'),
    'metrics_file': 'metrics.csv',
    'predictions_file': 'predictions.csv',
    'models_file': 'models.json'
}

# =============================================================================
# ALERTING CONFIGURATION
# =============================================================================

ALERT_CONFIG = {
    # Slack notifications
    'slack_webhook_url': os.getenv('SLACK_WEBHOOK_URL', ''),
    'slack_channel': os.getenv('SLACK_CHANNEL', '#ml-alerts'),
    
    # Email notifications
    'smtp_host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
    'smtp_username': os.getenv('SMTP_USERNAME', ''),
    'smtp_password': os.getenv('SMTP_PASSWORD', ''),
    'alert_recipients': os.getenv('ALERT_RECIPIENTS', '').split(','),
    
    # Alert thresholds
    'psi_warning_threshold': 0.1,
    'psi_critical_threshold': 0.2,
    'latency_warning_ms': 50,
    'latency_critical_ms': 100,
    'error_rate_warning': 0.005,
    'error_rate_critical': 0.01,
    
    # Cooldown (prevent alert spam)
    'alert_cooldown_minutes': 30
}

# =============================================================================
# MONITORING CONFIGURATION  
# =============================================================================

MONITORING_CONFIG = {
    # Refresh intervals
    'metrics_refresh_seconds': 60,
    'drift_check_interval_minutes': 15,
    
    # Data retention
    'metrics_retention_days': 30,
    'predictions_retention_days': 7,
    
    # Drift detection
    'reference_window_size': 1000,
    'detection_window_size': 100,
    
    # Performance baselines (auto-established if not set)
    'baseline_lookback_days': 7
}

# =============================================================================
# UI CONFIGURATION
# =============================================================================

UI_CONFIG = {
    'page_title': 'ML Model Monitor Pro',
    'page_icon': '🔍',
    'layout': 'wide',
    'theme_primary_color': '#4F46E5',
    'max_models_display': 10,
    'chart_height': 400
}
