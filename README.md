# 🔍 ML Model Monitor Pro

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

**Enterprise-grade ML model monitoring with drift detection, explainability & intelligent alerting**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Screenshots](#-screenshots)

</div>

---

## 🎯 Purpose

ML Model Monitor Pro is a comprehensive machine learning monitoring platform designed to help data science and ML engineering teams:

- **Detect problems early** - Catch model degradation, data drift, and anomalies before they impact business outcomes
- **Understand predictions** - Explainable AI features help build trust with stakeholders
- **Respond faster** - Intelligent alerting with actionable recommendations
- **Maintain compliance** - Track model performance for regulatory requirements

## 💎 Key Benefits

| Benefit | Description |
|---------|-------------|
| 🚨 **Early Detection** | Catch issues before they impact customers. Reduce MTTR by up to 80% |
| 🧠 **Explainability** | Understand why models make specific predictions |
| ⚡ **Faster Response** | Actionable recommendations with every alert |
| 📊 **360° Visibility** | Performance, latency, drift, and data quality in one place |
| 🔗 **Easy Integration** | REST API, webhooks, Slack, email notifications |

## ✨ Features

### 📈 Performance Monitoring
- Real-time tracking of MAE, RMSE, R², accuracy, precision, recall, F1
- Latency monitoring with SLA tracking (P50, P95, P99)
- Throughput analysis and prediction counts
- Resource utilization (CPU, memory)
- Historical trend visualization

### 🔄 Data Drift Detection
- **Population Stability Index (PSI)** - Industry standard drift metric
- **Kolmogorov-Smirnov Test** - Statistical distribution comparison
- **Wasserstein Distance** - Earth Mover's Distance
- **Jensen-Shannon Divergence** - Symmetric KL divergence
- Feature-level and overall drift scoring
- Adaptive thresholds based on baseline

### 🔎 Anomaly Detection
- Identify unusual predictions and out-of-distribution inputs
- Anomaly scoring with explanations
- Trend analysis (increasing/decreasing/stable)
- Configurable sensitivity thresholds

### 📋 Data Quality Profiling
- Missing value detection
- Feature statistics (mean, std, min, max)
- Outlier detection rates
- Data freshness monitoring
- Overall quality scoring

### 🧠 Model Explainability
- **SHAP Integration** - SHapley Additive exPlanations
- **Permutation Importance** - Model-agnostic feature importance
- **Gradient-based Importance** - For neural networks
- Feature contribution tracking over time
- Global and local explanations

### 🚨 Intelligent Alerting
- Multi-channel notifications (Slack, Email, Webhooks)
- Configurable alert rules and thresholds
- Alert deduplication and aggregation
- Cooldown periods to prevent alert fatigue
- Actionable recommendations with every alert

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-monitor-pro.git
cd ml-monitor-pro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

## 📁 Project Structure

```
ml_monitor_platform_v2/
├── app.py                      # Main Streamlit dashboard (enhanced UI)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── backend/
│   └── core/
│       ├── __init__.py
│       ├── drift_detector.py   # Drift detection engine (PSI, KS, Wasserstein)
│       ├── model_explainer.py  # SHAP, permutation, gradient importance
│       ├── performance_tracker.py  # Metrics, baselines, SLA tracking
│       └── alert_manager.py    # Multi-channel alerting
│
├── config/                     # Configuration files (optional)
│   ├── models.yaml
│   └── alerts.yaml
│
└── tests/                      # Unit tests
    └── test_*.py
```

## 📖 How to Use

### 1. Monitor Model Health
Start at the **Overview** section to see the health status of all deployed models:
- 🟢 **Healthy** - All metrics within normal range
- 🟡 **Warning** - Some metrics showing drift or degradation
- 🔴 **Critical** - Immediate attention required

### 2. Analyze Performance Trends
Use the **Performance** tab to track metrics over time:
- Look for upward trends in error metrics (MAE, RMSE)
- Monitor latency against SLA thresholds
- Check throughput patterns

### 3. Detect Data Drift
The **Drift Detection** tab shows:
- PSI scores for each feature (< 0.1 good, > 0.2 critical)
- Statistical tests (KS-test, Wasserstein distance)
- Which features have drifted

### 4. Investigate Anomalies
**Anomaly Detection** helps identify:
- Unusual predictions
- Out-of-distribution inputs
- Potential model issues

### 5. Understand Predictions
**Explainability** tab provides:
- Feature importance rankings
- Which features drive predictions
- Positive vs negative impacts

### 6. Respond to Alerts
**Alerts** tab shows active issues with:
- Severity levels (Critical, High, Medium, Low)
- Actionable recommendations
- Time to respond guidance

## 📊 Metric Thresholds Reference

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| **PSI** | < 0.1 | 0.1 - 0.2 | > 0.2 |
| **Latency** | < 50ms | 50-100ms | > 100ms |
| **Anomaly Rate** | < 0.5% | 0.5-1% | > 1% |
| **Error Rate** | < 0.1% | 0.1-1% | > 1% |

## 🔌 Integration Options

### REST API
```python
# Log a prediction
POST /api/v1/predictions/{model_id}
{
  "features": {...},
  "prediction": 0.85,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Webhook Notifications
```json
{
  "webhook_url": "https://your-server.com/alerts",
  "events": ["drift_detected", "performance_degradation", "sla_violation"]
}
```

### Slack Integration
```python
from backend.core.alert_manager import AlertManager, SlackNotificationChannel

alert_manager = AlertManager()
alert_manager.register_channel(
    "slack",
    SlackNotificationChannel(webhook_url="https://hooks.slack.com/...")
)
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html
```

## 📈 Supported Model Types

| Framework | Support Level |
|-----------|--------------|
| Scikit-learn | ✅ Full |
| XGBoost | ✅ Full |
| LightGBM | ✅ Full |
| PyTorch | ✅ Full |
| TensorFlow/Keras | ✅ Full |
| Custom Models | ✅ Via predict() interface |

## 🗺️ Roadmap

- [ ] A/B testing support
- [ ] Automated retraining triggers
- [ ] MLflow integration
- [ ] Kubernetes deployment templates
- [ ] Real-time streaming (Kafka, Kinesis)
- [ ] Custom metric plugins
- [ ] Multi-tenant support

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Explainability powered by [SHAP](https://github.com/slundberg/shap)
- Charts by [Plotly](https://plotly.com/)

---

<div align="center">

**Made with ❤️ for the ML community**

[⬆ Back to Top](#-ml-model-monitor-pro)

</div>
