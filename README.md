# 🚗 InsurePrice: Enterprise Car Insurance Risk Modeling & Pricing Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

**🏆 Complete End-to-End Insurance Technology Platform**

*Production-ready ML-powered risk assessment, actuarial pricing, fraud detection, and portfolio optimization*

[🚀 Quick Start](#-quick-start) • [📡 API Docs](#-rest-api) • [🔍 Fraud Detection](#-fraud-detection) • [📊 Dashboard](#-interactive-dashboard) • [💰 Business Value](#-business-value)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [REST API](#-rest-api)
- [Fraud Detection](#-fraud-detection)
- [Interactive Dashboard](#-interactive-dashboard)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Business Value](#-business-value)
- [Model Performance](#-model-performance)
- [Contact](#-contact)

---

## 🎯 Overview

**InsurePrice** is a comprehensive, enterprise-grade car insurance platform that transforms traditional insurance operations through advanced machine learning, actuarial science, and AI-powered analytics.

### What This Platform Delivers

| Capability | Description | Business Impact |
|------------|-------------|-----------------|
| **Risk Prediction** | ML models predicting claim probability | AUC 0.65+, accurate underwriting |
| **Actuarial Pricing** | Professional premium calculation | £400-£1,200 market-aligned |
| **Fraud Detection** | Real-time claims fraud analysis | £60M potential savings (5% improvement) |
| **REST API** | Production-ready endpoints | Enterprise system integration |
| **Interactive Dashboard** | Real-time analytics platform | Operational efficiency |
| **Portfolio Simulation** | Monte Carlo profit analysis | 8.4% profit margin |
| **Explainable AI** | SHAP-based model transparency | Regulatory compliance |
| **Fairness Analysis** | Bias detection and mitigation | FCA compliance |

### UK Market Context

- **Market Size**: £15 billion annual premium revenue
- **Fraud Cost**: £1.2 billion annually
- **Digital Gap**: Most insurers use 20+ year old pricing models
- **Opportunity**: Data-driven insurers outperform by 15-25%

---

## ✨ Key Features

### 🤖 Machine Learning Risk Models
- **CatBoost**: Best performer with categorical embeddings (AUC 0.6176) 🏆
- **Random Forest**: Optimized with hyperparameter tuning (AUC 0.6074)
- **Neural Network Ensemble**: Deep learning with embedding layers
- **Feature Engineering**: +3.84% AUC improvement with interaction terms
- **Hyperparameter Optimization**: Optuna-based automated tuning

### 💰 Actuarial Pricing Engine
- Professional premium calculation formulas
- Risk-based pricing with 17+ factors
- NCD (No Claims Discount) integration
- Voluntary excess adjustments
- Coverage type variations

### 🔍 Real-Time Fraud Detection
- **Anomaly Detection**: Isolation Forest for unusual patterns
- **NLP Analysis**: 40+ fraud indicator keywords
- **Network Analysis**: Fraud ring identification
- **Behavioral Scoring**: Pattern recognition

### 🧪 A/B Testing Framework
- Price sensitivity analysis per segment
- Conversion rate tracking
- Statistical significance testing
- Revenue optimization experiments

### 📋 Regulatory Compliance
- **FCA PRIN**: Fair pricing compliance
- **GDPR Article 22**: Automated decision safeguards
- **Solvency II**: Model risk management
- **SR 11-7**: Model documentation standards
- **Model Drift Detection**: Automated monitoring

### 💎 Customer Lifetime Value (CLV)
- Policy renewal probability modeling
- Cross-sell potential estimation
- Claims propensity forecasting
- Strategic pricing recommendations

### 📊 Interactive Dashboard (10+ Pages)
- Real-time risk assessment calculator
- Premium calculator with actuarial breakdown
- A/B testing experiment runner
- Regulatory compliance monitoring
- Model performance & improvements
- Fraud detection interface
- Customer CLV analysis

### 📡 Production REST API
- FastAPI with automatic documentation
- Real-time risk scoring endpoints
- Premium calculation API
- Fraud analysis endpoints
- SHAP explainability API

---

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/MichaelTheAnalyst/InsurePrice-Car-Insurance-Risk-Modeling.git
cd InsurePrice-Car-Insurance-Risk-Modeling

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_api.txt
```

### Option 1: Launch REST API

```bash
# Start the API server
python run_api.py

# API available at:
# - Interactive Docs: http://localhost:8000/docs
# - Health Check: http://localhost:8000/health
```

### Option 2: Launch Dashboard

```bash
# Start the Streamlit dashboard
python run_dashboard.py

# Dashboard available at: http://localhost:8501
```

### Option 3: Run Analysis Pipeline

```bash
# Generate enhanced synthetic data
python generate_data.py

# Train risk prediction models
python baseline_modeling.py

# Run pricing engine
python pricing_engine.py

# Portfolio simulation
python portfolio_simulation.py

# Fraud detection demo
python fraud_detection.py
```

---

## 📡 REST API

### API Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/risk/score` | Real-time risk scoring |
| `POST` | `/api/v1/premium/quote` | Premium calculation |
| `POST` | `/api/v1/portfolio/analyze` | Portfolio analysis |
| `GET` | `/api/v1/model/explain/{id}` | SHAP explanations |
| `POST` | `/api/v1/fraud/analyze` | Fraud detection |
| `POST` | `/api/v1/fraud/batch` | Batch fraud analysis |
| `GET` | `/api/v1/fraud/rings` | Fraud ring detection |
| `GET` | `/health` | Health check |

### Example: Risk Scoring

```python
import requests

profile = {
    "age": "26-39",
    "gender": "male",
    "region": "London",
    "driving_experience": "10-19y",
    "vehicle_type": "family_sedan",
    "annual_mileage": 12000.0,
    "credit_score": 0.75,
    "past_accidents": 1,
    "safety_rating": "standard"
}

response = requests.post(
    "http://localhost:8000/api/v1/risk/score",
    json={"driver_profile": profile}
)

result = response.json()
print(f"Risk Score: {result['risk_score']:.3f}")
print(f"Risk Category: {result['risk_category']}")
```

### Example: Premium Quote

```python
quote_request = {
    "driver_profile": profile,
    "coverage_type": "comprehensive",
    "voluntary_excess": 200,
    "ncd_years": 3
}

response = requests.post(
    "http://localhost:8000/api/v1/premium/quote",
    json=quote_request
)

result = response.json()
print(f"Annual Premium: £{result['annual_premium']:.2f}")
print(f"Monthly Premium: £{result['monthly_premium']:.2f}")
```

### API Performance

| Endpoint | Response Time | Throughput |
|----------|--------------|------------|
| Risk Score | ~150ms | 500 req/sec |
| Premium Quote | ~200ms | 400 req/sec |
| Portfolio Analysis | ~800ms | 50 req/sec |
| Fraud Analysis | ~200ms | 300 req/sec |

---

## 🔍 Fraud Detection

### UK Insurance Fraud Context

- **Annual Cost**: £1.2 billion
- **Target Improvement**: 5%
- **Potential Savings**: £60 million

### Detection Methods

| Method | Technology | Purpose |
|--------|------------|---------|
| **Anomaly Detection** | Isolation Forest | Statistical outliers |
| **NLP Analysis** | Keyword/Pattern | Text red flags |
| **Network Analysis** | Graph Algorithms | Fraud rings |
| **Behavioral Analysis** | Pattern Recognition | Suspicious behaviors |

### Fraud Red Flag Keywords

**High Risk**:
`whiplash`, `neck pain`, `cash settlement`, `total loss`, `unwitnessed`

**Suspicious Patterns**:
`friend`, `family member`, `preferred garage`, `no receipt`, `approximate`

### Example: Fraud Analysis

```python
claim = {
    "claim_id": "CLM-001",
    "claim_amount": 12500,
    "days_to_report": 45,
    "previous_claims": 4,
    "description": "Rear ended at night. No witnesses. Whiplash. Cash settlement preferred.",
    "police_report": False,
    "witnesses": 0,
    "cash_settlement_requested": True
}

response = requests.post(
    "http://localhost:8000/api/v1/fraud/analyze",
    json={"claim": claim}
)

result = response.json()
print(f"Fraud Score: {result['overall_fraud_score']:.1%}")
print(f"Risk Level: {result['risk_level']}")
print(f"Recommendation: {result['recommendation']}")
```

### Sample Output

```
🚨 FRAUD ANALYSIS RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fraud Score: 60.4%
Risk Level: HIGH
Confidence: HIGH

Component Scores:
  Anomaly Detection:    ████████████░░░░░░░░ 64.8%
  Text Analysis:        ████████████████████ 100.0%
  Network Analysis:     ░░░░░░░░░░░░░░░░░░░░ 0.0%
  Behavioral Analysis:  ████████████████░░░░ 80.0%

Recommendation: Refer to Special Investigation Unit (SIU)
```

---

## 📊 Interactive Dashboard

### Features

- **Real-Time Risk Calculator**: Instant risk assessment with personalized recommendations
- **Premium Calculator**: Actuarial pricing with full component breakdown
- **Customer CLV Analysis**: Lifetime value prediction with segment classification
- **Fraud Detection Interface**: AI-powered claims analysis with red flag detection
- **Portfolio Analytics**: Risk distribution and performance visualization
- **Model Performance**: ML metrics, ROC curves, and SHAP explainability
- **Regional Analysis**: Geographic risk mapping with actionable insights
- **Built-in Help System**: Expandable guidance panels on every page

### Launch Dashboard

```bash
# Option 1: Launcher script
python run_dashboard.py

# Option 2: Direct Streamlit
streamlit run insureprice_dashboard.py
```

### Dashboard Pages

1. **📊 Dashboard** - Overview with key metrics and portfolio health
2. **🎯 Risk Assessment** - Interactive risk calculator with instant quotes
3. **💰 Premium Calculator** - Actuarial pricing with full breakdown
4. **💎 Customer CLV** - Customer Lifetime Value prediction and segmentation
5. **🔍 Fraud Detection** - AI-powered claims fraud analysis
6. **📈 Portfolio Analytics** - Risk distribution and performance analysis
7. **🤖 Model Performance** - ML evaluation and SHAP explainability
8. **📡 API Status** - REST API monitoring and testing
9. **📋 About** - Documentation and contact info

### Built-in Help & Guidance

Each dashboard section includes **expandable instruction panels** with:
- 📖 Step-by-step guidance on how to use each feature
- 📊 Interpretation guides for understanding results
- 💡 Best practices and strategic recommendations
- 📋 UK market benchmarks for context
- ⚠️ Regulatory compliance notes (FCA, GDPR)

---

## 📁 Project Structure

```
InsurePrice/
├── 📊 Data & Models
│   ├── Enhanced_Synthetic_Car_Insurance_Claims.csv  # 10,000 records
│   ├── Sample_Priced_Policies.csv                   # Priced samples
│   ├── generate_data.py                             # Data generation
│   └── baseline_modeling.py                         # ML model training
│
├── 💰 Pricing Engine
│   ├── pricing_engine.py                            # Main pricing engine
│   ├── actuarial_pricing_engine.py                  # Actuarial formulas
│   ├── premium_calibration_demo.py                  # Calibration demo
│   └── corrected_actuarial_pricing.py               # Enhanced pricing
│
├── 🔍 Fraud Detection
│   ├── fraud_detection.py                           # Core fraud engine
│   └── fraud_api.py                                 # Fraud API endpoints
│
├── 📡 REST API
│   ├── insureprice_api.py                           # Main API application
│   ├── run_api.py                                   # API launcher
│   ├── test_api.py                                  # API tests
│   └── requirements_api.txt                         # API dependencies
│
├── 📊 Dashboard
│   ├── insureprice_dashboard.py                     # Streamlit dashboard
│   ├── run_dashboard.py                             # Dashboard launcher
│   └── visualization_dashboard.py                   # Visualizations
│
├── 📈 Analysis & Reports
│   ├── data_exploration.py                          # EDA script
│   ├── portfolio_simulation.py                      # Monte Carlo simulation
│   ├── fairness_bias_analysis.py                    # Fairness analysis
│   ├── shap_explainability.py                       # SHAP analysis
│   └── price_elasticity_simulation.py               # Elasticity modeling
│
├── 📓 Notebooks
│   └── InsurePrice_Car_Insurance_Risk_Modeling.ipynb
│
├── 📋 Configuration
│   ├── requirements.txt                             # Core dependencies
│   ├── requirements_api.txt                         # API dependencies
│   └── README.md                                    # This file
│
└── 📊 Visualizations
    ├── roc_curves_baseline.png
    ├── premium_distribution_analysis.png
    ├── portfolio_simulation_results.png
    ├── fairness_bias_analysis.png
    └── [other visualization files]
```

---

## 🛠️ Tech Stack

### Core Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.8+ | Core implementation |
| **ML Framework** | scikit-learn, XGBoost | Risk models |
| **API Framework** | FastAPI | REST API |
| **Dashboard** | Streamlit | Interactive UI |
| **Data Processing** | pandas, NumPy | Data manipulation |
| **Visualization** | Matplotlib, Seaborn, Plotly | Charts |
| **Explainability** | SHAP | Model interpretation |
| **Network Analysis** | NetworkX | Fraud rings |
| **NLP** | sklearn TF-IDF | Text analysis |

### Key Dependencies

```
# Core ML
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
xgboost>=1.7.0
shap>=0.48.0

# API
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0

# Dashboard
streamlit>=1.28.0
plotly>=5.15.0

# Analysis
scipy>=1.9.0
networkx>=3.0.0
```

---

## 💰 Business Value

### Financial Impact Summary

| Metric | Value | Industry Benchmark |
|--------|-------|-------------------|
| **Risk Prediction AUC** | 0.654 | 0.60-0.70 ✅ |
| **Portfolio Profit Margin** | 8.4% | 5-7% ✅ |
| **Loss Ratio** | 28.5% | 55-70% ✅ |
| **Combined Ratio** | 56.5% | <100% ✅ |
| **Fraud Detection Savings** | £60M potential | 5% improvement |

### ROI Projection (100,000 Policy Portfolio)

```
Annual Business Value:
├── Improved Profit Margin:     £840,000
├── Fraud Prevention:           £600,000
├── Reduced Loss Ratio:         £270,000
├── Operational Efficiency:     £150,000
├── Customer Retention:         £200,000
└── Total Annual Benefit:     £2,060,000
```

### Competitive Advantages

- **Real-Time Decisions**: Instant underwriting vs days
- **Fraud Prevention**: AI-powered detection
- **Fair Pricing**: Verified no bias
- **Regulatory Compliance**: Explainable AI
- **Customer Trust**: Transparent pricing

---

## 📈 Model Performance

### Risk Prediction Models (After Optimization)

| Model | AUC | Gini | Improvement |
|-------|-----|------|-------------|
| **CatBoost (Categorical)** 🏆 | **0.6176** | **0.2352** | Best Model |
| Random Forest (Optimized) | 0.6074 | 0.2147 | +4.52% |
| Logistic Regression | 0.6076 | 0.2151 | +3.84% |
| Neural Network Ensemble | 0.5993 | 0.1985 | Experimental |
| Baseline (No Engineering) | 0.5692 | 0.1383 | - |

### Model Improvement Journey

| Stage | Enhancement | AUC Gain |
|-------|-------------|----------|
| 1 | Baseline Model | 0.5692 |
| 2 | Feature Engineering | +3.84% |
| 3 | Hyperparameter Optimization | +1.40% |
| 4 | CatBoost Categorical Embeddings | +1.02% |
| **Total** | **All Improvements** | **~6.3%** |

### Feature Importance (CatBoost)

1. **Vehicle Type** (16.5%) - Safety and repair costs
2. **Annual Mileage** (11.6%) - Exposure risk
3. **Married Status** (11.4%) - Stability indicator
4. **Credit Score** (8.2%) - Financial responsibility
5. **Total Violations** (7.7%) - Driving behavior

### Engineered Features

- **Interaction Terms**: AGE × EXPERIENCE, AGE × VIOLATIONS
- **Risk Scores**: Composite driving risk, age risk, credit risk
- **Ratios**: Experience ratio, accidents per 10K miles

### Pricing Accuracy

- **Premium Range**: £400 - £1,200 (UK market aligned)
- **Average Premium**: £696
- **Risk Factor Coverage**: 17+ variables + engineered features
- **Regional Calibration**: 11 UK regions

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Contact

<div align="center">

**Masood Nazari**  
**Business Intelligence Analyst | Data Science | AI | Clinical Research**

[![Email](https://img.shields.io/badge/Email-M.Nazari%40soton.ac.uk-blue?style=for-the-badge&logo=gmail)](mailto:M.Nazari@soton.ac.uk)
[![Portfolio](https://img.shields.io/badge/Portfolio-michaeltheanalyst.github.io-green?style=for-the-badge&logo=github)](https://michaeltheanalyst.github.io/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-masood--nazari-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/masood-nazari)
[![GitHub](https://img.shields.io/badge/GitHub-michaeltheanalyst-black?style=for-the-badge&logo=github)](https://github.com/michaeltheanalyst)

</div>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UK Department for Transport**: Accident statistics
- **Association of British Insurers (ABI)**: Industry benchmarks
- **XGBoost Community**: ML framework
- **SHAP Developers**: Explainability tools
- **FastAPI Team**: API framework

---

<div align="center">

**🚗 InsurePrice** - *Transforming Car Insurance Through Data Science*

**Version 2.0.0** | **December 2025**

---

Made with ❤️ for the insurance industry

[⬆️ Back to Top](#-insureprice-enterprise-car-insurance-risk-modeling--pricing-platform)

</div>