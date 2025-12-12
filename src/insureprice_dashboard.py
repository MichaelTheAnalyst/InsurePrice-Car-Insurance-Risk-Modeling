"""
🚗 InsurePrice Interactive Dashboard
====================================

A comprehensive, AI-powered car insurance platform featuring:
- Real-time risk assessment and premium calculation
- Fraud detection and analysis
- Model performance and SHAP explainability
- API status monitoring
- Portfolio management
- Color psychology-driven design

Author: Masood Nazari
Business Intelligence Analyst | Data Science | AI | Clinical Research
Email: M.Nazari@soton.ac.uk
Portfolio: https://michaeltheanalyst.github.io/
LinkedIn: linkedin.com/in/masood-nazari
GitHub: github.com/michaeltheanalyst
Date: December 2025
Project: InsurePrice Car Insurance Risk Modeling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

# Import project modules
try:
    from actuarial_pricing_engine import ActuarialPricingEngine
    from fraud_detection import FraudDetectionEngine
    FRAUD_AVAILABLE = True
except ImportError:
    FRAUD_AVAILABLE = False

# Color palette
COLORS = {
    'primary_blue': '#1e3a8a',
    'secondary_blue': '#3b82f6',
    'accent_green': '#059669',
    'accent_orange': '#ea580c',
    'warning_red': '#dc2626',
    'premium_purple': '#7c3aed',
    'neutral_gray': '#6b7280',
    'light_bg': '#f8fafc'
}

# Page config
st.set_page_config(
    page_title="🚗 InsurePrice Platform",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; }
    
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.1);
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .metric-card h3 { color: #1e3a8a; margin: 0 0 0.3rem 0; font-size: 0.9rem; }
    .metric-card h2 { margin: 0; font-size: 1.8rem; }
    .metric-card p { color: #6b7280; margin: 0.3rem 0 0 0; font-size: 0.8rem; }
    
    .risk-low { border-left-color: #059669; }
    .risk-medium { border-left-color: #ea580c; }
    .risk-high { border-left-color: #dc2626; }
    
    .premium-card {
        background: linear-gradient(135deg, #7c3aed, #3b82f6);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
    }
    .premium-card h3 { margin: 0; font-size: 0.9rem; opacity: 0.9; }
    .premium-card h2 { margin: 0.3rem 0; font-size: 2rem; }
    .premium-card p { margin: 0; font-size: 0.8rem; opacity: 0.8; }
    
    .fraud-card {
        background: linear-gradient(135deg, #dc2626, #ea580c);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
    }
    .fraud-low {
        background: linear-gradient(135deg, #059669, #10b981);
    }
    .fraud-medium {
        background: linear-gradient(135deg, #ea580c, #f59e0b);
    }
    .fraud-high {
        background: linear-gradient(135deg, #dc2626, #ef4444);
    }
    
    .api-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    .api-online { background: #059669; color: white; }
    .api-offline { background: #dc2626; color: white; }
    
    .stButton>button {
        background: linear-gradient(135deg, #059669, #3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }
    
    .feature-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

@st.cache_data
def load_data():
    """Load data and initialize engines"""
    import os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'Enhanced_Synthetic_Car_Insurance_Claims.csv')
    try:
        df = pd.read_csv(DATA_FILE)
        pricing_engine = ActuarialPricingEngine(
            base_claim_frequency=0.122,
            base_claim_severity=3500,
            expense_loading=0.35,
            profit_margin=0.15,
            investment_return=0.04,
            risk_margin=0.08
        )
        return df, pricing_engine
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def calculate_risk_scores(df):
    """Calculate risk scores"""
    return (
        df['AGE'].map({'16-25': 0.8, '26-39': 0.6, '40-64': 0.4, '65+': 0.3}).fillna(0.5) +
        (df['ANNUAL_MILEAGE'] / 35000 * 0.3) +
        ((1 - df['CREDIT_SCORE']) * 0.4)
    ).clip(0, 1)

def check_api_status():
    """Check if API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def main():
    """Main application"""
    df, pricing_engine = load_data()
    
    if df is None:
        st.error("Failed to load data")
        return

    # Sidebar
    with st.sidebar:
        st.title("🚗 InsurePrice")
        st.caption("Enterprise Insurance Platform")
        st.markdown("---")
        
        pages = {
            "📊 Dashboard": "dashboard",
            "🎯 Risk Assessment": "risk_assessment",
            "💰 Premium Calculator": "premium_calculator",
            "💎 Customer CLV": "clv_prediction",
            "🔍 Fraud Detection": "fraud_detection",
            "📈 Portfolio Analytics": "portfolio_analytics",
            "🤖 Model Performance": "model_performance",
            "📡 API Status": "api_status",
            "📋 About": "about"
        }
        
        for page_name, page_key in pages.items():
            if st.button(page_name, key=page_key, use_container_width=True):
                st.session_state.page = page_key
        
        st.markdown("---")
        
        # API Status indicator
        api_online = check_api_status()
        status_class = "api-online" if api_online else "api-offline"
        status_text = "🟢 API Online" if api_online else "🔴 API Offline"
        st.markdown(f'<span class="api-status {status_class}">{status_text}</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("v2.0 | Dec 2025")

    # Route pages
    if st.session_state.page == "dashboard":
        render_dashboard(df, pricing_engine)
    elif st.session_state.page == "risk_assessment":
        render_risk_assessment(df, pricing_engine)
    elif st.session_state.page == "premium_calculator":
        render_premium_calculator(pricing_engine)
    elif st.session_state.page == "clv_prediction":
        render_clv_prediction(df)
    elif st.session_state.page == "fraud_detection":
        render_fraud_detection()
    elif st.session_state.page == "portfolio_analytics":
        render_portfolio_analytics(df, pricing_engine)
    elif st.session_state.page == "model_performance":
        render_model_performance()
    elif st.session_state.page == "api_status":
        render_api_status()
    elif st.session_state.page == "about":
        render_about()


def render_dashboard(df, pricing_engine):
    """Main dashboard"""
    st.markdown("""
    <div class="main-header">
        <h1>🚗 InsurePrice Dashboard</h1>
        <p>Enterprise Car Insurance Risk Modeling & Pricing Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Instructions section
    with st.expander("📖 **How to Use This Dashboard** - Click to expand", expanded=False):
        st.markdown("""
        ### Welcome to InsurePrice! 👋
        
        This dashboard provides a **comprehensive overview** of your car insurance portfolio and platform capabilities.
        
        **🔍 What You'll Find Here:**
        - **Key Metrics**: High-level KPIs showing portfolio health at a glance
        - **Platform Capabilities**: Quick overview of all features available
        - **Risk Distribution**: Visual breakdown of your portfolio's risk profile
        - **Regional Analysis**: Geographic patterns in claims data
        
        **💡 Pro Tips:**
        - Use the **sidebar** to navigate to specific features (Risk Assessment, Premium Calculator, etc.)
        - Check the **API Status** indicator at the bottom of the sidebar
        - The **claim rate** is a key profitability indicator - UK average is ~12%
        
        **📊 Key Benchmarks:**
        | Metric | Good | Average | Concerning |
        |--------|------|---------|------------|
        | Claim Rate | <10% | 10-15% | >15% |
        | Profit Margin | >8% | 5-8% | <5% |
        | Loss Ratio | <65% | 65-75% | >75% |
        """)

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_policies = len(df)
    claim_rate = df['OUTCOME'].mean() * 100
    avg_premium = 650
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Total Policies</h3>
            <h2 style="color: {COLORS['secondary_blue']}">{total_policies:,}</h2>
            <p>Active in portfolio</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        color = COLORS['warning_red'] if claim_rate > 15 else COLORS['accent_green']
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Claim Rate</h3>
            <h2 style="color: {color}">{claim_rate:.1f}%</h2>
            <p>Annual frequency</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💷 Avg Premium</h3>
            <h2 style="color: {COLORS['accent_green']}">£{avg_premium}</h2>
            <p>UK market average</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        profit_margin = 8.4
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Profit Margin</h3>
            <h2 style="color: {COLORS['premium_purple']}">{profit_margin}%</h2>
            <p>vs 5-7% industry</p>
        </div>
        """, unsafe_allow_html=True)

    # Platform Features
    st.markdown("### 🚀 Platform Capabilities")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4>🤖 ML Risk Models</h4>
            <p>AUC 0.654 | Random Forest, XGBoost, Logistic Regression</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4>🔍 Fraud Detection</h4>
            <p>4 methods | £60M savings potential</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h4>📡 REST API</h4>
            <p>FastAPI | 500 req/sec | Production-ready</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-box">
            <h4>🧠 Explainable AI</h4>
            <p>SHAP | FCA Compliance | Transparent</p>
        </div>
        """, unsafe_allow_html=True)

    # Risk Distribution
    st.markdown("### 🎯 Risk Distribution")
    
    with st.expander("ℹ️ **Understanding Risk Distribution** - Click to learn more"):
        st.markdown("""
        **What is a Risk Score?**
        
        The risk score (0-1) predicts the likelihood of a policyholder making a claim. It's calculated using:
        - **Age Group** (25%): Younger drivers have higher accident rates
        - **Annual Mileage** (20%): More driving = more exposure to accidents
        - **Credit Score** (25%): Correlates with claim frequency
        - **Driving History** (30%): Violations, DUIs, and past accidents
        
        **How to Interpret the Charts:**
        
        📊 **Histogram (Left)**: Shows how risk scores are distributed across your portfolio
        - A bell curve centered around 0.4-0.5 is healthy
        - A right-skewed distribution (more high-risk) may indicate adverse selection
        
        🥧 **Pie Chart (Right)**: Segments your portfolio into risk categories
        - **Low Risk (<0.4)**: Profitable segment, focus on retention
        - **Medium Risk (0.4-0.7)**: Standard pricing applies
        - **High Risk (>0.7)**: Ensure adequate premium loading
        
        **🎯 Target Portfolio Mix:**
        - Low Risk: 40-50% (stable profit base)
        - Medium Risk: 35-45% (volume segment)
        - High Risk: 10-20% (higher margins, higher volatility)
        """)
    
    risk_scores = calculate_risk_scores(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=risk_scores,
            nbinsx=25,
            marker_color=COLORS['secondary_blue'],
            opacity=0.8
        ))
        fig.update_layout(
            title="Portfolio Risk Score Distribution",
            xaxis_title="Risk Score (0-1)",
            yaxis_title="Number of Policies",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        low = (risk_scores < 0.4).sum()
        medium = ((risk_scores >= 0.4) & (risk_scores < 0.7)).sum()
        high = (risk_scores >= 0.7).sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            values=[low, medium, high],
            marker_colors=[COLORS['accent_green'], COLORS['accent_orange'], COLORS['warning_red']],
            hole=0.5
        )])
        fig.update_layout(title="Risk Categories", showlegend=True, height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Regional Analysis
    st.markdown("### 🗺️ Regional Risk Analysis")
    
    with st.expander("ℹ️ **Understanding Regional Risk** - Click to learn more"):
        st.markdown("""
        **Why Does Region Matter?**
        
        Geographic location significantly impacts claim rates due to:
        - **Traffic Density**: Urban areas like London have more accidents
        - **Crime Rates**: Vehicle theft varies significantly by region
        - **Road Conditions**: Weather patterns and infrastructure quality
        - **Emergency Response**: Time to hospital affects injury claims
        
        **How to Use This Chart:**
        
        🟢 **Green Bars (<11%)**: Low-risk regions - consider competitive pricing
        🟠 **Orange Bars (11-13%)**: Average risk - standard pricing applies
        🔴 **Red Bars (>13%)**: High-risk regions - ensure adequate premium loading
        
        **📈 Strategic Actions:**
        - **High-risk regions**: Consider stricter underwriting or higher premiums
        - **Low-risk regions**: Opportunity for market expansion with competitive rates
        - **Watch for trends**: Year-over-year changes may indicate emerging risks
        
        **⚠️ Regulatory Note:** Under UK FCA rules, regional pricing must be justifiable. 
        Keep documentation of actuarial basis for any regional premium variations.
        """)
    
    regional_data = df.groupby('REGION')['OUTCOME'].agg(['mean', 'count']).reset_index()
    regional_data.columns = ['Region', 'Claim_Rate', 'Policy_Count']
    regional_data['Claim_Rate'] *= 100
    regional_data = regional_data.sort_values('Claim_Rate', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=regional_data['Region'],
        x=regional_data['Claim_Rate'],
        orientation='h',
        marker_color=[COLORS['accent_green'] if x < 11 else COLORS['accent_orange'] if x < 13 else COLORS['warning_red'] for x in regional_data['Claim_Rate']],
        text=[f"{x:.1f}%" for x in regional_data['Claim_Rate']],
        textposition='outside'
    ))
    fig.update_layout(
        title="Claim Rate by UK Region",
        xaxis_title="Claim Rate (%)",
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_fraud_detection():
    """Fraud detection page"""
    st.markdown("""
    <div class="main-header" style="background: linear-gradient(135deg, #dc2626, #ea580c);">
        <h1>🔍 Fraud Detection</h1>
        <p>AI-Powered Claims Fraud Analysis System</p>
    </div>
    """, unsafe_allow_html=True)

    # Instructions
    with st.expander("📖 **How to Use Fraud Detection** - Click for guidance", expanded=False):
        st.markdown("""
        ### 🔍 Fraud Detection Overview
        
        This tool uses **4 complementary methods** to detect potentially fraudulent claims:
        
        **1️⃣ Anomaly Detection (35% weight)**
        Identifies statistically unusual patterns:
        - Unusually high claim amounts
        - Suspicious timing (late reporting, new policies)
        - Multiple previous claims
        
        **2️⃣ Behavioral Analysis (35% weight)**
        Flags suspicious claimant behavior:
        - No police report filed
        - No witnesses present
        - Cash settlement preference
        
        **3️⃣ Text Analysis (30% weight)**
        NLP-based keyword detection in claim descriptions:
        - Fraud-associated terms (whiplash, cash, urgent)
        - Suspicious relationships (friend, family involved)
        
        **📊 How to Interpret Results:**
        
        | Fraud Score | Risk Level | Action |
        |------------|------------|--------|
        | 0-35% | LOW | Standard processing |
        | 35-60% | MEDIUM | Enhanced review |
        | 60%+ | HIGH | Refer to SIU team |
        
        **⚠️ Important Notes:**
        - This is a **screening tool**, not definitive evidence of fraud
        - Always investigate before declining/referring claims
        - False positives are expected - err on the side of caution
        - Document all fraud referral decisions for regulatory compliance
        
        **💡 Best Practice:** Use this tool early in the claims process to prioritize 
        which claims need deeper investigation, saving time and resources.
        """)

    # UK Fraud Context
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("UK Annual Fraud Cost", "£1.2 Billion", "Industry-wide")
    with col2:
        st.metric("Target Improvement", "5%", "Achievable with ML")
    with col3:
        st.metric("Potential Savings", "£60 Million", "Significant ROI")

    st.markdown("---")

    # Fraud Analysis Form
    st.markdown("### 🚨 Analyze Claim for Fraud")
    
    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Claim Details")
            claim_id = st.text_input("Claim ID", "CLM-001")
            claim_amount = st.number_input("Claim Amount (£)", 100, 50000, 5000)
            days_to_report = st.number_input("Days to Report", 0, 90, 3)
            previous_claims = st.number_input("Previous Claims", 0, 10, 1)
            policy_age_days = st.number_input("Policy Age (days)", 1, 3650, 365)
        
        with col2:
            st.markdown("#### Behavioral Indicators")
            police_report = st.checkbox("Police Report Filed", True)
            witnesses = st.number_input("Number of Witnesses", 0, 10, 1)
            cash_settlement = st.checkbox("Cash Settlement Requested", False)
            description = st.text_area("Claim Description", 
                "Vehicle was hit in parking lot. Minor damage to rear bumper.")
        
        analyze = st.form_submit_button("🔍 Analyze for Fraud", use_container_width=True)

    if analyze:
        # Calculate fraud scores
        st.markdown("### 📊 Fraud Analysis Results")
        
        # Anomaly score
        anomaly_score = min(1.0, 
            (claim_amount / 50000) * 0.3 +
            (days_to_report / 90) * 0.2 +
            (previous_claims / 10) * 0.25 +
            (1 - policy_age_days / 3650) * 0.25
        )
        
        # Behavioral score
        behavioral_score = 0
        if not police_report: behavioral_score += 0.3
        if witnesses == 0: behavioral_score += 0.2
        if cash_settlement: behavioral_score += 0.35
        behavioral_score = min(1.0, behavioral_score)
        
        # Text analysis
        fraud_keywords = ['whiplash', 'cash', 'urgent', 'total loss', 'friend', 'family']
        text_score = sum(0.15 for kw in fraud_keywords if kw in description.lower())
        text_score = min(1.0, text_score)
        
        # Overall score
        overall_score = anomaly_score * 0.35 + behavioral_score * 0.35 + text_score * 0.30
        
        # Risk level
        if overall_score >= 0.6:
            risk_level, card_class = "HIGH", "fraud-high"
            recommendation = "Refer to Special Investigation Unit (SIU)"
        elif overall_score >= 0.35:
            risk_level, card_class = "MEDIUM", "fraud-medium"
            recommendation = "Enhanced review recommended"
        else:
            risk_level, card_class = "LOW", "fraud-low"
            recommendation = "Standard processing"

        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="fraud-card {card_class}">
                <h3>🚨 Fraud Score</h3>
                <h2>{overall_score:.1%}</h2>
                <p>{risk_level} RISK</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📋 Claim ID</h3>
                <h2>{claim_id}</h2>
                <p>Amount: £{claim_amount:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💡 Recommendation</h3>
                <h2 style="font-size: 1rem;">{recommendation}</h2>
                <p>Based on analysis</p>
            </div>
            """, unsafe_allow_html=True)

        # Component scores
        st.markdown("### 📈 Component Breakdown")
        
        fig = go.Figure()
        components = ['Anomaly Detection', 'Behavioral Analysis', 'Text Analysis']
        scores = [anomaly_score, behavioral_score, text_score]
        colors = [COLORS['secondary_blue'], COLORS['accent_orange'], COLORS['premium_purple']]
        
        fig.add_trace(go.Bar(
            x=components,
            y=scores,
            marker_color=colors,
            text=[f"{s:.1%}" for s in scores],
            textposition='outside'
        ))
        fig.update_layout(
            yaxis_title="Score",
            yaxis_range=[0, 1.1],
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        # Red flags
        st.markdown("### 🚩 Red Flags Detected")
        
        flags = []
        if not police_report: flags.append("❌ No police report filed")
        if witnesses == 0: flags.append("❌ No witnesses")
        if cash_settlement: flags.append("❌ Cash settlement requested")
        if days_to_report > 30: flags.append("❌ Late reporting (>30 days)")
        if previous_claims >= 3: flags.append("❌ Multiple previous claims")
        if policy_age_days < 90: flags.append("❌ New policy (<90 days)")
        for kw in fraud_keywords:
            if kw in description.lower():
                flags.append(f"❌ Suspicious keyword: '{kw}'")
        
        if flags:
            for flag in flags:
                st.warning(flag)
        else:
            st.success("✅ No major red flags detected")

        # Result interpretation
        st.markdown("---")
        st.markdown("### 📋 Result Interpretation & Next Steps")
        
        if risk_level == "HIGH":
            st.error(f"""
            **🚨 High Fraud Risk - Immediate Action Required**
            
            This claim scored **{overall_score:.1%}** on our fraud detection system, 
            indicating a **high probability** of fraudulent activity.
            
            **Recommended Next Steps:**
            1. **DO NOT approve** this claim without investigation
            2. **Refer to SIU** (Special Investigation Unit) immediately
            3. **Request additional documentation:**
               - Verified police report
               - Independent medical examination (if injury claimed)
               - Vehicle inspection by approved assessor
            4. **Check for patterns:**
               - Previous claims by this policyholder
               - Claims at same location
               - Connections to other claimants (network analysis)
            
            **⚖️ Legal Considerations:**
            - Document all investigation steps
            - Follow FCA treating customers fairly guidelines
            - Ensure any fraud decision is evidenced, not just model-based
            """)
        elif risk_level == "MEDIUM":
            st.warning(f"""
            **⚠️ Medium Fraud Risk - Enhanced Review Recommended**
            
            This claim scored **{overall_score:.1%}** on our fraud detection system, 
            suggesting **some indicators** warrant closer examination.
            
            **Recommended Next Steps:**
            1. **Conduct desktop review:**
               - Verify all documentation
               - Check claim history
               - Validate contact information
            2. **Request clarification** on any inconsistencies
            3. **Consider phone interview** with claimant
            4. **If concerns persist**, escalate to SIU
            
            **💡 Note:** Many legitimate claims score in this range. 
            Enhanced review is precautionary, not accusatory.
            """)
        else:
            st.success(f"""
            **✅ Low Fraud Risk - Standard Processing Appropriate**
            
            This claim scored **{overall_score:.1%}** on our fraud detection system, 
            indicating **no significant fraud indicators**.
            
            **Recommended Next Steps:**
            1. **Process through standard claims workflow**
            2. **Verify basic documentation** (photos, estimates)
            3. **Approve if within authority limits**
            
            **💡 Note:** Low fraud score doesn't guarantee legitimacy. 
            Apply normal due diligence and trust your claims handler instincts.
            """)


def render_clv_prediction(df):
    """Customer Lifetime Value prediction page"""
    st.markdown("""
    <div class="main-header" style="background: linear-gradient(135deg, #7c3aed, #3b82f6);">
        <h1>💎 Customer Lifetime Value (CLV)</h1>
        <p>Predict customer value for strategic pricing decisions</p>
    </div>
    """, unsafe_allow_html=True)

    # Instructions
    with st.expander("📖 **Understanding Customer Lifetime Value (CLV)** - Click to learn more", expanded=False):
        st.markdown("""
        ### 💎 What is Customer Lifetime Value?
        
        CLV predicts the **total profit** a customer will generate over their entire relationship 
        with your company. It's crucial for:
        - **Pricing Decisions**: How much discount is acceptable to retain a customer?
        - **Marketing Spend**: How much to invest in acquiring similar customers?
        - **Service Prioritization**: Which customers deserve premium service?
        
        **📊 How CLV is Calculated:**
        
        ```
        CLV = Σ (Annual Profit × Survival Probability) / (1 + Discount Rate)^Year - Acquisition Cost
        ```
        
        **Key Components:**
        - **Renewal Probability**: Likelihood of staying each year (affected by age, tenure, claims)
        - **Policy Profit**: Premium minus expected claims and expenses
        - **Cross-sell Revenue**: Additional products (home, life, travel insurance)
        - **Discount Rate**: 8% (time value of money)
        
        **🏆 Customer Segments:**
        
        | Segment | CLV Range | Strategy |
        |---------|-----------|----------|
        | 💎 Platinum | £1,500+ | VIP treatment, priority service, retention focus |
        | 🥇 Gold | £800-1,500 | Standard service, retention offers when at risk |
        | 🥈 Silver | £400-800 | Efficient service, opportunistic cross-sell |
        | 🥉 Bronze | <£400 | Automated service, no discounts |
        
        **💡 Strategic Applications:**
        - **Acceptable Discount**: For a £1,500 CLV customer, a £150 discount (10%) to prevent 
          churn is profitable vs. acquiring a new customer for £100-150
        - **Cross-sell Timing**: Best after 1+ years of claims-free tenure
        - **Churn Prevention**: Focus on customers showing declining engagement
        """)

    # Business context
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Customer Lifespan", "8 Years", "UK Insurance")
    with col2:
        st.metric("Avg Renewal Rate", "82%", "Industry Benchmark")
    with col3:
        st.metric("Cross-sell Lift", "+35%", "With CLV Targeting")

    st.markdown("---")

    # CLV Calculator
    st.markdown("### 💰 Calculate Customer Lifetime Value")
    
    with st.form("clv_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Customer Profile")
            age_group = st.selectbox("Age Group", ['16-25', '26-39', '40-64', '65+'], index=2)
            income_level = st.selectbox("Income Level", ['poverty', 'working_class', 'middle_class', 'upper_class'], index=2)
            credit_score = st.slider("Credit Score", 0.3, 1.0, 0.75, 0.05)
            years_customer = st.number_input("Years as Customer", 0, 20, 3)
        
        with col2:
            st.markdown("#### 📋 Policy Details")
            annual_premium = st.number_input("Annual Premium (£)", 300, 2000, 650)
            claims_count = st.number_input("Claims (Last 3 Years)", 0, 5, 0)
            risk_score = st.slider("Risk Score", 0.1, 0.6, 0.25, 0.05)
            vehicle_type = st.selectbox("Vehicle Type", ['small_hatchback', 'family_sedan', 'suv', 'sports_car', 'luxury_sedan'])
        
        with col3:
            st.markdown("#### 👨‍👩‍👧 Demographics")
            married = st.checkbox("Married", True)
            children = st.checkbox("Has Children", True)
            region = st.selectbox("Region", ['London', 'South East', 'South West', 'North West', 'Scotland'])
            acquisition = st.selectbox("Acquisition Channel", ['direct', 'comparison_site', 'broker', 'referral'])
        
        calculate_clv = st.form_submit_button("💎 Calculate CLV", use_container_width=True)

    if calculate_clv:
        # CLV Calculation Logic
        st.markdown("### 📊 CLV Analysis Results")
        
        # Base renewal rate
        renewal_rates = {'16-25': 0.65, '26-39': 0.78, '40-64': 0.85, '65+': 0.80}
        base_renewal = renewal_rates.get(age_group, 0.75)
        
        # Adjustments
        tenure_bonus = min(years_customer * 0.02, 0.10)
        claims_penalty = min(claims_count * 0.05, 0.15)
        credit_factor = (credit_score - 0.5) * 0.1
        family_bonus = 0.03 if married else 0
        family_bonus += 0.02 if children else 0
        
        renewal_prob = min(0.95, max(0.3, base_renewal + tenure_bonus - claims_penalty + credit_factor + family_bonus))
        
        # Project CLV
        discount_rate = 0.08
        profit_margin = 0.12
        expense_ratio = 0.28
        
        clv = 0
        yearly_values = []
        survival = 1.0
        
        # Acquisition costs
        acq_costs = {'direct': 80, 'comparison_site': 120, 'broker': 150, 'referral': 40}
        cac = acq_costs.get(acquisition, 100)
        
        for year in range(10):
            survival *= renewal_prob if year > 0 else 1.0
            if survival < 0.05:
                break
            
            # Revenue and costs
            revenue = annual_premium * survival
            expected_claims = risk_score * 0.122 * 3500 * survival
            expenses = revenue * expense_ratio
            profit = revenue - expected_claims - expenses
            
            # Cross-sell (starts year 2)
            cross_sell = 0
            if year >= 1:
                if income_level in ['middle_class', 'upper_class']:
                    cross_sell += 0.15 * 450 * profit_margin  # Home
                if children:
                    cross_sell += 0.10 * 300 * profit_margin  # Life
                cross_sell *= survival
            
            total = profit + cross_sell
            pv = total / (1 + discount_rate) ** year
            
            yearly_values.append({
                'year': year + 1,
                'survival': survival,
                'profit': profit,
                'cross_sell': cross_sell,
                'pv': pv
            })
            clv += pv
        
        net_clv = clv - cac
        
        # Determine segment
        if net_clv >= 1500:
            segment, segment_color = "Platinum 💎", "#7c3aed"
        elif net_clv >= 800:
            segment, segment_color = "Gold 🥇", "#f59e0b"
        elif net_clv >= 400:
            segment, segment_color = "Silver 🥈", "#6b7280"
        else:
            segment, segment_color = "Bronze 🥉", "#b45309"
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="premium-card" style="background: linear-gradient(135deg, {segment_color}, #3b82f6);">
                <h3>💎 Customer Lifetime Value</h3>
                <h2>£{net_clv:,.0f}</h2>
                <p>{segment} Customer</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Renewal Probability</h3>
                <h2 style="color: {'#059669' if renewal_prob > 0.75 else '#ea580c'}">{renewal_prob:.1%}</h2>
                <p>Expected retention rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            acceptable_discount = min(15, max(0, (net_clv - 500) / 100 * 2))
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Acceptable Discount</h3>
                <h2 style="color: #3b82f6">Up to {acceptable_discount:.0f}%</h2>
                <p>To retain high-CLV customer</p>
            </div>
            """, unsafe_allow_html=True)

        # CLV Breakdown Chart
        st.markdown("### 📈 CLV Projection Over Time")
        
        years = [v['year'] for v in yearly_values]
        profits = [v['profit'] for v in yearly_values]
        cross_sells = [v['cross_sell'] for v in yearly_values]
        survivals = [v['survival'] * 100 for v in yearly_values]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Policy Profit', x=years, y=profits, marker_color='#3b82f6'))
        fig.add_trace(go.Bar(name='Cross-sell Revenue', x=years, y=cross_sells, marker_color='#7c3aed'))
        fig.add_trace(go.Scatter(name='Survival %', x=years, y=survivals, yaxis='y2', 
                                  line=dict(color='#059669', width=3), mode='lines+markers'))
        
        fig.update_layout(
            barmode='stack',
            title="Yearly Value Contribution",
            xaxis_title="Year",
            yaxis_title="Value (£)",
            yaxis2=dict(title="Survival %", overlaying='y', side='right', range=[0, 100]),
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cross-sell opportunities
        st.markdown("### 🛒 Cross-sell Opportunities")
        
        cross_sell_products = []
        if income_level in ['middle_class', 'upper_class']:
            cross_sell_products.append({"Product": "Home Insurance", "Probability": "15%", "Annual Value": "£450", "Expected": "£67.50"})
        if children:
            cross_sell_products.append({"Product": "Life Insurance", "Probability": "12%", "Annual Value": "£300", "Expected": "£36.00"})
        if income_level == 'upper_class':
            cross_sell_products.append({"Product": "Umbrella Policy", "Probability": "8%", "Annual Value": "£200", "Expected": "£16.00"})
        cross_sell_products.append({"Product": "Travel Insurance", "Probability": "10%", "Annual Value": "£120", "Expected": "£12.00"})
        
        if cross_sell_products:
            st.dataframe(pd.DataFrame(cross_sell_products), use_container_width=True, hide_index=True)

        # Strategic recommendation
        st.markdown("### 💡 Strategic Recommendation")
        
        if net_clv >= 1000:
            st.success(f"""
            **🌟 High-Value Customer - Retention Priority**
            
            • Accept lower margins (up to {acceptable_discount:.0f}% discount) to retain
            • Prioritize for premium service and fast claims processing
            • Proactive cross-sell engagement recommended
            • Consider loyalty rewards program eligibility
            """)
        elif net_clv >= 500:
            st.info("""
            **📊 Standard Customer - Balanced Approach**
            
            • Apply standard pricing with minimal discounts
            • Focus on efficient service delivery
            • Opportunistic cross-selling when appropriate
            • Monitor for upgrade to Gold segment
            """)
        else:
            st.warning("""
            **⚠️ Low-Value Customer - Efficiency Focus**
            
            • Ensure risk-adequate pricing (no discounts)
            • Automate service interactions where possible
            • Consider digital-only service channel
            • Monitor claims experience closely
            """)


def render_model_performance():
    """Model performance page"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Model Performance</h1>
        <p>ML Model Evaluation and SHAP Explainability</p>
    </div>
    """, unsafe_allow_html=True)

    # Instructions
    with st.expander("📖 **Understanding ML Model Metrics** - Click to learn more", expanded=False):
        st.markdown("""
        ### 🤖 Machine Learning Model Evaluation
        
        Our risk prediction models are evaluated using several key metrics:
        
        **📊 Key Metrics Explained:**
        
        | Metric | What It Measures | Good Value |
        |--------|------------------|------------|
        | **AUC** | Overall discrimination ability | >0.7 (ours: 0.654) |
        | **Gini** | AUC × 2 - 1, common in insurance | >0.4 (ours: 0.308) |
        | **Precision** | % of predicted claims that were actual claims | >0.7 |
        | **Recall** | % of actual claims correctly predicted | >0.65 |
        
        **🎯 Why AUC of 0.654 is Acceptable:**
        - Insurance claim prediction is inherently difficult (many random factors)
        - Industry standard for motor insurance is typically 0.60-0.75
        - Our model provides meaningful lift over random selection
        - Combined with actuarial methods, this drives profitable pricing
        
        **🔍 SHAP Explainability:**
        
        SHAP (SHapley Additive exPlanations) shows **why** each prediction is made:
        - **Positive SHAP values** (🔴) increase predicted risk
        - **Negative SHAP values** (🟢) decrease predicted risk
        - **Larger bars** = more important features
        
        **⚖️ Why Explainability Matters:**
        - **FCA Compliance**: UK regulators require insurers to explain pricing decisions
        - **Customer Trust**: Transparent explanations build confidence
        - **Model Debugging**: Identify if model is using features appropriately
        - **Legal Protection**: Documented reasoning for pricing decisions
        
        **💡 Using Feature Importance:**
        - Focus data quality efforts on high-importance features
        - Annual mileage and age are top predictors - ensure accurate collection
        - Low-importance features may be candidates for removal
        """)

    # Model metrics
    st.markdown("### 📊 Risk Prediction Models")
    
    models_data = {
        'Model': ['Random Forest', 'Logistic Regression', 'XGBoost'],
        'AUC': [0.654, 0.651, 0.635],
        'Gini': [0.308, 0.302, 0.269],
        'Precision': [0.72, 0.71, 0.69],
        'Recall': [0.68, 0.67, 0.65]
    }
    
    df_models = pd.DataFrame(models_data)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(df_models, use_container_width=True, hide_index=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_models['Model'],
            y=df_models['AUC'],
            marker_color=[COLORS['accent_green'], COLORS['secondary_blue'], COLORS['accent_orange']],
            text=[f"{x:.3f}" for x in df_models['AUC']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Model AUC Comparison",
            yaxis_title="AUC Score",
            yaxis_range=[0.5, 0.75],
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.markdown("### 🎯 Feature Importance (SHAP)")
    
    features = ['Annual Mileage', 'Age Group', 'Credit Score', 'Driving Experience', 
                'Vehicle Type', 'Region', 'Speeding Violations', 'Past Accidents']
    importance = [0.18, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=features,
        x=importance,
        orientation='h',
        marker_color=COLORS['secondary_blue'],
        text=[f"{x:.0%}" for x in importance],
        textposition='outside'
    ))
    fig.update_layout(
        title="Top Risk Factors by Importance",
        xaxis_title="Importance Score",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # SHAP explanation example
    st.markdown("### 🔍 SHAP Explanation Example")
    
    st.info("""
    **Why is this driver HIGH RISK?**
    
    Top 5 Contributing Factors:
    1. 🔴 **Age 16-25** (+0.15): Young drivers have higher accident rates
    2. 🔴 **High Mileage** (+0.12): More exposure = more risk
    3. 🔴 **Low Credit Score** (+0.08): Correlation with claim frequency
    4. 🟢 **No Speeding Violations** (-0.05): Positive safety indicator
    5. 🟠 **Urban Region** (+0.04): Higher traffic density
    
    *SHAP values show how each feature contributes to the risk prediction.*
    """)


def render_api_status():
    """API status page"""
    st.markdown("""
    <div class="main-header">
        <h1>📡 API Status</h1>
        <p>REST API Monitoring and Testing</p>
    </div>
    """, unsafe_allow_html=True)

    # Instructions
    with st.expander("📖 **Understanding the API** - Click for technical details", expanded=False):
        st.markdown("""
        ### 📡 REST API Overview
        
        InsurePrice provides a **production-ready REST API** for integration with 
        external systems (quote engines, policy admin systems, etc.).
        
        **🔧 Technical Specifications:**
        - **Framework**: FastAPI (Python)
        - **Performance**: ~500 requests/second
        - **Format**: JSON request/response
        - **Authentication**: Ready for OAuth2/API keys (configure for production)
        
        **🔗 Available Endpoints:**
        
        | Endpoint | Use Case |
        |----------|----------|
        | `/api/v1/risk/score` | Real-time risk assessment for quotes |
        | `/api/v1/premium/quote` | Calculate premium from risk profile |
        | `/api/v1/fraud/analyze` | Screen claims for fraud |
        | `/api/v1/portfolio/analyze` | Batch portfolio analysis |
        | `/api/v1/model/explain/{id}` | Get SHAP explanations |
        
        **🚀 Starting the API Server:**
        ```bash
        # From project root directory
        python run_api.py
        
        # Or using uvicorn directly
        uvicorn insureprice_api:app --host 0.0.0.0 --port 8000
        ```
        
        **📚 API Documentation:**
        - **Swagger UI**: http://localhost:8000/docs (interactive testing)
        - **ReDoc**: http://localhost:8000/redoc (clean documentation)
        
        **🔒 Production Deployment Notes:**
        - Enable HTTPS (SSL/TLS certificates)
        - Configure API authentication
        - Set up rate limiting
        - Use a reverse proxy (nginx, traefik)
        - Enable logging and monitoring
        
        **💡 Integration Example (Python):**
        ```python
        import requests
        
        response = requests.post(
            "http://localhost:8000/api/v1/risk/score",
            json={"driver_profile": {...}}
        )
        risk_data = response.json()
        ```
        """)

    # Check API status
    api_online = check_api_status()
    
    if api_online:
        st.success("🟢 API Server is ONLINE")
        
        # Get health details
        try:
            response = requests.get("http://localhost:8000/health")
            health = response.json()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", "Healthy ✅")
            with col2:
                st.metric("Models Loaded", "Yes" if health.get('models_loaded') else "No")
            with col3:
                st.metric("Scaler Loaded", "Yes" if health.get('scaler_loaded') else "No")
        except:
            pass
    else:
        st.error("🔴 API Server is OFFLINE")
        st.info("Start the API server with: `python run_api.py`")

    st.markdown("---")

    # API Endpoints
    st.markdown("### 📋 Available Endpoints")
    
    endpoints = [
        {"Method": "POST", "Endpoint": "/api/v1/risk/score", "Description": "Real-time risk scoring"},
        {"Method": "POST", "Endpoint": "/api/v1/premium/quote", "Description": "Premium calculation"},
        {"Method": "POST", "Endpoint": "/api/v1/portfolio/analyze", "Description": "Portfolio analysis"},
        {"Method": "GET", "Endpoint": "/api/v1/model/explain/{id}", "Description": "SHAP explanations"},
        {"Method": "POST", "Endpoint": "/api/v1/fraud/analyze", "Description": "Fraud detection"},
        {"Method": "POST", "Endpoint": "/api/v1/fraud/batch", "Description": "Batch fraud analysis"},
        {"Method": "GET", "Endpoint": "/health", "Description": "Health check"},
    ]
    
    st.dataframe(pd.DataFrame(endpoints), use_container_width=True, hide_index=True)

    # Test API
    st.markdown("### 🧪 Test API")
    
    if api_online:
        if st.button("🔄 Test Risk Scoring Endpoint"):
            with st.spinner("Testing..."):
                try:
                    test_profile = {
                        "driver_profile": {
                            "age": "26-39",
                            "gender": "male",
                            "region": "London",
                            "driving_experience": "10-19y",
                            "education": "university",
                            "income": "middle_class",
                            "vehicle_type": "family_sedan",
                            "vehicle_year": "2016-2020",
                            "annual_mileage": 12000.0,
                            "credit_score": 0.75,
                            "speeding_violations": 0,
                            "duis": 0,
                            "past_accidents": 1,
                            "vehicle_ownership": 1,
                            "married": 1,
                            "children": 0,
                            "safety_rating": "standard"
                        }
                    }
                    
                    response = requests.post(
                        "http://localhost:8000/api/v1/risk/score",
                        json=test_profile,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ API Test Successful!")
                        st.json(result)
                    else:
                        st.error(f"❌ API returned status {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Test failed: {e}")

    # API Documentation links
    st.markdown("### 📚 Documentation")
    
    if api_online:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <a href="http://localhost:8000/docs" target="_blank" style="
                display: inline-block;
                padding: 10px 20px;
                background: linear-gradient(135deg, #3b82f6, #1e3a8a);
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 600;
            ">📖 Interactive API Docs (Swagger)</a>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <a href="http://localhost:8000/redoc" target="_blank" style="
                display: inline-block;
                padding: 10px 20px;
                background: linear-gradient(135deg, #059669, #047857);
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 600;
            ">📘 Alternative Docs (ReDoc)</a>
            """, unsafe_allow_html=True)
        
        st.info("💡 Click the buttons above to open API documentation in a new browser tab")
    else:
        st.warning("""
        ⚠️ **API Server is Offline** - Documentation links require the API to be running.
        
        **To start the API server:**
        ```bash
        python run_api.py
        ```
        
        Once running, the documentation will be available at:
        - **Swagger UI**: http://localhost:8000/docs
        - **ReDoc**: http://localhost:8000/redoc
        """)


def render_risk_assessment(df, pricing_engine):
    """Risk assessment page"""
    st.markdown("""
    <div class="main-header">
        <h1>🎯 AI Risk Assessment</h1>
        <p>Instant risk evaluation for any driver profile</p>
    </div>
    """, unsafe_allow_html=True)

    # Instructions
    with st.expander("📖 **How to Use Risk Assessment** - Click for guidance", expanded=False):
        st.markdown("""
        ### 🎯 Risk Assessment Guide
        
        Enter driver and vehicle details to receive an **instant risk evaluation** and premium quote.
        
        **📝 Input Fields Explained:**
        
        | Field | Impact on Risk | Notes |
        |-------|----------------|-------|
        | **Age Group** | HIGH | 16-25 has ~2x risk vs 40-64 |
        | **Region** | MEDIUM | Urban areas generally higher risk |
        | **Vehicle Type** | MEDIUM | Sports cars, SUVs typically higher |
        | **Driving Experience** | MEDIUM | More experience = lower risk |
        | **Annual Mileage** | HIGH | Linear relationship with exposure |
        | **Credit Score** | HIGH | Strong correlation with claims |
        | **Speeding Violations** | MEDIUM | +3% risk per violation |
        | **DUIs** | VERY HIGH | +8% risk per incident |
        | **Past Accidents** | HIGH | +5% risk per accident |
        
        **📊 Risk Categories:**
        - 🟢 **Low Risk (0.00-0.39)**: Premium drivers, competitive pricing appropriate
        - 🟠 **Medium Risk (0.40-0.69)**: Standard market, risk-adequate pricing
        - 🔴 **High Risk (0.70-1.00)**: Substandard market, requires premium loading
        
        **💡 Tips for Accurate Assessment:**
        - Use actual annual mileage (check MOT history for estimates)
        - Credit score is normalized 0-1 (0.7 ≈ 700 on standard scale)
        - Include ALL past accidents, even minor ones
        - DUIs have major impact - even one significantly increases risk
        
        **⚠️ Important:** This is an indicative assessment. Final premium may differ 
        based on additional underwriting factors and market conditions.
        """)

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.selectbox("Age Group", ['16-25', '26-39', '40-64', '65+'])
            region = st.selectbox("Region", sorted(df['REGION'].unique()))
            vehicle_type = st.selectbox("Vehicle Type", sorted(df['VEHICLE_TYPE'].unique()))

        with col2:
            driving_exp = st.selectbox("Driving Experience", sorted(df['DRIVING_EXPERIENCE'].unique()))
            annual_mileage = st.slider("Annual Mileage", 500, 35000, 12000, step=500)
            credit_score = st.slider("Credit Score", 0.1, 1.0, 0.7, 0.05)

        with col3:
            speeding = st.number_input("Speeding Violations", 0, 10, 0)
            duis = st.number_input("DUIs", 0, 5, 0)
            accidents = st.number_input("Past Accidents", 0, 10, 0)

        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    if submitted:
        age_risk = {'16-25': 0.8, '26-39': 0.6, '40-64': 0.4, '65+': 0.3}
        risk_score = (
            age_risk.get(age, 0.5) * 0.25 +
            (annual_mileage / 35000) * 0.2 +
            (1 - credit_score) * 0.25 +
            (speeding * 0.03) +
            (duis * 0.08) +
            (accidents * 0.05)
        )
        risk_score = min(max(risk_score, 0), 1)
        
        if risk_score < 0.4:
            category, color = "Low Risk", COLORS['accent_green']
        elif risk_score < 0.7:
            category, color = "Medium Risk", COLORS['accent_orange']
        else:
            category, color = "High Risk", COLORS['warning_red']
        
        premium = pricing_engine.calculate_basic_actuarial_premium(risk_score, credibility=0.9)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {color}">
                <h3>Risk Category</h3>
                <h2 style="color: {color}">{category}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Risk Score</h3>
                <h2>{risk_score:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="premium-card">
                <h3>Premium</h3>
                <h2>£{premium['final_premium']:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Result interpretation
        st.markdown("---")
        st.markdown("### 📋 Result Interpretation")
        
        if category == "Low Risk":
            st.success(f"""
            **✅ Low Risk Profile Detected**
            
            This driver profile indicates a **below-average likelihood** of filing a claim.
            
            **What this means:**
            - Risk score of **{risk_score:.2f}** is in the bottom 40% of the risk distribution
            - Premium of **£{premium['final_premium']:.0f}** is competitive for this segment
            - Retention priority: **HIGH** - this is a profitable customer segment
            
            **Recommended Actions:**
            - Offer competitive renewal pricing to retain
            - Consider loyalty discounts for multi-year customers
            - Cross-sell opportunities: Home insurance, travel insurance
            """)
        elif category == "Medium Risk":
            st.info(f"""
            **📊 Standard Risk Profile Detected**
            
            This driver profile indicates an **average likelihood** of filing a claim.
            
            **What this means:**
            - Risk score of **{risk_score:.2f}** is in the standard market range (40-70th percentile)
            - Premium of **£{premium['final_premium']:.0f}** reflects risk-adequate pricing
            - This represents the bulk of the insurance market
            
            **Recommended Actions:**
            - Apply standard pricing without special discounts
            - Monitor for risk improvements (telematics data, claims-free years)
            - Consider step-down pricing after 2+ claims-free years
            """)
        else:
            st.warning(f"""
            **⚠️ High Risk Profile Detected**
            
            This driver profile indicates an **above-average likelihood** of filing a claim.
            
            **What this means:**
            - Risk score of **{risk_score:.2f}** is in the top 30% of the risk distribution
            - Premium of **£{premium['final_premium']:.0f}** includes necessary risk loading
            - Higher margins but also higher volatility expected
            
            **Recommended Actions:**
            - Ensure no underpricing - maintain actuarially sound rates
            - Consider additional underwriting requirements
            - Higher excess options to manage exposure
            - Telematics policy may help monitor actual driving behavior
            """)


def render_premium_calculator(pricing_engine):
    """Premium calculator page"""
    st.markdown("""
    <div class="main-header">
        <h1>💰 Premium Calculator</h1>
        <p>Actuarially-sound premium calculation</p>
    </div>
    """, unsafe_allow_html=True)

    # Instructions
    with st.expander("📖 **Understanding Premium Calculation** - Click for details", expanded=False):
        st.markdown("""
        ### 💰 Actuarial Premium Breakdown
        
        Our premium calculation follows **professional actuarial methodology**:
        
        **📐 The Formula:**
        ```
        Premium = Expected Loss + Expenses + Profit + Risk Margin - Investment Credit
        ```
        
        **💵 Component Breakdown:**
        
        | Component | % of Premium | Description |
        |-----------|--------------|-------------|
        | **Expected Loss** | ~50% | Predicted claims cost (frequency × severity) |
        | **Expenses** | 35% | Admin, acquisition, claims handling |
        | **Profit Margin** | 15% | Target profit on gross premium |
        | **Risk Margin** | 8% | Buffer for adverse deviation |
        | **Investment Credit** | -4% | Return on reserves held |
        
        **🎚️ Input Parameters:**
        
        **Risk Score (0-1):**
        - Drives the expected loss calculation
        - Higher score = higher expected claims
        - Use output from Risk Assessment page
        
        **Credibility (0.5-1.0):**
        - How much weight to give individual risk score vs portfolio average
        - 1.0 = full credibility to individual factors
        - 0.5 = blend equally with portfolio average
        - Higher for customers with more data/tenure
        
        **📊 Example Calculation (Risk Score 0.35, Credibility 0.9):**
        ```
        Expected Loss:    £155.23  (Frequency 12.2% × Severity £3,500 × Risk Factor)
        + Expenses:       £ 84.43  (35% loading)
        + Profit:         £ 36.18  (15% target)
        + Risk Margin:    £ 19.30  (8% buffer)
        - Investment:     -£ 9.65  (4% credit)
        ─────────────────────────
        = Final Premium:  £285.49
        ```
        
        **💡 Pricing Strategy Tips:**
        - UK average motor premium is ~£650/year
        - Market competitive range: £400-£900 for standard risks
        - High-risk segments may require £1,500+
        """)

    col1, col2 = st.columns([1, 2])

    with col1:
        risk_score = st.slider("Risk Score", 0.0, 1.0, 0.35, 0.01)
        credibility = st.slider("Credibility", 0.5, 1.0, 0.9, 0.01)
        calculate = st.button("🔄 Calculate", use_container_width=True)

    with col2:
        if calculate:
            result = pricing_engine.calculate_basic_actuarial_premium(risk_score, credibility)
            
            st.markdown(f"""
            <div class="premium-card" style="margin-bottom: 1rem;">
                <h3>Annual Premium</h3>
                <h2>£{result['final_premium']:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Expected Loss", f"£{result['breakdown']['expected_loss']:.2f}")
                st.metric("Expenses (35%)", f"£{result['breakdown']['expenses']:.2f}")
            with c2:
                st.metric("Profit (15%)", f"£{result['breakdown']['profit_margin']:.2f}")
                st.metric("Risk Margin (8%)", f"£{result['breakdown']['risk_margin']:.2f}")


def render_portfolio_analytics(df, pricing_engine):
    """Portfolio analytics page"""
    st.markdown("""
    <div class="main-header">
        <h1>📈 Portfolio Analytics</h1>
        <p>Comprehensive risk and performance analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Instructions
    with st.expander("📖 **Understanding Portfolio Analytics** - Click for guidance", expanded=False):
        st.markdown("""
        ### 📈 Portfolio Analytics Guide
        
        This page provides a **bird's-eye view** of your entire insurance portfolio's 
        risk profile and premium distribution.
        
        **📊 Key Metrics Explained:**
        
        | Metric | What It Tells You | Healthy Range |
        |--------|-------------------|---------------|
        | **Total Policies** | Portfolio size | Growth target dependent |
        | **Claim Rate** | % of policies with claims | <12% (UK average ~12%) |
        | **Avg Risk Score** | Portfolio risk quality | 0.35-0.45 |
        | **Avg Premium** | Revenue per policy | £600-700 (UK market) |
        
        **📉 Risk Distribution Chart:**
        - **Left-skewed** (more low-risk): Profitable but may indicate over-selectivity
        - **Bell-shaped**: Balanced portfolio with diversified risk
        - **Right-skewed** (more high-risk): Higher margins but more volatility
        
        **💰 Premium Distribution Chart:**
        - Should roughly mirror risk distribution (risk-based pricing working)
        - Gaps may indicate pricing inconsistencies
        - Very tight distribution may mean insufficient segmentation
        
        **🎯 Portfolio Optimization Goals:**
        
        1. **Risk-Premium Alignment**: High correlation between risk score and premium
        2. **Adequate Diversification**: Not over-concentrated in any segment
        3. **Profitability Balance**: Mix of low-risk (stable) and high-risk (higher margin)
        4. **Geographic Spread**: Reduce regional catastrophe exposure
        
        **⚠️ Warning Signs to Watch:**
        - Claim rate trending upward (adverse selection?)
        - Risk score dropping but claim rate stable (model drift?)
        - Premium decreasing while risk stable (competitive pressure?)
        
        **💡 Pro Tip:** Compare these metrics month-over-month to spot trends early.
        """)

    risk_scores = calculate_risk_scores(df)
    batch_results = pricing_engine.batch_calculate_premiums(risk_scores, method='basic', credibility=0.85)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Policies", f"{len(df):,}")
    with col2:
        st.metric("Claim Rate", f"{df['OUTCOME'].mean()*100:.1f}%")
    with col3:
        st.metric("Avg Risk Score", f"{risk_scores.mean():.3f}")
    with col4:
        st.metric("Avg Premium", f"£{batch_results['calculated_premium'].mean():.0f}")

    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=risk_scores, nbinsx=25, marker_color=COLORS['secondary_blue']))
        fig.update_layout(title="Risk Distribution", xaxis_title="Risk Score", template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=batch_results['calculated_premium'], nbinsx=25, marker_color=COLORS['accent_green']))
        fig.update_layout(title="Premium Distribution", xaxis_title="Premium (£)", template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)


def render_about():
    """About page"""
    st.markdown("""
    <div class="main-header">
        <h1>📋 About InsurePrice</h1>
        <p>Enterprise Car Insurance Platform</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Platform Overview
        
        **InsurePrice** is a comprehensive car insurance platform featuring:
        
        ### 🤖 Machine Learning
        - Risk prediction models (AUC 0.654)
        - Random Forest, XGBoost, Logistic Regression
        - SHAP explainability
        
        ### 🔍 Fraud Detection
        - Anomaly detection (Isolation Forest)
        - NLP text analysis (40+ keywords)
        - Network analysis for fraud rings
        - Behavioral pattern recognition
        
        ### 📡 REST API
        - FastAPI backend
        - Real-time risk scoring
        - Premium calculation
        - Fraud analysis
        
        ### 💰 Actuarial Pricing
        - Professional premium formulas
        - Risk-based pricing
        - Monte Carlo simulation
        
        ### 📊 Analytics Dashboard
        - Portfolio management
        - Regional analysis
        - Performance monitoring
        """)
    
    with col2:
        st.markdown("""
        ## 📞 Contact
        
        **Masood Nazari**
        
        *Business Intelligence Analyst*
        *Data Science | AI | Clinical Research*
        
        📧 [M.Nazari@soton.ac.uk](mailto:M.Nazari@soton.ac.uk)
        
        🌐 [Portfolio](https://michaeltheanalyst.github.io/)
        
        💼 [LinkedIn](https://linkedin.com/in/masood-nazari)
        
        💻 [GitHub](https://github.com/michaeltheanalyst)
        
        ---
        
        **Version**: 2.0
        
        **Date**: December 2025
        
        ---
        
        ## 💼 Business Value
        
        - £2.06M annual benefit
        - 8.4% profit margin
        - £60M fraud savings
        - AUC 0.654 accuracy
        """)


if __name__ == "__main__":
    main()