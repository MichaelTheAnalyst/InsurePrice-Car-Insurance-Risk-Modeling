"""
🚗 InsurePrice Interactive Dashboard
====================================

A modern, AI-powered car insurance risk modeling dashboard featuring:
- Real-time risk assessment and premium calculation
- Interactive portfolio management
- Advanced analytics and visualizations
- Color psychology-driven design for optimal user experience

Author: Masood Nazari (Creative Data Scientist & UI/UX Designer)
Business Intelligence Analyst | Data Science | AI | Clinical Research
Email: M.Nazari@soton.ac.uk
Portfolio: https://michaeltheanalyst.github.io/
LinkedIn: linkedin.com/in/masood-nazari
GitHub: github.com/michaeltheanalyst
Date: December 2025
Project: InsurePrice Car Insurance Risk Modeling

🎨 Design Philosophy - Color Psychology:
- Deep Blue (#1e3a8a): Trust, Security, Professionalism
- Ocean Blue (#3b82f6): Confidence, Reliability, Stability
- Emerald Green (#059669): Success, Growth, Financial Prosperity
- Sunset Orange (#ea580c): Energy, Action, Premium Features
- Crimson Red (#dc2626): Urgency, High Risk Alerts
- Royal Purple (#7c3aed): Luxury, Sophistication, Premium Services
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from actuarial_pricing_engine import ActuarialPricingEngine
import os

# Color palette constants
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

# Set page configuration
st.set_page_config(
    page_title="🚗 InsurePrice Dashboard",
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
    
    .stButton>button {
        background: linear-gradient(135deg, #059669, #3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }
    
    .legend-box {
        display: flex;
        justify-content: center;
        gap: 25px;
        padding: 12px;
        background: #f8fafc;
        border-radius: 8px;
        margin: 10px 0;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 13px;
        color: #374151;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

@st.cache_data
def load_data():
    """Load and prepare all necessary data"""
    try:
        df = pd.read_csv('Enhanced_Synthetic_Car_Insurance_Claims.csv')
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
    """Calculate risk scores for the dataset"""
    return (
        df['AGE'].map({'16-25': 0.8, '26-39': 0.6, '40-64': 0.4, '65+': 0.3}).fillna(0.5) +
        (df['ANNUAL_MILEAGE'] / 35000 * 0.3) +
        ((1 - df['CREDIT_SCORE']) * 0.4)
    ).clip(0, 1)

def main():
    """Main dashboard application"""
    df, pricing_engine = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check your files.")
        return

    # Sidebar navigation
    with st.sidebar:
        st.title("🚗 InsurePrice")
        st.caption("AI-Powered Risk Modeling")
        st.markdown("---")
        
        pages = {
            "📊 Dashboard": "dashboard",
            "🎯 Risk Assessment": "risk_assessment",
            "💰 Premium Calculator": "premium_calculator",
            "📈 Portfolio Analytics": "portfolio_analytics",
            "📋 About": "about"
        }
        
        for page_name, page_key in pages.items():
            if st.button(page_name, key=page_key, use_container_width=True):
                st.session_state.page = page_key
        
        st.markdown("---")
        st.caption("**Risk Legend**")
        st.markdown("🟢 Low Risk (<40%)")
        st.markdown("🟠 Medium Risk (40-70%)")
        st.markdown("🔴 High Risk (>70%)")

    # Route to pages
    if st.session_state.page == "dashboard":
        render_dashboard(df, pricing_engine)
    elif st.session_state.page == "risk_assessment":
        render_risk_assessment(df, pricing_engine)
    elif st.session_state.page == "premium_calculator":
        render_premium_calculator(pricing_engine)
    elif st.session_state.page == "portfolio_analytics":
        render_portfolio_analytics(df, pricing_engine)
    elif st.session_state.page == "about":
        render_about()

def render_dashboard(df, pricing_engine):
    """Main dashboard page"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 InsurePrice Dashboard</h1>
        <p>AI-Powered Car Insurance Risk Modeling & Premium Optimization</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics
    total_policies = len(df)
    claim_rate = df['OUTCOME'].mean() * 100
    avg_premium = 650
    portfolio_value = total_policies * avg_premium

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Total Policies</h3>
            <h2 style="color: {COLORS['secondary_blue']}">{total_policies:,}</h2>
            <p>Active policies in portfolio</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        color = COLORS['warning_red'] if claim_rate > 15 else COLORS['accent_green']
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Claim Rate</h3>
            <h2 style="color: {color}">{claim_rate:.1f}%</h2>
            <p>Annual claims frequency</p>
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
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Portfolio Value</h3>
            <h2 style="color: {COLORS['premium_purple']}">£{portfolio_value:,}</h2>
            <p>Annual revenue potential</p>
        </div>
        """, unsafe_allow_html=True)

    # Risk Distribution
    st.markdown("### 🎯 Risk Distribution")
    
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
            height=350,
            margin=dict(t=50, b=40)
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
            hole=0.5,
            textinfo='percent+label'
        )])
        fig.update_layout(
            title="Risk Categories",
            showlegend=False,
            height=350,
            margin=dict(t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Regional Analysis
    st.markdown("### 🗺️ Regional Risk Analysis")
    
    regional_data = df.groupby('REGION')['OUTCOME'].agg(['mean', 'count']).reset_index()
    regional_data.columns = ['Region', 'Claim_Rate', 'Policy_Count']
    regional_data['Claim_Rate'] *= 100
    regional_data = regional_data.sort_values('Claim_Rate', ascending=True)
    
    # Color coding
    def get_color(rate):
        if rate < 11: return COLORS['accent_green']
        elif rate < 13: return COLORS['accent_orange']
        return COLORS['warning_red']
    
    regional_data['Color'] = regional_data['Claim_Rate'].apply(get_color)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=regional_data['Region'],
            x=regional_data['Claim_Rate'],
            orientation='h',
            marker_color=regional_data['Color'],
            text=[f"{x:.1f}%" for x in regional_data['Claim_Rate']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Claim Rate by Region",
            xaxis_title="Claim Rate (%)",
            yaxis_title="",
            height=400,
            template="plotly_white",
            margin=dict(l=10, r=80, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=regional_data['Region'],
            x=regional_data['Policy_Count'],
            orientation='h',
            marker_color=COLORS['secondary_blue'],
            text=[f"{x:,}" for x in regional_data['Policy_Count']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Policies by Region",
            xaxis_title="Number of Policies",
            yaxis_title="",
            height=400,
            template="plotly_white",
            margin=dict(l=10, r=80, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    <div class="legend-box">
        <span class="legend-item"><span class="legend-dot" style="background:#059669"></span>Low Risk (&lt;11%)</span>
        <span class="legend-item"><span class="legend-dot" style="background:#ea580c"></span>Medium Risk (11-13%)</span>
        <span class="legend-item"><span class="legend-dot" style="background:#dc2626"></span>High Risk (&gt;13%)</span>
    </div>
    """, unsafe_allow_html=True)

def render_risk_assessment(df, pricing_engine):
    """Risk assessment page"""
    
    st.markdown("""
    <div class="main-header">
        <h1>🎯 AI Risk Assessment</h1>
        <p>Get instant risk evaluation for any driver profile</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("risk_form"):
        st.markdown("### 👤 Driver Information")
        
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
        # Calculate risk score
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
        
        # Determine category
        if risk_score < 0.4:
            category, color, icon = "Low Risk", COLORS['accent_green'], "🟢"
        elif risk_score < 0.7:
            category, color, icon = "Medium Risk", COLORS['accent_orange'], "🟠"
        else:
            category, color, icon = "High Risk", COLORS['warning_red'], "🔴"
        
        # Calculate premium
        premium = pricing_engine.calculate_basic_actuarial_premium(risk_score, credibility=0.9)
        
        st.markdown("### 📊 Assessment Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {color}">
                <h3>{icon} Risk Category</h3>
                <h2 style="color: {color}">{category}</h2>
                <p>Based on profile analysis</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Risk Score</h3>
                <h2 style="color: {COLORS['secondary_blue']}">{risk_score:.2f}</h2>
                <p>Scale: 0 (low) to 1 (high)</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="premium-card">
                <h3>💰 Recommended Premium</h3>
                <h2>£{premium['final_premium']:.0f}</h2>
                <p>Annual comprehensive</p>
            </div>
            """, unsafe_allow_html=True)

        # Risk factors radar
        st.markdown("### 🔍 Risk Factor Breakdown")
        
        factors = ['Age', 'Mileage', 'Credit', 'Violations', 'Accidents']
        values = [
            age_risk.get(age, 0.5),
            annual_mileage / 35000,
            1 - credit_score,
            min((speeding + duis * 2) / 10, 1),
            min(accidents / 5, 1)
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=factors + [factors[0]],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.3)',
            line_color=COLORS['secondary_blue']
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=400,
            margin=dict(t=30, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

def render_premium_calculator(pricing_engine):
    """Premium calculator page"""
    
    st.markdown("""
    <div class="main-header">
        <h1>💰 Premium Calculator</h1>
        <p>Calculate optimal premiums based on actuarial principles</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🧮 Parameters")
        risk_score = st.slider("Risk Score", 0.0, 1.0, 0.35, 0.01)
        credibility = st.slider("Credibility Factor", 0.5, 1.0, 0.9, 0.01)
        
        calculate = st.button("🔄 Calculate Premium", use_container_width=True)

    with col2:
        if calculate:
            result = pricing_engine.calculate_basic_actuarial_premium(risk_score, credibility)
            breakdown = result['breakdown']
            ratios = result['ratios']
            
            st.markdown("### 📊 Results")
            
            # Premium display
            st.markdown(f"""
            <div class="premium-card" style="margin-bottom: 1rem;">
                <h3>💷 Final Premium</h3>
                <h2>£{result['final_premium']:.2f}</h2>
                <p>Annual comprehensive coverage</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Breakdown metrics
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Expected Loss", f"£{breakdown['expected_loss']:.2f}")
                st.metric("Expenses (35%)", f"£{breakdown['expenses']:.2f}")
            with c2:
                st.metric("Profit Margin (15%)", f"£{breakdown['profit_margin']:.2f}")
                st.metric("Risk Margin (8%)", f"£{breakdown['risk_margin']:.2f}")
            
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Expected Loss', 'Expenses', 'Profit', 'Risk Margin'],
                values=[breakdown['expected_loss'], breakdown['expenses'], 
                       breakdown['profit_margin'], breakdown['risk_margin']],
                marker_colors=[COLORS['secondary_blue'], COLORS['accent_orange'], 
                              COLORS['accent_green'], COLORS['warning_red']],
                hole=0.4
            )])
            fig.update_layout(
                title="Premium Composition",
                height=350,
                margin=dict(t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Status
            status = "✅ Profitable" if ratios['combined_ratio'] < 1 else "❌ Loss-making"
            st.info(f"Combined Ratio: {ratios['combined_ratio']:.3f} — {status}")
        else:
            st.info("👆 Adjust parameters and click Calculate to see results")

def render_portfolio_analytics(df, pricing_engine):
    """Portfolio analytics page"""
    
    st.markdown("""
    <div class="main-header">
        <h1>📈 Portfolio Analytics</h1>
        <p>Comprehensive portfolio risk and performance analysis</p>
    </div>
    """, unsafe_allow_html=True)

    risk_scores = calculate_risk_scores(df)
    batch_results = pricing_engine.batch_calculate_premiums(risk_scores, method='basic', credibility=0.85)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Policies", f"{len(df):,}")
    with col2:
        st.metric("Claim Rate", f"{df['OUTCOME'].mean()*100:.1f}%")
    with col3:
        st.metric("Avg Risk Score", f"{risk_scores.mean():.3f}")
    with col4:
        st.metric("Avg Premium", f"£{batch_results['calculated_premium'].mean():.0f}")

    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Risk Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=risk_scores,
            nbinsx=25,
            marker_color=COLORS['secondary_blue'],
            opacity=0.8
        ))
        fig.update_layout(
            xaxis_title="Risk Score",
            yaxis_title="Count",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 💰 Premium Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=batch_results['calculated_premium'],
            nbinsx=25,
            marker_color=COLORS['accent_green'],
            opacity=0.8
        ))
        fig.update_layout(
            xaxis_title="Premium (£)",
            yaxis_title="Count",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    st.markdown("### 📈 Premium vs Risk Relationship")
    
    sample = pd.DataFrame({
        'risk': risk_scores,
        'premium': batch_results['calculated_premium']
    }).sample(min(1000, len(df)))
    
    fig = px.scatter(
        sample,
        x='risk',
        y='premium',
        color='risk',
        color_continuous_scale=['green', 'orange', 'red'],
        opacity=0.6
    )
    fig.update_layout(
        xaxis_title="Risk Score",
        yaxis_title="Premium (£)",
        template="plotly_white",
        height=400,
        coloraxis_colorbar_title="Risk"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_about():
    """About page"""
    
    st.markdown("""
    <div class="main-header">
        <h1>📋 About InsurePrice</h1>
        <p>AI-Powered Car Insurance Risk Modeling Platform</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Project Overview
        
        **InsurePrice** is a comprehensive car insurance risk modeling platform that leverages 
        advanced machine learning and actuarial science to provide accurate risk assessment 
        and premium optimization.
        
        ## 🧠 Key Features
        
        - **Risk Assessment**: Real-time driver risk evaluation
        - **Premium Calculation**: Actuarially-sound pricing
        - **Portfolio Analytics**: Risk and performance analysis
        - **Interactive Visualizations**: Modern dashboards
        
        ## 🎨 Design Philosophy
        
        Color psychology principles for optimal UX:
        - 🔵 **Blue**: Trust, Security, Professionalism
        - 🟢 **Green**: Success, Growth, Prosperity
        - 🟠 **Orange**: Energy, Action
        - 🔴 **Red**: Alerts, High Risk
        - 🟣 **Purple**: Premium, Sophistication
        
        ## 👨‍💻 Technical Stack
        
        - **Frontend**: Streamlit + Plotly
        - **Backend**: Python, scikit-learn, XGBoost
        - **Data**: UK Insurance Statistics (Synthetic)
        """)
    
    with col2:
        st.markdown("""
        ## 📞 Contact
        
        **Masood Nazari**
        
        *Business Intelligence Analyst*
        *Data Science | AI | Clinical Research*
        
        📧 M.Nazari@soton.ac.uk
        
        🌐 [Portfolio](https://michaeltheanalyst.github.io/)
        
        💼 [LinkedIn](https://linkedin.com/in/masood-nazari)
        
        💻 [GitHub](https://github.com/michaeltheanalyst)
        
        ---
        
        **Version**: 2.0
        
        **Date**: December 2025
        """)

if __name__ == "__main__":
    main()
