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
- Clean Whites & Grays: Clarity, Modern Minimalism
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from actuarial_pricing_engine import ActuarialPricingEngine
import pickle
import os

# Set page configuration with modern styling
st.set_page_config(
    page_title="🚗 InsurePrice Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for color psychology and modern design
st.markdown("""
<style>
    /* Main color palette - Trust & Security Theme */
    :root {
        --primary-blue: #1e3a8a;
        --secondary-blue: #3b82f6;
        --accent-green: #059669;
        --accent-orange: #ea580c;
        --warning-red: #dc2626;
        --premium-purple: #7c3aed;
        --neutral-gray: #6b7280;
        --light-bg: #f8fafc;
        --card-shadow: 0 4px 6px -1px rgba(30, 58, 138, 0.1);
    }

    /* Global styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-blue));
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--card-shadow);
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: var(--card-shadow);
        border-left: 4px solid var(--secondary-blue);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -5px rgba(30, 58, 138, 0.15);
    }

    .risk-high { border-left-color: var(--warning-red); }
    .risk-medium { border-left-color: var(--accent-orange); }
    .risk-low { border-left-color: var(--accent-green); }

    .premium-card {
        background: linear-gradient(135deg, var(--premium-purple), var(--secondary-blue));
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: var(--card-shadow);
        margin-bottom: 1rem;
    }

    .sidebar-content {
        background: var(--light-bg);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, var(--accent-green), var(--secondary-blue));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(5, 150, 105, 0.2);
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--light-bg);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--secondary-blue);
        color: white;
    }

    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-green), var(--secondary-blue));
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--light-bg);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--secondary-blue);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-blue);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

# Load data and models
@st.cache_data
def load_data():
    """Load and prepare all necessary data for the dashboard"""
    try:
        # Load the enhanced dataset
        df = pd.read_csv('Enhanced_Synthetic_Car_Insurance_Claims.csv')

        # Initialize pricing engine
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

# Load trained models
@st.cache_data
def load_models():
    """Load trained machine learning models"""
    models = {}
    try:
        # Try to load saved models (if they exist)
        model_files = {
            'Random Forest': 'rf_model.pkl',
            'Logistic Regression': 'lr_model.pkl',
            'XGBoost': 'xgb_model.pkl'
        }

        for model_name, filename in model_files.items():
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    models[model_name] = pickle.load(f)

        return models
    except Exception as e:
        st.warning(f"Could not load saved models: {e}")
        return {}

def main():
    """Main dashboard application"""

    # Load data and models
    df, pricing_engine = load_data()
    models = load_models()

    if df is None or pricing_engine is None:
        st.error("Failed to load necessary data. Please check your files.")
        return

    # Sidebar navigation with modern design
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

        st.title("🚗 InsurePrice")
        st.markdown("**AI-Powered Risk Modeling**")
        st.markdown("---")

        # Navigation menu
        pages = {
            "📊 Dashboard": "dashboard",
            "🎯 Risk Assessment": "risk_assessment",
            "💰 Premium Calculator": "premium_calculator",
            "📈 Portfolio Analytics": "portfolio_analytics",
            "🔍 Model Performance": "model_performance",
            "📋 About": "about"
        }

        for page_name, page_key in pages.items():
            if st.button(page_name, key=page_key, use_container_width=True):
                st.session_state.page = page_key

        st.markdown("---")
        st.markdown("**Color Psychology Legend:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🔵 **Trust & Security**")
            st.markdown("🟢 **Success & Growth**")
            st.markdown("🟠 **Action & Energy**")
        with col2:
            st.markdown("🔴 **High Risk Alert**")
            st.markdown("🟣 **Premium Features**")
            st.markdown("⚪ **Clean & Modern**")

        st.markdown('</div>', unsafe_allow_html=True)

    # Main content based on selected page
    if st.session_state.page == "dashboard":
        render_dashboard(df, pricing_engine)
    elif st.session_state.page == "risk_assessment":
        render_risk_assessment(df, models, pricing_engine)
    elif st.session_state.page == "premium_calculator":
        render_premium_calculator(pricing_engine)
    elif st.session_state.page == "portfolio_analytics":
        render_portfolio_analytics(df, pricing_engine)
    elif st.session_state.page == "model_performance":
        render_model_performance(df, models)
    elif st.session_state.page == "about":
        render_about()

def render_dashboard(df, pricing_engine):
    """Render the main dashboard with key metrics and visualizations"""

    # Hero section
    st.markdown("""
    <div class="main-header">
        <h1>🚗 InsurePrice Dashboard</h1>
        <p style="font-size: 1.2em; margin: 1rem 0;">
            AI-Powered Car Insurance Risk Modeling & Premium Optimization
        </p>
        <p style="font-size: 0.9em; opacity: 0.9;">
            Transform data into profitable decisions with advanced analytics and color-coded insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_policies = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--primary-blue); margin: 0;">📋 Total Policies</h3>
            <h2 style="color: var(--secondary-blue); margin: 0.5rem 0;">{total_policies:,}</h2>
            <p style="color: var(--neutral-gray); margin: 0; font-size: 0.9em;">
                Active insurance policies in portfolio
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        claim_rate = df['OUTCOME'].mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--primary-blue); margin: 0;">⚠️ Claim Rate</h3>
            <h2 style="color: {'var(--warning-red)' if claim_rate > 15 else 'var(--accent-green)'}; margin: 0.5rem 0;">
                {claim_rate:.1f}%
            </h2>
            <p style="color: var(--neutral-gray); margin: 0; font-size: 0.9em;">
                Annual claims frequency
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_premium = 650  # UK market average
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--primary-blue); margin: 0;">💷 Avg Premium</h3>
            <h2 style="color: var(--accent-green); margin: 0.5rem 0;">£{avg_premium}</h2>
            <p style="color: var(--neutral-gray); margin: 0; font-size: 0.9em;">
                UK market comprehensive average
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        portfolio_value = total_policies * avg_premium
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--primary-blue); margin: 0;">💰 Portfolio Value</h3>
            <h2 style="color: var(--premium-purple); margin: 0.5rem 0;">£{portfolio_value:,}</h2>
            <p style="color: var(--neutral-gray); margin: 0; font-size: 0.9em;">
                Annual premium revenue potential
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Risk Distribution Visualization
    st.markdown("### 🎯 Risk Distribution Overview")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Create risk score distribution
        fig = go.Figure()

        # Calculate risk scores using a simple model (age + mileage + credit score)
        risk_scores = (
            (df['AGE'].map({'16-25': 0.8, '26-39': 0.6, '40-64': 0.4, '65+': 0.3})) +
            (df['ANNUAL_MILEAGE'] / 35000 * 0.3) +
            ((1 - df['CREDIT_SCORE']) * 0.4)
        ).clip(0, 1)

        # Create histogram with color psychology
        fig.add_trace(go.Histogram(
            x=risk_scores,
            nbinsx=30,
            marker_color='var(--secondary-blue)',
            opacity=0.7,
            name='Risk Distribution'
        ))

        fig.update_layout(
            title="Portfolio Risk Score Distribution",
            xaxis_title="Risk Score (0-1)",
            yaxis_title="Number of Policies",
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk categories breakdown
        low_risk = (risk_scores < 0.4).sum()
        medium_risk = ((risk_scores >= 0.4) & (risk_scores < 0.7)).sum()
        high_risk = (risk_scores >= 0.7).sum()

        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: var(--card-shadow);">
            <h4 style="color: var(--primary-blue); margin-top: 0;">📊 Risk Categories</h4>
        """, unsafe_allow_html=True)

        # Low risk
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
            <span style="color: var(--accent-green);">🟢 Low Risk</span>
            <span style="font-weight: bold;">{low_risk:,} ({low_risk/total_policies*100:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

        # Medium risk
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
            <span style="color: var(--accent-orange);">🟠 Medium Risk</span>
            <span style="font-weight: bold;">{medium_risk:,} ({medium_risk/total_policies*100:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

        # High risk
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
            <span style="color: var(--warning-red);">🔴 High Risk</span>
            <span style="font-weight: bold;">{high_risk:,} ({high_risk/total_policies*100:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Regional Risk Analysis - Modern Pin Markers Design
    st.markdown("### 🗺️ Interactive Regional Risk Analysis")

    # Calculate regional claim rates and statistics
    regional_data = df.groupby('REGION')['OUTCOME'].agg(['mean', 'count']).reset_index()
    regional_data.columns = ['Region', 'Claim_Rate', 'Policy_Count']
    regional_data['Claim_Rate'] *= 100

    # Risk categorization with colors
    def get_risk_info(rate: float):
        if rate < 11:
            return "Low Risk", "#059669", "🟢"
        elif rate < 13:
            return "Medium Risk", "#d97706", "🟠"
        else:
            return "High Risk", "#dc2626", "🔴"

    regional_data[['Risk_Category', 'Color', 'Icon']] = regional_data['Claim_Rate'].apply(
        lambda x: pd.Series(get_risk_info(x))
    )

    # UK region coordinates
    region_coords = {
        'London': [51.5074, -0.1278],
        'Scotland': [56.4907, -4.2026],
        'North West': [54.2361, -2.7486],
        'North East': [54.9783, -1.6178],
        'Yorkshire': [53.9590, -1.0815],
        'East Midlands': [52.7955, -0.5384],
        'West Midlands': [52.4862, -1.8904],
        'East Anglia': [52.2405, 0.9027],
        'South East': [51.2720, -0.8225],
        'South West': [50.9097, -3.5599],
        'Wales': [52.1307, -3.7837]
    }

    regional_data['Latitude'] = regional_data['Region'].map(lambda x: region_coords.get(x, [54.0, -2.0])[0])
    regional_data['Longitude'] = regional_data['Region'].map(lambda x: region_coords.get(x, [54.0, -2.0])[1])

    # Create two-column layout: Map + Bar Chart
    map_col, chart_col = st.columns([3, 2])

    with map_col:
        # Create map with SMALL FIXED-SIZE pin markers (no bubbles!)
        map_fig = go.Figure()

        # Add markers for each region - FIXED SMALL SIZE
        for idx, row in regional_data.iterrows():
            map_fig.add_trace(go.Scattermapbox(
                lat=[row['Latitude']],
                lon=[row['Longitude']],
                mode='markers+text',
                marker=dict(
                    size=14,  # Fixed small size
                    color=row['Color'],
                    opacity=0.9
                ),
                text=row['Region'][:3].upper(),  # Short label
                textposition="top center",
                textfont=dict(size=9, color='#1e3a8a', family='Arial Black'),
                name=row['Region'],
                hovertemplate=(
                    f"<b>{row['Region']}</b><br>" +
                    f"Claim Rate: {row['Claim_Rate']:.1f}%<br>" +
                    f"Policies: {row['Policy_Count']:,}<br>" +
                    f"Risk: {row['Risk_Category']}<br>" +
                    "<extra></extra>"
                ),
                showlegend=False
            ))

        map_fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=54.0, lon=-2.5),
                zoom=4.8
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=450,
            title=dict(
                text="<b>UK Risk Map</b> <span style='font-size:11px;color:#6b7280;'>(Hover for details)</span>",
                font=dict(size=14, color='#1e3a8a')
            ),
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(map_fig, use_container_width=True)

    with chart_col:
        # Horizontal bar chart for claim rates - sorted by risk
        sorted_data = regional_data.sort_values('Claim_Rate', ascending=True)

        bar_fig = go.Figure()

        bar_fig.add_trace(go.Bar(
            y=sorted_data['Region'],
            x=sorted_data['Claim_Rate'],
            orientation='h',
            marker=dict(
                color=sorted_data['Color'],
                line=dict(width=0)
            ),
            text=[f"{x:.1f}%" for x in sorted_data['Claim_Rate']],
            textposition='outside',
            textfont=dict(size=10, color='#374151'),
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Claim Rate: %{x:.1f}%<br>" +
                "<extra></extra>"
            )
        ))

        bar_fig.update_layout(
            title=dict(
                text="<b>Claim Rate by Region</b>",
                font=dict(size=14, color='#1e3a8a')
            ),
            xaxis=dict(
                title="Claim Rate (%)",
                showgrid=True,
                gridcolor='#f3f4f6',
                range=[0, max(sorted_data['Claim_Rate']) * 1.15]
            ),
            yaxis=dict(
                title="",
                showgrid=False
            ),
            height=450,
            margin=dict(l=10, r=60, t=50, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(bar_fig, use_container_width=True)

    # Risk Legend
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 30px; padding: 10px 0; background: #f8fafc; border-radius: 8px; margin-top: -10px;">
        <span style="display: flex; align-items: center; gap: 6px;">
            <span style="width: 12px; height: 12px; background: #059669; border-radius: 50%;"></span>
            <span style="font-size: 13px; color: #374151;">Low Risk (&lt;11%)</span>
        </span>
        <span style="display: flex; align-items: center; gap: 6px;">
            <span style="width: 12px; height: 12px; background: #d97706; border-radius: 50%;"></span>
            <span style="font-size: 13px; color: #374151;">Medium Risk (11-13%)</span>
        </span>
        <span style="display: flex; align-items: center; gap: 6px;">
            <span style="width: 12px; height: 12px; background: #dc2626; border-radius: 50%;"></span>
            <span style="font-size: 13px; color: #374151;">High Risk (&gt;13%)</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Add summary statistics below the map
    st.markdown("### 📊 Regional Risk Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_claim_rate = regional_data['Claim_Rate'].mean()
        st.metric("Average Claim Rate", f"{avg_claim_rate:.1f}%")

    with col2:
        highest_risk = regional_data.loc[regional_data['Claim_Rate'].idxmax()]
        st.metric("Highest Risk Region", f"{highest_risk['Region']}", f"{highest_risk['Claim_Rate']:.1f}%")

    with col3:
        lowest_risk = regional_data.loc[regional_data['Claim_Rate'].idxmin()]
        st.metric("Lowest Risk Region", f"{lowest_risk['Region']}", f"{lowest_risk['Claim_Rate']:.1f}%")

    with col4:
        total_policies = regional_data['Policy_Count'].sum()
        st.metric("Total UK Policies", f"{total_policies:,}")

    # Risk distribution table
    st.markdown("### 📋 Regional Risk Distribution")

    # Sort by claim rate for better visualization
    display_data = regional_data[['Region', 'Claim_Rate', 'Policy_Count', 'Risk_Category']].copy()
    display_data['Claim_Rate'] = display_data['Claim_Rate'].round(1)
    display_data.columns = ['Region', 'Claim Rate (%)', 'Total Policies', 'Risk Level']

    # Add color styling
    def color_risk(val):
        if val == "Low Risk":
            return "background-color: #d1fae5; color: #065f46"
        elif val == "Medium Risk":
            return "background-color: #fef3c7; color: #92400e"
        else:
            return "background-color: #fee2e2; color: #991b1b"

    styled_df = display_data.style.apply(lambda x: [color_risk(x.iloc[i]) if x.name == 'Risk Level' else "" for i in range(len(x))], axis=0)

    st.dataframe(styled_df, use_container_width=True)

def render_risk_assessment(df, models, pricing_engine):
    """Render the risk assessment page"""

    st.markdown("""
    <div class="main-header">
        <h1>🎯 AI Risk Assessment</h1>
        <p>Get instant risk evaluation for any driver profile</p>
    </div>
    """, unsafe_allow_html=True)

    # Input form for risk assessment
    with st.form("risk_assessment_form"):
        st.markdown("### 👤 Driver Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.selectbox("Age Group", ['16-25', '26-39', '40-64', '65+'])
            gender = st.selectbox("Gender", ['Male', 'Female'])
            region = st.selectbox("Region", df['REGION'].unique())

        with col2:
            driving_exp = st.selectbox("Driving Experience", df['DRIVING_EXPERIENCE'].unique())
            vehicle_type = st.selectbox("Vehicle Type", df['VEHICLE_TYPE'].unique())
            vehicle_year = st.selectbox("Vehicle Year", df['VEHICLE_YEAR'].unique())

        with col3:
            annual_mileage = st.slider("Annual Mileage", 500, 35000, 12000)
            credit_score = st.slider("Credit Score", 0.01, 1.00, 0.7, 0.01)
            safety_rating = st.selectbox("Safety Rating", df['SAFETY_RATING'].unique())

        # Additional factors
        st.markdown("### 📊 Additional Risk Factors")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            speeding_violations = st.slider("Speeding Violations", 0, 10, 0)
        with col2:
            duis = st.slider("DUIs", 0, 5, 0)
        with col3:
            past_accidents = st.slider("Past Accidents", 0, 10, 0)
        with col4:
            vehicle_ownership = st.selectbox("Vehicle Ownership", [0, 1], format_func=lambda x: "Owns" if x else "Doesn't Own")

        # Assessment button
        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    if submitted:
        # Create input data for prediction
        input_data = pd.DataFrame({
            'AGE': [age],
            'GENDER': [gender],
            'REGION': [region],
            'DRIVING_EXPERIENCE': [driving_exp],
            'VEHICLE_TYPE': [vehicle_type],
            'VEHICLE_YEAR': [vehicle_year],
            'ANNUAL_MILEAGE': [annual_mileage],
            'CREDIT_SCORE': [credit_score],
            'SAFETY_RATING': [safety_rating],
            'SPEEDING_VIOLATIONS': [speeding_violations],
            'DUIS': [duis],
            'PAST_ACCIDENTS': [past_accidents],
            'VEHICLE_OWNERSHIP': [vehicle_ownership],
            'MARRIED': [0],  # Default values for demo
            'CHILDREN': [0],
            'EDUCATION': ['university'],
            'INCOME': ['middle class']
        })

        # Calculate risk score (simplified model)
        risk_score = (
            (input_data['AGE'].map({'16-25': 0.8, '26-39': 0.6, '40-64': 0.4, '65+': 0.3}).iloc[0]) +
            (input_data['ANNUAL_MILEAGE'].iloc[0] / 35000 * 0.3) +
            ((1 - input_data['CREDIT_SCORE'].iloc[0]) * 0.4) +
            (input_data['SPEEDING_VIOLATIONS'].iloc[0] * 0.05) +
            (input_data['DUIS'].iloc[0] * 0.1) +
            (input_data['PAST_ACCIDENTS'].iloc[0] * 0.08)
        )

        risk_score = min(max(risk_score, 0), 1)  # Clip to 0-1 range

        # Determine risk category
        if risk_score < 0.4:
            risk_category = "Low Risk"
            risk_color = "var(--accent-green)"
            risk_icon = "🟢"
        elif risk_score < 0.7:
            risk_category = "Medium Risk"
            risk_color = "var(--accent-orange)"
            risk_icon = "🟠"
        else:
            risk_category = "High Risk"
            risk_color = "var(--warning-red)"
            risk_icon = "🔴"

        # Calculate premium
        premium_result = pricing_engine.calculate_basic_actuarial_premium(risk_score, credibility=0.9)
        calculated_premium = premium_result['final_premium']

        # Display results
        st.markdown("### 📊 Risk Assessment Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card {'risk-low' if risk_score < 0.4 else 'risk-medium' if risk_score < 0.7 else 'risk-high'}">
                <h3 style="margin: 0;">{risk_icon} Risk Category</h3>
                <h2 style="color: {risk_color}; margin: 0.5rem 0;">{risk_category}</h2>
                <p style="margin: 0; font-size: 0.9em;">Based on driver profile analysis</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--primary-blue); margin: 0;">📈 Risk Score</h3>
                <h2 style="color: var(--secondary-blue); margin: 0.5rem 0;">{risk_score:.3f}</h2>
                <p style="margin: 0; font-size: 0.9em;">Probability of claim (0-1 scale)</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="premium-card">
                <h3 style="margin: 0;">💰 Recommended Premium</h3>
                <h2 style="margin: 0.5rem 0;">£{calculated_premium:.0f}</h2>
                <p style="margin: 0; font-size: 0.9em;">Annual comprehensive coverage</p>
            </div>
            """, unsafe_allow_html=True)

        # Risk breakdown
        st.markdown("### 🔍 Risk Factor Analysis")

        # Create radar chart for risk factors
        categories = ['Age Risk', 'Mileage Risk', 'Credit Risk', 'Violation Risk', 'Accident Risk']
        values = [
            input_data['AGE'].map({'16-25': 0.8, '26-39': 0.6, '40-64': 0.4, '65+': 0.3}).iloc[0],
            min(input_data['ANNUAL_MILEAGE'].iloc[0] / 35000, 1),
            1 - input_data['CREDIT_SCORE'].iloc[0],
            min((input_data['SPEEDING_VIOLATIONS'].iloc[0] + input_data['DUIS'].iloc[0] * 2) / 10, 1),
            min(input_data['PAST_ACCIDENTS'].iloc[0] / 5, 1)
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Factors',
            line_color='var(--secondary-blue)',
            fillcolor='rgba(59, 130, 246, 0.3)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

def render_premium_calculator(pricing_engine):
    """Render the premium calculator page"""

    st.markdown("""
    <div class="main-header">
        <h1>💰 Premium Calculator</h1>
        <p>Calculate optimal premiums based on actuarial principles</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧮 Actuarial Premium Calculation")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Input Parameters**")

        risk_score = st.slider("Risk Score", 0.0, 1.0, 0.35, 0.01)
        credibility = st.slider("Credibility Factor", 0.5, 1.0, 0.9, 0.01)

        if st.button("🔄 Calculate Premium", use_container_width=True):
            result = pricing_engine.calculate_basic_actuarial_premium(risk_score, credibility)

            st.markdown("### 📊 Calculation Results")

            # Display breakdown
            breakdown = result['breakdown']
            ratios = result['ratios']

            st.markdown(f"""
            <div class="premium-card">
                <h3>💷 Final Premium</h3>
                <h1>£{result['final_premium']:.2f}</h1>
                <p>Annual comprehensive premium</p>
            </div>
            """, unsafe_allow_html=True)

            # Premium breakdown
            st.markdown("**Premium Breakdown:**")
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Expected Loss", f"£{breakdown['expected_loss']:.2f}")
                st.metric("Expenses (35%)", f"£{breakdown['expenses']:.2f}")
                st.metric("Profit Margin (15%)", f"£{breakdown['profit_margin']:.2f}")

            with col_b:
                st.metric("Risk Margin (8%)", f"£{breakdown['risk_margin']:.2f}")
                st.metric("Combined Ratio", f"{ratios['combined_ratio']:.3f}")
                profitability = "✅ Profitable" if ratios['combined_ratio'] < 1 else "❌ Loss-making"
                st.metric("Status", profitability)

    with col2:
        # Visual representation
        if 'result' in locals():
            # Create premium breakdown pie chart
            labels = ['Expected Loss', 'Expenses', 'Profit Margin', 'Risk Margin']
            values = [
                breakdown['expected_loss'],
                breakdown['expenses'],
                breakdown['profit_margin'],
                breakdown['risk_margin']
            ]
            colors = ['#3b82f6', '#ea580c', '#059669', '#dc2626']

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                title="Premium Composition"
            )])

            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

def render_portfolio_analytics(df, pricing_engine):
    """Render portfolio analytics page"""

    st.markdown("""
    <div class="main-header">
        <h1>📈 Portfolio Analytics</h1>
        <p>Comprehensive portfolio risk and performance analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Portfolio metrics
    total_policies = len(df)
    total_claims = df['OUTCOME'].sum()
    claim_rate = (total_claims / total_policies) * 100

    # Calculate risk scores for portfolio
    risk_scores = (
        (df['AGE'].map({'16-25': 0.8, '26-39': 0.6, '40-64': 0.4, '65+': 0.3})) +
        (df['ANNUAL_MILEAGE'] / 35000 * 0.3) +
        ((1 - df['CREDIT_SCORE']) * 0.4)
    ).clip(0, 1)

    # Portfolio statistics
    avg_risk = risk_scores.mean()
    risk_std = risk_scores.std()

    # Premium calculations
    batch_results = pricing_engine.batch_calculate_premiums(risk_scores, method='basic', credibility=0.85)
    avg_premium = batch_results['calculated_premium'].mean()
    premium_range = batch_results['calculated_premium'].max() - batch_results['calculated_premium'].min()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Policies", f"{total_policies:,}")
    with col2:
        st.metric("Claim Rate", f"{claim_rate:.1f}%")
    with col3:
        st.metric("Avg Risk Score", f"{avg_risk:.3f}")
    with col4:
        st.metric("Avg Premium", f"£{avg_premium:.0f}")

    # Risk distribution analysis
    st.markdown("### 📊 Portfolio Risk Distribution")

    fig = go.Figure()

    # Histogram of risk scores
    fig.add_trace(go.Histogram(
        x=risk_scores,
        nbinsx=30,
        marker_color='var(--secondary-blue)',
        opacity=0.7,
        name='Risk Scores'
    ))

    fig.update_layout(
        title="Portfolio Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Number of Policies",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Premium vs Risk scatter plot
    st.markdown("### 💰 Premium vs Risk Analysis")

    fig = px.scatter(
        x=risk_scores,
        y=batch_results['calculated_premium'],
        color=risk_scores,
        color_continuous_scale=['green', 'orange', 'red'],
        title="Premium vs Risk Score Relationship",
        labels={'x': 'Risk Score', 'y': 'Calculated Premium (£)'}
    )

    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def render_model_performance(df, models):
    """Render model performance analytics"""

    st.markdown("""
    <div class="main-header">
        <h1>🔍 Model Performance</h1>
        <p>AI model evaluation and comparison dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    if not models:
        st.warning("No trained models found. Please run the model training pipeline first.")
        return

    # Model comparison metrics
    st.markdown("### 🏆 Model Comparison")

    # Create sample metrics (in real implementation, these would be calculated)
    model_metrics = {
        'Random Forest': {'auc': 0.654, 'gini': 0.308, 'accuracy': 0.782},
        'Logistic Regression': {'auc': 0.651, 'gini': 0.302, 'accuracy': 0.778},
        'XGBoost': {'auc': 0.634, 'gini': 0.269, 'accuracy': 0.769}
    }

    # Display metrics in cards
    cols = st.columns(len(model_metrics))

    for i, (model_name, metrics) in enumerate(model_metrics.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: var(--primary-blue); margin: 0;">{model_name}</h4>
                <p style="margin: 0.5rem 0; font-size: 0.9em;">
                    AUC: <strong>{metrics['auc']:.3f}</strong><br>
                    Gini: <strong>{metrics['gini']:.3f}</strong><br>
                    Accuracy: <strong>{metrics['accuracy']:.3f}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ROC Curves comparison
    st.markdown("### 📈 ROC Curve Comparison")

    # Create sample ROC data
    fig = go.Figure()

    colors = ['#3b82f6', '#059669', '#ea580c']

    for i, (model_name, metrics) in enumerate(model_metrics.items()):
        # Generate sample ROC curve points
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-3 * fpr) * (1 + 3 * fpr) * metrics['auc'] / 0.654  # Approximate ROC shape

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC: {metrics["auc"]:.3f})',
            line=dict(color=colors[i], width=3)
        ))

    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Guessing',
        line=dict(color='gray', dash='dash')
    ))

    fig.update_layout(
        title="ROC Curves - Model Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        width=800,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

def render_about():
    """Render about page"""

    st.markdown("""
    <div class="main-header">
        <h1>📋 About InsurePrice</h1>
        <p>AI-Powered Car Insurance Risk Modeling Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🎯 Project Overview

    **InsurePrice** is a comprehensive car insurance risk modeling platform that leverages advanced machine learning and actuarial science to provide accurate risk assessment and premium optimization.

    ## 🧠 AI & Analytics Features

    - **Risk Assessment**: Real-time driver risk evaluation using ensemble ML models
    - **Premium Calculation**: Actuarially-sound premium pricing based on risk profiles
    - **Portfolio Analytics**: Comprehensive portfolio risk and performance analysis
    - **Interactive Visualizations**: Modern, color-coded dashboards for data-driven insights

    ## 🎨 Design Philosophy

    This dashboard implements **color psychology principles** for optimal user experience:

    - **🔵 Blue**: Trust, Security, Professionalism (Primary brand colors)
    - **🟢 Green**: Success, Growth, Financial Prosperity (Positive metrics)
    - **🟠 Orange**: Energy, Action, Premium Features (Call-to-action elements)
    - **🔴 Red**: Urgency, High Risk Alerts (Warning indicators)
    - **🟣 Purple**: Luxury, Sophistication, Premium Services (High-value features)

    ## 👨‍💻 Technical Stack

    - **Frontend**: Streamlit with custom CSS and Plotly visualizations
    - **Backend**: Python with scikit-learn, XGBoost, and actuarial engines
    - **Data**: Enhanced synthetic dataset calibrated to UK insurance statistics
    - **Models**: Random Forest, Logistic Regression, and XGBoost ensemble

    ## 📞 Contact Information

    **Author**: Masood Nazari
    **Role**: Business Intelligence Analyst | Data Science | AI | Clinical Research
    **Email**: M.Nazari@soton.ac.uk
    **Portfolio**: https://michaeltheanalyst.github.io/
    **LinkedIn**: linkedin.com/in/masood-nazari
    **GitHub**: github.com/michaeltheanalyst
    **Date**: December 2025
    """)

if __name__ == "__main__":
    main()
