import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from p2p.community_pool import CommunityCluster, DividendCalculator

st.set_page_config(page_title="My Village Pool", page_icon="🏘️", layout="wide")

st.markdown("""
<style>
    .village-header {
        background: linear-gradient(90deg, #4f46e5, #0ea5e9);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .dividend-box {
        background-color: #dcfce7;
        border: 2px solid #22c55e;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: #15803d;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏘️ Village P2P Insurance")

# Initialize Logic
cluster = CommunityCluster()
calculator = DividendCalculator()

# Mock User Login
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {'age': 28, 'annual_mileage': 4000}

# Sidebar for Demo controls
with st.sidebar:
    st.header("👤 Mock User Profile")
    age = st.slider("User Age", 18, 80, 42)
    miles = st.slider("Annual Mileage", 1000, 20000, 8000)
    night_pct = st.slider("Night Driving %", 0.0, 1.0, 0.1)
    
    st.session_state.user_profile = {
        'age': age, 
        'annual_mileage': miles,
        'night_driving_percent': night_pct
    }
    
    # Help text to explain logic
    st.info("""
    **Assignment Logic:**
    - Age < 25 → **Young Pros**
    - Miles < 5k → **Weekend Warriors**
    - Night > 30% → **Night Owls**
    - Else → **Safe Commuters**
    """)

# 1. Determine Village
village_id = cluster.assign_user_to_village(st.session_state.user_profile)
stats = cluster.get_village_stats(village_id)

# Header
st.markdown(f"""
<div class="village-header">
    <h3>You are a member of:</h3>
    <h1>{stats['name']} 🛡️</h1>
    <p>Collective Risk Factor: {stats['risk_factor']}x | {stats['members']:,} Members</p>
</div>
""", unsafe_allow_html=True)

# New Stats Row
m1, m2, m3 = st.columns(3)
m1.metric("Avg Driver Fatigue (0-10)", f"{stats.get('avg_fatigue', 5.0):.1f}")
m2.metric("Avg Annual Mileage", f"{stats.get('avg_mileage', 8000):,.0f} miles")
m3.metric("Pool Value", f"£{stats['total_pool_value']:,.0f}")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Pool Health (Live)")
    
    # Financial Simulation
    total_pot = stats['total_pool_value']
    
    # Slider to simulate chaos
    st.markdown("#### Simulator: Month-End Scenarios")
    claims_scenario = st.slider("Total Claims Paid this Month (£)", 
                               min_value=0, 
                               max_value=int(total_pot * 1.2), 
                               value=int(total_pot * 0.4))
    
    finances = calculator.calculate_period_performance(
        pool_value=total_pot, 
        claims_paid=claims_scenario
    )
    
    # Visualizing the Pot
    data = {
        'Category': ['Claims Paid', 'Platform Fee', 'Reserve Fund', 'Surplus (DIVIDEND)'],
        'Amount': [
            finances['expenses']['claims'],
            finances['expenses']['platform_fee'],
            finances['expenses']['reserve_buffer'],
            max(0, finances['surplus_deficit'])
        ]
    }
    
    fig = px.pie(data, values='Amount', names='Category', 
                 title="Where does your premium go?",
                 hole=0.4,
                 color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💰 Your Dividend")
    
    surplus = finances['surplus_deficit']
    
    if surplus > 0:
        member_share = surplus / stats['members']
        st.markdown(f"""
        <div class="dividend-box">
            <h3>PROJECTED CASHBACK</h3>
            <h1>£{member_share:,.2f}</h1>
            <p>Hitting your account in 3 days.</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.error(f"No Dividend this month. The pool had high claims (Deficit: -£{abs(surplus):,.2f}).")
        st.markdown("*Better luck next month! Drive safely to protect the pool.*")

    st.divider()
    
    st.markdown("### 🗳️ Governance Vote")
    st.info("A member submitted a claim for a 'hit and run' providing no police report. Should we pay?")
    c1, c2 = st.columns(2)
    c1.button("✅ Approve", use_container_width=True)
    c2.button("❌ Reject", use_container_width=True)
