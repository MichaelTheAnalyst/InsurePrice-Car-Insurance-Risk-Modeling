import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
import time

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from gamification.carbon_engine import CarbonGamificationEngine

st.set_page_config(page_title="Eco-Credits Mining", page_icon="🌱", layout="wide")

st.markdown("""
<style>
    .mining-card {
        background: radial-gradient(circle, #22c55e, #14532d);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 0 20px #22c55e;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }
    .stat-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌱 Eco-Credits: Drive Green, Mine Crypto")
st.markdown("Your low-RPM driving is strictly verified on the blockchain. Earn **InsureCoin** to pay your premium.")

engine = CarbonGamificationEngine()

# Simulation Control
with st.sidebar:
    st.header("🔌 Telematics Simulator")
    user_id_input = st.text_input("Vehicle ID", "USR_00042")
    if st.button("🎲 Random Driver"):
        import random
        random_id = f"USR_{random.randint(1, 50000):05d}"
        st.session_state.eco_id = random_id 
        
    if 'eco_id' in st.session_state:
        user_id_input = st.session_state.eco_id
        st.caption(f"Selected: {user_id_input}")

# Get Data
stats = engine.get_driver_eco_stats(user_id_input)
mining = engine.calculate_mining_rate(stats)

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🏎️ Live Mining Rig (Your Car)")
    
    # Mining Gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = mining['mining_rate_per_100_miles'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Mining Hashrate (Credits/100mi)"},
        delta = {'reference': 10, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [0, 20], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#22c55e"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 5], 'color': '#ef4444'},
                {'range': [5, 12], 'color': '#eab308'},
                {'range': [12, 20], 'color': '#22c55e'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': mining['mining_rate_per_100_miles']}
        }))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2, c3 = st.columns(3)
    c1.info(f"**Avg RPM**: {stats['avg_rpm']:,}")
    c2.info(f"**Smoothness**: {stats['smoothness_score']:.2f}")
    c3.info(f"**Efficiency**: {mining['efficiency_score_pct']}%")

with col2:
    st.subheader("💰 Digital Wallet")
    
    st.markdown(f"""
    <div class="mining-card">
        <h3>TOTAL BALANCE</h3>
        <h1>{stats['carbon_credits_balance']:,} ISC</h1>
        <p>≈ £{stats['carbon_credits_balance'] * 0.15:.2f} Insurance Credit</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown(f"### Current Tier: **{mining['tier']}**")
    if mining['rpm_gap'] > 0:
        st.warning(f"⚠️ You are losing tokens! Lower your average RPM by **{mining['rpm_gap']}** to max out mining.")
    else:
        st.success("✅ OPTIMAL MINING STATE REACHED. You are essentially driving for free.")
        
    if st.button("🚀 Start Mining Drive", type="primary", use_container_width=True):
        progress_text = "Calibrating High-Frequency Sensors..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1, text=f"Mining Block #{percent_complete+4021}...")
        st.balloons()
        st.success(f"Drive Complete! Mined {mining['mining_rate_per_100_miles'] * 0.1:.2f} ISC in this session.")
