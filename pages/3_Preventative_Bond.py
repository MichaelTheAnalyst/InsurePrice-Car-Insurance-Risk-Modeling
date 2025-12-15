import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from maintenance.bond_engine import MaintenanceBondEngine

st.set_page_config(page_title="Preventative Bond", page_icon="🛠️", layout="wide")

st.markdown("""
<style>
    .bond-card {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .warning-card {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #f8fafc;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛠️ Preventative Maintenance Bond")
st.markdown("We pay for repairs **before** you crash. Connected to your vehicle's IoT Telematics.")

engine = MaintenanceBondEngine()

# Simulation Control
with st.sidebar:
    st.header("🔌 Telematics Simulator")
    user_id_input = st.text_input("Vehicle ID", "USR_00042")
    if st.button("🎲 Random Vehicle"):
        import random
        random_id = f"USR_{random.randint(1, 50000):05d}"
        st.session_state.r_id = random_id # persistence
        
    if 'r_id' in st.session_state:
        user_id_input = st.session_state.r_id
        st.caption(f"Selected: {user_id_input}")

# Get Data
health = engine.get_vehicle_health(user_id_input)
analysis = engine.analyze_preventative_risk(health)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Vehicle Health")
    
    # GaugesRow
    g1, g2, g3 = st.columns(3)
    
    # Tyre Gauge
    fig_tyre = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = health['tyre_tread_mm'],
        title = {'text': "Tyre Tread (mm)"},
        gauge = {'axis': {'range': [0, 10]},
                 'bar': {'color': "#1e293b"},
                 'steps': [
                     {'range': [0, 1.6], 'color': "#ef4444"},
                     {'range': [1.6, 3.0], 'color': "#f59e0b"},
                     {'range': [3.0, 10], 'color': "#22c55e"}],
                 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1.6}}))
    fig_tyre.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
    g1.plotly_chart(fig_tyre, use_container_width=True)
    
    # Brake Gauge
    fig_brake = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = health['brake_wear_pct'],
        title = {'text': "Brake Pad Wear (%)"},
        # Reversed colors: High wear is bad
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "#1e293b"},
                 'steps': [
                     {'range': [0, 60], 'color': "#22c55e"},
                     {'range': [60, 85], 'color': "#f59e0b"},
                     {'range': [85, 100], 'color': "#ef4444"}]
                 }))
    fig_brake.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20))
    g2.plotly_chart(fig_brake, use_container_width=True)
    
    # Battery Gauge
    fig_batt = go.Figure(go.Indicator(
        mode = "number+delta",
        value = health['battery_soh_pct'],
        title = {"text": "Battery Health"},
        delta = {'reference': 100, 'position': "top"},
        domain = {'x': [0, 1], 'y': [0, 1]}))
    fig_batt.update_layout(height=250)
    g3.plotly_chart(fig_batt, use_container_width=True)

    st.markdown("### ⚠️ Active Alerts")
    if not analysis['alerts']:
        st.success("No active alerts. Vehicle is healthy.")
    else:
        for alert in analysis['alerts']:
            st.markdown(f"""
            <div class="warning-card">
                <strong>{alert['status']} - {alert['component']}</strong><br>
                {alert['msg']}
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.subheader("🛡️ Bond Status")
    
    if analysis['bond_available']:
        st.markdown(f"""
        <div class="bond-card">
            <h3>BOND TRIGGERED</h3>
            <h1>£{analysis['bond_value']:.2f}</h1>
            <p>Immediate Credit Available</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Why?")
        st.write("Our model predicts a **94% chance** of failure/accident in the next 500 miles due to critical wear.")
        
        st.divider()
        if st.button("🔧 Book Repair (Free)", type="primary", use_container_width=True):
            st.balloons()
            st.success("Appointment booked at KwikFit Southampton for tomorrow! Costs covered by InsurePrice.")
            
    else:
         st.info("Preventative Bond inactive. Keep driving safely!")
         st.write(f"Current Risk Score: {analysis['risk_score']:.2f}/1.0")
         st.progress(analysis['risk_score'])
