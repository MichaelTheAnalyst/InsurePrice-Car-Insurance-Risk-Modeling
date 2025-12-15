import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import sys
import os

# Add src to path so we can import our modules
# Assuming this file is in PROJECT_ROOT/pages/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from simulation.digital_twin import DigitalTwinSimulator

st.set_page_config(page_title="Risk Twin Simulator", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .big-stat {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    .stat-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Digital Risk Twin Simulator")
st.markdown("""
**Concept**: Instead of a static risk score, simulate your *actual* daily commute 10,000 times against historical weather and traffic patterns to determine your **Probabilistic Risk Profile**.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🛠️ Commute Configuration")
    
    with st.container(border=True):
        commute_name = st.text_input("Route Name", value="Home to Office (Southampton)")
        
        region = st.selectbox(
            "Region (Environmental Data)",
            ["London", "South East", "South West", "East Anglia", "West Midlands", "North West", "North East", "Scotland", "Wales"],
            index=1
        )
        
        distance = st.slider("Distance (miles)", 1, 100, 15)
        
        st.markdown("### Driver State")
        fatigue = st.select_slider(
            "Typical Driver Fatigue",
            options=["Alert", "Tired", "Exhausted"],
            value="Alert"
        )
        
        fatigue_map = {"Alert": 1.0, "Tired": 1.5, "Exhausted": 2.5}
        
        sim_count = st.number_input("Simulation Runs", min_value=100, max_value=50000, value=2000, step=100)
        
        run_btn = st.button("🚀 Run Risk Simulation", use_container_width=True, type="primary")

with col2:
    if run_btn:
        simulator = DigitalTwinSimulator()
        
        with st.spinner(f"Running {sim_count} simulations for '{region}' weather patterns..."):
            df_results = simulator.simulate_commute(
                distance_miles=distance,
                n_simulations=sim_count,
                driver_fatigue_level=fatigue_map[fatigue],
                region=region
            )
            
            stats = simulator.get_summary_stats(df_results)
            
            # --- Results Dashboard ---
            
            # 1. Top Level Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Crash Probability", f"{stats['crash_probability']:.2%}")
            m2.metric("Avg Loss (if crash)", f"£{stats['avg_damage_if_crash']:,.0f}")
            m3.metric("Max Potential Loss", f"£{stats['max_simulated_loss']:,.0f}")
            m4.metric("Worst Weather", stats['most_dangerous_weather'])
            
            # 2. Risk Distribution Chart
            st.divider()
            
            tab1, tab2 = st.tabs(["📉 Probabilistic Outcome", "🌪️ Environmental Impact"])
            
            with tab1:
                # Filter out 0 (safe trips) for the histogram, or show them?
                # Showing safe trips dwarfs the accidents, so let's focus on non-zero risk or just accident probability density?
                # Better: Histogram of 'accident_prob_percent' for all trips (Risk Exposure)
                
                fig_dist = px.histogram(
                    df_results, 
                    x="accident_prob_percent", 
                    nbins=50,
                    title="Distribution of Daily Accident Risk (%)",
                    color="weather",
                    labels={"accident_prob_percent": "Probability of Accident Today (%)"},
                    color_discrete_map={
                        "Clear": "#22c55e", "Rain": "#3b82f6", "Fog": "#64748b", "Snow": "#f59e0b", "Storm": "#ef4444"
                    }
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                st.info("This chart shows that on most days, your risk is low (green). However, the long tail (red/orange) represents 'bad days' where weather and traffic align to spike your risk significantly.")

            with tab2:
                # Impact of Weather on Accidents
                weather_risk = df_results.groupby('weather')['accident_prob_percent'].mean().reset_index()
                fig_bar = px.bar(
                    weather_risk, 
                    x='weather', 
                    y='accident_prob_percent',
                    title="Average Risk by Weather Condition",
                    color='weather',
                    color_discrete_map={
                        "Clear": "#22c55e", "Rain": "#3b82f6", "Fog": "#64748b", "Snow": "#f59e0b", "Storm": "#ef4444"
                    }
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("👈 Configure your commute and click 'Run Simulation' to generate your Digital Risk Twin.")
        
        # Placeholder image or skeleton
        st.markdown("""
        ### What is a Digital Risk Twin?
        Using stochastic modeling, we don't just ask "Are you a good driver?". We ask: 
        > *"If you drove this route 2,000 times in different UK weather conditions, how many times would you crash?"*
        
        This accounts for:
        - **Black Swan Events**: Rare storms combined with heavy traffic.
        - **Route Complexity**: Longer routes exponentially increase exposure.
        - **Fatigue Multipliers**: How your state affects the outcome.
        """)
