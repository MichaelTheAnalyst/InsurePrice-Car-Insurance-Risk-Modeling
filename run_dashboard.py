#!/usr/bin/env python3
"""
🚗 InsurePrice Dashboard Launcher
==================================

Quick launcher script for the InsurePrice interactive dashboard.

Usage:
    python run_dashboard.py

This will start the Streamlit dashboard on http://localhost:8501

Author: Masood Nazari
Date: December 2025
"""

import subprocess
import sys
import os

def main():
    """Launch the InsurePrice dashboard"""

    print("🚗 Starting InsurePrice Dashboard...")
    print("=" * 50)
    print("🎨 AI-Powered Car Insurance Risk Modeling")
    print("🎯 Interactive Dashboard with Color Psychology")
    print("=" * 50)

    # Check if dashboard file exists
    dashboard_file = "insureprice_dashboard.py"
    if not os.path.exists(dashboard_file):
        print(f"❌ Error: {dashboard_file} not found!")
        print("Please ensure you're in the correct directory.")
        sys.exit(1)

    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Error: Streamlit not installed!")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

    # Check if data files exist
    required_files = [
        "Enhanced_Synthetic_Car_Insurance_Claims.csv",
        "actuarial_pricing_engine.py"
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Error: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        sys.exit(1)

    print("✅ All dependencies verified")
    print("🚀 Launching dashboard...")

    # Launch streamlit
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", dashboard_file,
               "--server.port", "8501", "--server.address", "localhost"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
