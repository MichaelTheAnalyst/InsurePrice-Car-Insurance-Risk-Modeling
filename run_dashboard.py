#!/usr/bin/env python3
"""
🚗 InsurePrice Dashboard Launcher
==================================

Quick launcher script for the InsurePrice interactive dashboard.

Usage:
    python run_dashboard.py

This will start the Streamlit dashboard on http://localhost:8501

Author: Masood Nazari
Business Intelligence Analyst | Data Science | AI | Clinical Research
Email: M.Nazari@soton.ac.uk
Portfolio: https://michaeltheanalyst.github.io/
LinkedIn: linkedin.com/in/masood-nazari
GitHub: github.com/michaeltheanalyst
Date: December 2025
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the InsurePrice dashboard"""

    print("🚗 Starting InsurePrice Dashboard...")
    print("=" * 50)
    print("🎨 AI-Powered Car Insurance Risk Modeling")
    print("🎯 Interactive Dashboard with Color Psychology")
    print("=" * 50)

    # Get project root
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    dashboard_file = src_path / "insureprice_dashboard.py"

    # Check if dashboard file exists
    if not dashboard_file.exists():
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
        project_root / "data" / "processed" / "Enhanced_Synthetic_Car_Insurance_Claims.csv",
        src_path / "actuarial_pricing_engine.py"
    ]

    missing_files = [str(f) for f in required_files if not f.exists()]
    if missing_files:
        print("❌ Error: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        sys.exit(1)

    print("✅ All dependencies verified")
    print("🚀 Launching dashboard...")

    # Change to src directory for proper imports
    os.chdir(src_path)

    # Launch streamlit
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_file),
               "--server.port", "8501", "--server.address", "localhost"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
