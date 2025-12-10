#!/usr/bin/env python3
"""
InsurePrice API Runner

Launches the FastAPI backend service for the InsurePrice car insurance risk modeling platform.

Usage:
    python run_api.py

The API will be available at:
- Local: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Author: Masood Nazari
Business Intelligence Analyst | Data Science | AI | Clinical Research
Email: M.Nazari@soton.ac.uk
Portfolio: https://michaeltheanalyst.github.io/
LinkedIn: linkedin.com/in/masood-nazari
GitHub: github.com/michaeltheanalyst
"""

import uvicorn
import sys
import os
from pathlib import Path

def main():
    """Launch the InsurePrice API server."""

    print("🚀 Starting InsurePrice API Server")
    print("=" * 50)
    print("Advanced Car Insurance Risk Modeling & Pricing Engine")
    print("=" * 50)

    # Add src to path for imports
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Check if API file exists
    api_file = src_path / "insureprice_api.py"
    if not api_file.exists():
        print("❌ Error: src/insureprice_api.py not found")
        print("Please ensure you're running from the project root directory")
        sys.exit(1)

    # Check for required dependencies
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ API dependencies verified")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install -r requirements_api.txt")
        sys.exit(1)

    print("\n📡 Starting server...")
    print("API Endpoints:")
    print("  • POST /api/v1/risk/score     - Real-time risk scoring")
    print("  • POST /api/v1/premium/quote  - Instant premium calculation")
    print("  • POST /api/v1/portfolio/analyze - Batch portfolio analysis")
    print("  • GET  /api/v1/model/explain/{policy_id} - SHAP explanations")
    print("\n📖 Documentation:")
    print("  • Interactive API Docs: http://localhost:8000/docs")
    print("  • Alternative Docs: http://localhost:8000/redoc")
    print("  • Health Check: http://localhost:8000/health")
    print("\n⚡ Server Configuration:")
    print("  • Host: 0.0.0.0 (accessible from network)")
    print("  • Port: 8000")
    print("  • Auto-reload: Enabled (for development)")
    print("\n" + "=" * 50)

    # Change to src directory for proper imports
    os.chdir(src_path)

    try:
        uvicorn.run(
            "insureprice_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
