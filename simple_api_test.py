#!/usr/bin/env python3
"""Simple API test to verify endpoints are working."""

import requests
import time

print("Testing InsurePrice API...")
time.sleep(3)

try:
    # Test health
    response = requests.get('http://localhost:8000/health', timeout=5)
    if response.status_code == 200:
        print("✅ API server is running!")
        data = response.json()
        print(f"Models loaded: {data['models_loaded']}")
        print(f"Scaler loaded: {data['scaler_loaded']}")
    else:
        print(f"❌ Health check failed: {response.status_code}")

except Exception as e:
    print(f"❌ Connection error: {e}")
    print("Make sure to start the API server with: python run_api.py")

