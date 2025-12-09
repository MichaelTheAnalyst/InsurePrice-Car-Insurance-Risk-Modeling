#!/usr/bin/env python3
"""
InsurePrice API Test Script

Demonstrates how to interact with the InsurePrice API endpoints.
Tests all major functionality: risk scoring, premium quotes, portfolio analysis, and explanations.

Usage:
    python test_api.py

Prerequisites:
    1. Start the API server: python run_api.py
    2. Run this test script in a separate terminal

Author: Masood Nazari
Business Intelligence Analyst | Data Science | AI | Clinical Research
Email: M.Nazari@soton.ac.uk
Portfolio: https://michaeltheanalyst.github.io/
LinkedIn: linkedin.com/in/masood-nazari
GitHub: github.com/michaeltheanalyst
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_api_health() -> bool:
    """Test API health endpoint."""
    print("🏥 Testing API Health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy")
            print(f"   Models loaded: {data['models_loaded']}")
            print(f"   Timestamp: {data['timestamp']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("   Make sure to start the server first: python run_api.py")
        return False

def test_risk_scoring() -> Dict[str, Any]:
    """Test risk scoring endpoint."""
    print("\\n🎯 Testing Risk Scoring...")

    # Sample driver profile (moderate risk)
    driver_profile = {
        "age": "26-39",
        "gender": "male",
        "region": "London",
        "driving_experience": "10-19y",
        "education": "university",
        "income": "middle_class",
        "vehicle_type": "family_sedan",
        "vehicle_year": "2016-2020",
        "annual_mileage": 12000.0,
        "credit_score": 0.75,
        "speeding_violations": 0,
        "duis": 0,
        "past_accidents": 1,
        "vehicle_ownership": 1,
        "married": 1,
        "children": 0,
        "safety_rating": "standard"
    }

    payload = {
        "driver_profile": driver_profile,
        "model_name": "Random Forest"
    }

    try:
        response = requests.post(f"{BASE_URL}/api/v1/risk/score", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ Risk scoring successful")
            print(".3f")
            print(f"   Risk Category: {data['risk_category']}")
            print(".3f")
            print(".3f")
            return data
        else:
            print(f"❌ Risk scoring failed: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Risk scoring error: {e}")
        return None

def test_premium_quote() -> Dict[str, Any]:
    """Test premium quote endpoint."""
    print("\\n💰 Testing Premium Quote...")

    # Same driver profile as risk scoring
    driver_profile = {
        "age": "26-39",
        "gender": "male",
        "region": "London",
        "driving_experience": "10-19y",
        "education": "university",
        "income": "middle_class",
        "vehicle_type": "family_sedan",
        "vehicle_year": "2016-2020",
        "annual_mileage": 12000.0,
        "credit_score": 0.75,
        "speeding_violations": 0,
        "duis": 0,
        "past_accidents": 1,
        "vehicle_ownership": 1,
        "married": 1,
        "children": 0,
        "safety_rating": "standard"
    }

    payload = {
        "driver_profile": driver_profile,
        "coverage_type": "comprehensive",
        "voluntary_excess": 200,
        "ncd_years": 2
    }

    try:
        response = requests.post(f"{BASE_URL}/api/v1/premium/quote", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ Premium quote successful")
            print(".2f")
            print(".2f")
            print(".3f")
            print(f"   Coverage: {data['coverage_details']['coverage_type']}")
            print(f"   NCD Years: {data['coverage_details']['ncd_years']}")
            print(f"   Voluntary Excess: £{data['coverage_details']['voluntary_excess']}")
            if data['discounts_applied']:
                print(f"   Discounts Applied: {', '.join(data['discounts_applied'])}")
            return data
        else:
            print(f"❌ Premium quote failed: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Premium quote error: {e}")
        return None

def test_portfolio_analysis() -> Dict[str, Any]:
    """Test portfolio analysis endpoint."""
    print("\\n📊 Testing Portfolio Analysis...")

    # Create a small portfolio of 3 drivers with different risk profiles
    portfolio = [
        # Low risk driver
        {
            "age": "40-64",
            "gender": "female",
            "region": "South West",
            "driving_experience": "20-29y",
            "education": "university",
            "income": "upper_class",
            "vehicle_type": "family_sedan",
            "vehicle_year": "2016-2020",
            "annual_mileage": 8000.0,
            "credit_score": 0.85,
            "speeding_violations": 0,
            "duis": 0,
            "past_accidents": 0,
            "vehicle_ownership": 1,
            "married": 1,
            "children": 1,
            "safety_rating": "advanced"
        },
        # Medium risk driver
        {
            "age": "26-39",
            "gender": "male",
            "region": "London",
            "driving_experience": "10-19y",
            "education": "university",
            "income": "middle_class",
            "vehicle_type": "family_sedan",
            "vehicle_year": "2016-2020",
            "annual_mileage": 12000.0,
            "credit_score": 0.75,
            "speeding_violations": 0,
            "duis": 0,
            "past_accidents": 1,
            "vehicle_ownership": 1,
            "married": 1,
            "children": 0,
            "safety_rating": "standard"
        },
        # High risk driver
        {
            "age": "16-25",
            "gender": "male",
            "region": "North East",
            "driving_experience": "0-2y",
            "education": "high_school",
            "income": "working_class",
            "vehicle_type": "sports_car",
            "vehicle_year": "after_2020",
            "annual_mileage": 15000.0,
            "credit_score": 0.45,
            "speeding_violations": 2,
            "duis": 0,
            "past_accidents": 1,
            "vehicle_ownership": 0,
            "married": 0,
            "children": 0,
            "safety_rating": "basic"
        }
    ]

    payload = {
        "driver_profiles": portfolio,
        "analysis_type": "summary"
    }

    try:
        response = requests.post(f"{BASE_URL}/api/v1/portfolio/analyze", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ Portfolio analysis successful")
            print(f"   Total Policies: {data['portfolio_summary']['total_policies']}")
            print(".3f")
            print(".2f")
            print(f"   Risk Distribution: {data['risk_distribution']}")

            if data['recommendations']:
                print("   Recommendations:")
                for rec in data['recommendations']:
                    print(f"     • {rec}")

            return data
        else:
            print(f"❌ Portfolio analysis failed: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Portfolio analysis error: {e}")
        return None

def test_model_explanation() -> Dict[str, Any]:
    """Test model explanation endpoint."""
    print("\\n🔍 Testing Model Explanation...")

    # Use the same driver profile as risk scoring
    driver_profile = {
        "age": "26-39",
        "gender": "male",
        "region": "London",
        "driving_experience": "10-19y",
        "education": "university",
        "income": "middle_class",
        "vehicle_type": "family_sedan",
        "vehicle_year": "2016-2020",
        "annual_mileage": 12000.0,
        "credit_score": 0.75,
        "speeding_violations": 0,
        "duis": 0,
        "past_accidents": 1,
        "vehicle_ownership": 1,
        "married": 1,
        "children": 0,
        "safety_rating": "standard"
    }

    policy_id = "test_policy_001"

    try:
        response = requests.get(f"{BASE_URL}/api/v1/model/explain/{policy_id}",
                              params={"driver_profile": json.dumps(driver_profile)})

        if response.status_code == 200:
            data = response.json()
            print("✅ Model explanation successful")
            print(f"   Policy ID: {data['policy_id']}")
            print(".3f")

            print("   Top Contributing Factors:")
            for i, factor in enumerate(data['top_factors'][:3], 1):
                direction = "↑ Increases Risk" if factor['shap_value'] > 0 else "↓ Decreases Risk"
                print(".3f")

            return data
        else:
            print(f"❌ Model explanation failed: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Model explanation error: {e}")
        return None

def main():
    """Run all API tests."""
    print("🧪 InsurePrice API Test Suite")
    print("=" * 50)
    print("Testing all major API endpoints")
    print("=" * 50)

    # Test API health
    if not test_api_health():
        print("\\n❌ API tests aborted - server not available")
        return

    # Test individual endpoints
    risk_result = test_risk_scoring()
    time.sleep(0.5)  # Brief pause between requests

    premium_result = test_premium_quote()
    time.sleep(0.5)

    portfolio_result = test_portfolio_analysis()
    time.sleep(0.5)

    explanation_result = test_model_explanation()

    # Summary
    print("\\n" + "=" * 50)
    print("📊 API TEST SUMMARY")
    print("=" * 50)

    tests = [
        ("API Health", True),  # Always true if we got here
        ("Risk Scoring", risk_result is not None),
        ("Premium Quote", premium_result is not None),
        ("Portfolio Analysis", portfolio_result is not None),
        ("Model Explanation", explanation_result is not None)
    ]

    passed = 0
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print("15")
        if success:
            passed += 1

    print(f"\\n🎯 Overall: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\\n🎉 All API endpoints are working correctly!")
        print("🚀 InsurePrice API is ready for production use!")
    else:
        print(f"\\n⚠️ {len(tests) - passed} tests failed. Check server logs for details.")

    print("\\n📖 API Documentation: http://localhost:8000/docs")
    print("=" * 50)

if __name__ == "__main__":
    main()
