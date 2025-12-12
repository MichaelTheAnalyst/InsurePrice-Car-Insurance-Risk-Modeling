"""
InsurePrice Fraud Detection API Extension

REST API endpoints for real-time fraud detection integrated with the claims workflow.

Endpoints:
- POST /api/v1/fraud/analyze - Analyze single claim for fraud indicators
- POST /api/v1/fraud/batch - Batch fraud analysis for multiple claims
- POST /api/v1/fraud/text-analyze - NLP analysis of claim description
- GET /api/v1/fraud/rings - Detect potential fraud rings
- GET /api/v1/fraud/stats - Fraud detection statistics

Author: Masood Nazari
Business Intelligence Analyst | Data Science | AI | Clinical Research
Email: M.Nazari@soton.ac.uk
Portfolio: https://michaeltheanalyst.github.io/
LinkedIn: linkedin.com/in/masood-nazari
GitHub: github.com/michaeltheanalyst
Date: December 2025
Project: InsurePrice Car Insurance Risk Modeling
"""

from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

# Import fraud detection engine
from fraud_detection import FraudDetectionEngine

# Create router for fraud endpoints
fraud_router = APIRouter(prefix="/api/v1/fraud", tags=["Fraud Detection"])

# Initialize fraud detection engine
fraud_engine = FraudDetectionEngine()

# Train with sample data on startup
def initialize_fraud_engine():
    """Initialize fraud detection engine with training data."""
    np.random.seed(42)
    n_claims = 1000
    
    historical_claims = pd.DataFrame({
        'claim_id': [f'CLM-{i:05d}' for i in range(n_claims)],
        'claim_amount': np.random.lognormal(7.5, 1.2, n_claims).clip(100, 50000),
        'days_to_report': np.random.exponential(7, n_claims).astype(int),
        'vehicle_age': np.random.randint(0, 15, n_claims),
        'driver_age': np.random.randint(18, 80, n_claims),
        'annual_mileage': np.random.normal(10000, 3000, n_claims).clip(1000, 30000),
        'previous_claims': np.random.poisson(0.5, n_claims),
        'policy_age_days': np.random.randint(30, 1825, n_claims),
        'time_since_last_claim': np.random.exponential(365, n_claims).astype(int)
    })
    
    fraud_engine.train_anomaly_detector(historical_claims)
    return True


# Request/Response Models
class ClaimData(BaseModel):
    """Claim data for fraud analysis."""
    claim_id: str = Field(..., description="Unique claim identifier")
    claim_amount: float = Field(..., ge=0, description="Claim amount in GBP")
    days_to_report: int = Field(0, ge=0, description="Days between accident and reporting")
    vehicle_age: int = Field(5, ge=0, description="Vehicle age in years")
    driver_age: int = Field(35, ge=17, description="Driver age")
    annual_mileage: float = Field(10000, ge=0, description="Annual mileage")
    previous_claims: int = Field(0, ge=0, description="Number of previous claims")
    policy_age_days: int = Field(365, ge=0, description="Policy age in days")
    time_since_last_claim: int = Field(730, ge=0, description="Days since last claim")
    description: Optional[str] = Field(None, description="Claim description text")
    accident_time: Optional[str] = Field(None, description="Time of accident (HH:MM)")
    accident_day: Optional[str] = Field(None, description="Day of week")
    police_report: Optional[bool] = Field(True, description="Police report filed")
    witnesses: Optional[int] = Field(1, description="Number of witnesses")
    cash_settlement_requested: Optional[bool] = Field(False, description="Cash settlement requested")
    third_party_injuries: Optional[int] = Field(0, description="Number of third-party injuries")
    single_vehicle: Optional[bool] = Field(False, description="Single vehicle accident")


class FraudAnalysisRequest(BaseModel):
    """Request for fraud analysis."""
    claim: ClaimData
    include_detailed_report: Optional[bool] = Field(True, description="Include detailed report")


class BatchFraudRequest(BaseModel):
    """Request for batch fraud analysis."""
    claims: List[ClaimData]


class TextAnalysisRequest(BaseModel):
    """Request for text-only fraud analysis."""
    claim_id: str = Field(..., description="Claim identifier")
    description: str = Field(..., description="Claim description text")


class FraudAnalysisResponse(BaseModel):
    """Response for fraud analysis."""
    claim_id: str
    overall_fraud_score: float
    risk_level: str
    recommendation: str
    component_scores: Dict[str, float]
    anomaly_factors: List[Dict[str, Any]]
    text_red_flags: List[Dict[str, Any]]
    executive_summary: Optional[str]
    recommended_actions: Optional[List[Dict[str, Any]]]
    confidence_level: str
    processing_time: float
    timestamp: str


class TextAnalysisResponse(BaseModel):
    """Response for text analysis."""
    claim_id: str
    text_risk_score: float
    red_flags: List[Dict[str, Any]]
    keyword_matches: Dict[str, List[str]]
    word_count: int
    processing_time: float
    timestamp: str


class FraudRingsResponse(BaseModel):
    """Response for fraud ring detection."""
    total_rings_detected: int
    fraud_rings: List[Dict[str, Any]]
    network_statistics: Dict[str, Any]
    timestamp: str


class FraudStatsResponse(BaseModel):
    """Response for fraud statistics."""
    total_claims_analyzed: int
    high_risk_claims: int
    medium_risk_claims: int
    low_risk_claims: int
    average_fraud_score: float
    top_fraud_indicators: List[Dict[str, Any]]
    timestamp: str


# API Endpoints
@fraud_router.post("/analyze", response_model=FraudAnalysisResponse)
async def analyze_claim_fraud(request: FraudAnalysisRequest):
    """
    Analyze a single claim for fraud indicators.
    
    Uses multiple detection methods:
    - Anomaly detection (Isolation Forest)
    - NLP text analysis
    - Behavioral pattern recognition
    - Network analysis (if applicable)
    """
    start_time = datetime.now()
    
    try:
        # Convert request to dict
        claim_data = request.claim.dict()
        
        if request.include_detailed_report:
            # Generate full fraud report
            report = fraud_engine.generate_fraud_report(claim_data)
            
            return FraudAnalysisResponse(
                claim_id=claim_data['claim_id'],
                overall_fraud_score=report['fraud_assessment']['overall_fraud_score'],
                risk_level=report['fraud_assessment']['risk_level'],
                recommendation=report['fraud_assessment']['recommendation'],
                component_scores=report['fraud_assessment']['component_scores'],
                anomaly_factors=report['fraud_assessment']['anomaly_factors'],
                text_red_flags=report['fraud_assessment']['text_red_flags'],
                executive_summary=report['executive_summary'],
                recommended_actions=report['recommended_actions'],
                confidence_level=report['confidence_level'],
                processing_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now().isoformat()
            )
        else:
            # Quick fraud score only
            fraud_score = fraud_engine.calculate_fraud_score(claim_data)
            
            return FraudAnalysisResponse(
                claim_id=claim_data['claim_id'],
                overall_fraud_score=fraud_score['overall_fraud_score'],
                risk_level=fraud_score['risk_level'],
                recommendation=fraud_score['recommendation'],
                component_scores=fraud_score['component_scores'],
                anomaly_factors=fraud_score['anomaly_factors'],
                text_red_flags=fraud_score['text_red_flags'],
                executive_summary=None,
                recommended_actions=None,
                confidence_level='MEDIUM',
                processing_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now().isoformat()
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fraud analysis failed: {str(e)}")


@fraud_router.post("/batch")
async def batch_fraud_analysis(request: BatchFraudRequest):
    """
    Analyze multiple claims for fraud in batch.
    
    Returns summary statistics and individual risk assessments.
    """
    start_time = datetime.now()
    
    try:
        results = []
        risk_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        total_score = 0
        
        for claim in request.claims:
            claim_data = claim.dict()
            fraud_score = fraud_engine.calculate_fraud_score(claim_data)
            
            results.append({
                'claim_id': claim_data['claim_id'],
                'fraud_score': fraud_score['overall_fraud_score'],
                'risk_level': fraud_score['risk_level'],
                'recommendation': fraud_score['recommendation']
            })
            
            risk_counts[fraud_score['risk_level']] += 1
            total_score += fraud_score['overall_fraud_score']
        
        return {
            'total_claims': len(results),
            'average_fraud_score': round(total_score / len(results), 3) if results else 0,
            'risk_distribution': risk_counts,
            'high_priority_claims': [r for r in results if r['risk_level'] in ['CRITICAL', 'HIGH']],
            'all_results': results,
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@fraud_router.post("/text-analyze", response_model=TextAnalysisResponse)
async def analyze_claim_text(request: TextAnalysisRequest):
    """
    Analyze claim description text for fraud red flags.
    
    Uses NLP to detect:
    - High-risk keywords
    - Suspicious patterns
    - Vague language
    - Excessive detail (rehearsed stories)
    """
    start_time = datetime.now()
    
    try:
        text_result = fraud_engine.analyze_claim_text(request.description)
        
        return TextAnalysisResponse(
            claim_id=request.claim_id,
            text_risk_score=text_result['text_risk_score'],
            red_flags=text_result['red_flags'],
            keyword_matches=text_result['keyword_matches'],
            word_count=text_result['word_count'],
            processing_time=(datetime.now() - start_time).total_seconds(),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")


@fraud_router.get("/rings", response_model=FraudRingsResponse)
async def detect_fraud_rings():
    """
    Detect potential fraud rings from claim network analysis.
    
    Identifies clusters of connected claims that may indicate organized fraud.
    """
    try:
        fraud_rings = fraud_engine.detect_fraud_rings()
        
        return FraudRingsResponse(
            total_rings_detected=len(fraud_rings),
            fraud_rings=fraud_rings,
            network_statistics={
                'total_nodes': fraud_engine.fraud_network.number_of_nodes(),
                'total_edges': fraud_engine.fraud_network.number_of_edges(),
                'average_connections': (
                    fraud_engine.fraud_network.number_of_edges() * 2 / 
                    max(1, fraud_engine.fraud_network.number_of_nodes())
                )
            },
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fraud ring detection failed: {str(e)}")


@fraud_router.get("/stats", response_model=FraudStatsResponse)
async def get_fraud_statistics():
    """
    Get fraud detection statistics and top indicators.
    """
    try:
        # This would typically pull from a database in production
        # For demo, return sample statistics
        return FraudStatsResponse(
            total_claims_analyzed=1000,
            high_risk_claims=47,
            medium_risk_claims=123,
            low_risk_claims=830,
            average_fraud_score=0.18,
            top_fraud_indicators=[
                {'indicator': 'Cash Settlement Requested', 'frequency': 0.23},
                {'indicator': 'No Police Report', 'frequency': 0.18},
                {'indicator': 'New Policy (< 90 days)', 'frequency': 0.15},
                {'indicator': 'Multiple Previous Claims', 'frequency': 0.12},
                {'indicator': 'Late Reporting (> 30 days)', 'frequency': 0.11}
            ],
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")


@fraud_router.get("/keywords")
async def get_fraud_keywords():
    """
    Get list of fraud detection keywords used in NLP analysis.
    """
    from fraud_detection import FRAUD_KEYWORDS
    return {
        'fraud_keywords': FRAUD_KEYWORDS,
        'total_keywords': sum(len(v) for v in FRAUD_KEYWORDS.values()),
        'categories': list(FRAUD_KEYWORDS.keys())
    }


# Initialize engine when module loads
initialize_fraud_engine()

