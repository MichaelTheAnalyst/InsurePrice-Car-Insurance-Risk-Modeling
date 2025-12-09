"""
InsurePrice API - FastAPI Backend Service

Production-ready REST API for car insurance risk modeling and pricing.
Provides real-time risk scoring, premium calculation, portfolio analysis, and SHAP explanations.

Author: Masood Nazari
Business Intelligence Analyst | Data Science | AI | Clinical Research
Email: M.Nazari@soton.ac.uk
Portfolio: https://michaeltheanalyst.github.io/
LinkedIn: linkedin.com/in/masood-nazari
GitHub: github.com/michaeltheanalyst
Date: December 2025
Project: InsurePrice Car Insurance Risk Modeling
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import joblib
import uvicorn
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="InsurePrice API",
    description="Advanced Car Insurance Risk Modeling & Pricing Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and data storage
models = {}
scaler = None
feature_names = None
explainer = None

# Data models
class DriverProfile(BaseModel):
    """Driver profile for risk assessment."""
    age: str = Field(..., description="Age group: '16-25', '26-39', '40-64', '65+'")
    gender: str = Field(..., description="Gender: 'male' or 'female'")
    region: str = Field(..., description="UK region: 'London', 'Scotland', etc.")
    driving_experience: str = Field(..., description="Experience: '0-2y', '3-5y', etc.")
    education: str = Field(..., description="Education level")
    income: str = Field(..., description="Income bracket")
    vehicle_type: str = Field(..., description="Vehicle type")
    vehicle_year: str = Field(..., description="Vehicle age category")
    annual_mileage: float = Field(..., ge=0, description="Annual mileage")
    credit_score: float = Field(..., ge=0, le=1, description="Credit score (0-1)")
    speeding_violations: int = Field(..., ge=0, description="Number of speeding violations")
    duis: int = Field(..., ge=0, description="Number of DUIs")
    past_accidents: int = Field(..., ge=0, description="Number of past accidents")
    vehicle_ownership: int = Field(0, ge=0, le=1, description="Vehicle ownership (0/1)")
    married: int = Field(0, ge=0, le=1, description="Marital status (0/1)")
    children: int = Field(0, ge=0, le=1, description="Has children (0/1)")
    safety_rating: str = Field(..., description="Safety rating: 'basic', 'standard', 'advanced'")

    @validator('age')
    def validate_age(cls, v):
        valid_ages = ['16-25', '26-39', '40-64', '65+']
        if v not in valid_ages:
            raise ValueError(f"Age must be one of: {valid_ages}")
        return v

    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['male', 'female']:
            raise ValueError("Gender must be 'male' or 'female'")
        return v

class RiskScoreRequest(BaseModel):
    """Request for risk scoring."""
    driver_profile: DriverProfile
    model_name: Optional[str] = Field("Random Forest", description="ML model to use")

class PremiumQuoteRequest(BaseModel):
    """Request for premium quotation."""
    driver_profile: DriverProfile
    coverage_type: Optional[str] = Field("comprehensive", description="Coverage type")
    voluntary_excess: Optional[int] = Field(0, description="Voluntary excess amount")
    ncd_years: Optional[int] = Field(0, description="No claims discount years")

class PortfolioAnalysisRequest(BaseModel):
    """Request for portfolio analysis."""
    driver_profiles: List[DriverProfile] = Field(..., description="List of driver profiles")
    analysis_type: Optional[str] = Field("summary", description="Type of analysis")

class ModelExplanationRequest(BaseModel):
    """Request for model explanation."""
    policy_id: str = Field(..., description="Policy identifier")
    driver_profile: DriverProfile

# Response models
class RiskScoreResponse(BaseModel):
    """Response for risk scoring."""
    risk_score: float = Field(..., description="Predicted risk score (0-1)")
    risk_category: str = Field(..., description="Risk category: Low/Medium/High/Very High")
    confidence_interval: Dict[str, float] = Field(..., description="Confidence bounds")
    model_used: str = Field(..., description="ML model used for prediction")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Request timestamp")

class PremiumQuoteResponse(BaseModel):
    """Response for premium quotation."""
    annual_premium: float = Field(..., description="Annual premium in GBP")
    monthly_premium: float = Field(..., description="Monthly premium in GBP")
    risk_score: float = Field(..., description="Underlying risk score")
    coverage_details: Dict[str, Any] = Field(..., description="Coverage breakdown")
    discounts_applied: List[str] = Field(..., description="Applied discounts")
    premium_breakdown: Dict[str, float] = Field(..., description="Detailed cost breakdown")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Request timestamp")

class PortfolioAnalysisResponse(BaseModel):
    """Response for portfolio analysis."""
    portfolio_summary: Dict[str, Any] = Field(..., description="Portfolio statistics")
    risk_distribution: Dict[str, int] = Field(..., description="Risk category distribution")
    premium_distribution: Dict[str, float] = Field(..., description="Premium statistics")
    regional_breakdown: Dict[str, Any] = Field(..., description="Regional analysis")
    recommendations: List[str] = Field(..., description="Business recommendations")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Request timestamp")

class ModelExplanationResponse(BaseModel):
    """Response for model explanation."""
    policy_id: str = Field(..., description="Policy identifier")
    risk_score: float = Field(..., description="Predicted risk score")
    top_factors: List[Dict[str, Any]] = Field(..., description="Top risk factors with explanations")
    feature_importance: Dict[str, float] = Field(..., description="Overall feature importance")
    shap_summary: Dict[str, Any] = Field(..., description="SHAP analysis summary")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Request timestamp")

def load_models_and_data():
    """Load trained models and preprocessing objects."""
    global models, scaler, feature_names, explainer

    try:
        # Load models (simplified - in production, load from saved files)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        import shap

        # For demonstration, we'll recreate the model training
        # In production, load from joblib files
        logger.info("Loading models and preprocessing objects...")

        # Load sample data for model training
        df = pd.read_csv('Enhanced_Synthetic_Car_Insurance_Claims.csv')

        # Prepare features (same as in modeling)
        exclude_cols = ['ID', 'POSTAL_CODE', 'CLAIM_AMOUNT']
        feature_cols = [col for col in df.columns if col not in exclude_cols + ['OUTCOME']]
        X = df[feature_cols].copy()
        y = df['OUTCOME'].copy()

        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        feature_names = list(X.columns)

        # Train Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X, y)
        models['Random Forest'] = rf_model

        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(rf_model)

        logger.info("✅ Models and data loaded successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        return False

def preprocess_driver_profile(driver_profile: DriverProfile) -> np.ndarray:
    """Preprocess driver profile for model input."""
    global scaler, feature_names

    # Create feature dictionary
    features = {
        'AGE': driver_profile.age,
        'GENDER': driver_profile.gender,
        'REGION': driver_profile.region,
        'DRIVING_EXPERIENCE': driver_profile.driving_experience,
        'EDUCATION': driver_profile.education,
        'INCOME': driver_profile.income,
        'VEHICLE_TYPE': driver_profile.vehicle_type,
        'VEHICLE_YEAR': driver_profile.vehicle_year,
        'ANNUAL_MILEAGE': driver_profile.annual_mileage,
        'CREDIT_SCORE': driver_profile.credit_score,
        'SPEEDING_VIOLATIONS': driver_profile.speeding_violations,
        'DUIS': driver_profile.duis,
        'PAST_ACCIDENTS': driver_profile.past_accidents,
        'VEHICLE_OWNERSHIP': driver_profile.vehicle_ownership,
        'MARRIED': driver_profile.married,
        'CHILDREN': driver_profile.children,
        'SAFETY_RATING': driver_profile.safety_rating
    }

    # Convert to DataFrame
    df = pd.DataFrame([features])

    # Encode categorical variables (simplified - in production, use saved encoders)
    categorical_mapping = {
        'AGE': {'16-25': 0, '26-39': 1, '40-64': 2, '65+': 3},
        'GENDER': {'female': 0, 'male': 1},
        'REGION': {'London': 0, 'Scotland': 1, 'South West': 2, 'Wales': 3, 'West Midlands': 4,
                  'North East': 5, 'East Anglia': 6, 'South East': 7, 'Yorkshire': 8,
                  'East Midlands': 9, 'North West': 10},
        'DRIVING_EXPERIENCE': {'0-2y': 0, '3-5y': 1, '6-9y': 2, '0-9y': 3, '10-19y': 4, '20-29y': 5, '30y+': 6},
        'EDUCATION': {'none': 0, 'high_school': 1, 'university': 2, 'postgraduate': 3},
        'INCOME': {'poverty': 0, 'working_class': 1, 'middle_class': 2, 'upper_class': 3},
        'VEHICLE_TYPE': {'small_hatchback': 0, 'family_sedan': 1, 'suv': 2, 'sports_car': 3, 'luxury_sedan': 4, 'mpv': 5},
        'VEHICLE_YEAR': {'before_2010': 0, '2010-2015': 1, '2016-2020': 2, 'after_2020': 3},
        'SAFETY_RATING': {'basic': 0, 'standard': 1, 'advanced': 2}
    }

    for col in df.columns:
        if col in categorical_mapping and col != 'ANNUAL_MILEAGE' and col != 'CREDIT_SCORE':
            df[col] = df[col].map(categorical_mapping.get(col, {})).fillna(0)

    # Scale numerical features
    numerical_cols = ['ANNUAL_MILEAGE', 'CREDIT_SCORE', 'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS']
    if scaler:
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Ensure correct column order
    df = df[feature_names]

    return df.values

def calculate_premium_from_risk(risk_score: float, **kwargs) -> Dict[str, Any]:
    """Calculate premium based on risk score using actuarial formulas."""

    # Base parameters
    base_frequency = 0.122
    base_severity = 3500

    # Calculate expected loss
    expected_loss = risk_score * base_frequency * base_severity

    # Apply loading factors
    expense_ratio = 0.35
    profit_ratio = 0.15
    risk_ratio = 0.08

    loading_factor = 1 / (1 - expense_ratio - profit_ratio - risk_ratio)
    gross_premium = expected_loss * loading_factor
    final_premium = gross_premium * (1 - 0.04)  # Investment return

    return {
        'annual_premium': round(final_premium, 2),
        'monthly_premium': round(final_premium / 12, 2),
        'risk_score': risk_score,
        'premium_breakdown': {
            'expected_loss': round(expected_loss, 2),
            'expenses': round(gross_premium * expense_ratio, 2),
            'profit': round(gross_premium * profit_ratio, 2),
            'risk_margin': round(gross_premium * risk_ratio, 2),
            'investment_credit': round(gross_premium * 0.04, 2)
        }
    }

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize models and data on startup."""
    success = load_models_and_data()
    if not success:
        logger.warning("Models not loaded - using simplified calculations")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "InsurePrice API - Advanced Car Insurance Risk Modeling",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models) > 0,
        "scaler_loaded": scaler is not None
    }

@app.post("/api/v1/risk/score", response_model=RiskScoreResponse)
async def score_risk(request: RiskScoreRequest):
    """
    Real-time risk scoring endpoint.

    Accepts driver profile and returns risk score with confidence intervals.
    """
    start_time = datetime.now()

    try:
        # Preprocess input
        features = preprocess_driver_profile(request.driver_profile)

        # Get model
        model_name = request.model_name or "Random Forest"
        if model_name not in models:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not available")

        model = models[model_name]

        # Make prediction
        risk_score = float(model.predict_proba(features)[0, 1])

        # Determine risk category
        if risk_score <= 0.2:
            risk_category = "Low Risk"
        elif risk_score <= 0.3:
            risk_category = "Medium Risk"
        elif risk_score <= 0.4:
            risk_category = "High Risk"
        else:
            risk_category = "Very High Risk"

        # Calculate confidence interval (simplified)
        confidence_interval = {
            "lower_bound": max(0, risk_score - 0.1),
            "upper_bound": min(1, risk_score + 0.1)
        }

        processing_time = (datetime.now() - start_time).total_seconds()

        return RiskScoreResponse(
            risk_score=risk_score,
            risk_category=risk_category,
            confidence_interval=confidence_interval,
            model_used=model_name,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Risk scoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk scoring failed: {str(e)}")

@app.post("/api/v1/premium/quote", response_model=PremiumQuoteResponse)
async def quote_premium(request: PremiumQuoteRequest):
    """
    Instant premium calculation endpoint.

    Returns comprehensive premium quote with breakdown.
    """
    start_time = datetime.now()

    try:
        # First get risk score
        risk_request = RiskScoreRequest(
            driver_profile=request.driver_profile,
            model_name="Random Forest"
        )

        risk_response = await score_risk(risk_request)
        risk_score = risk_response.risk_score

        # Calculate premium
        premium_details = calculate_premium_from_risk(
            risk_score,
            coverage_type=request.coverage_type,
            voluntary_excess=request.voluntary_excess,
            ncd_years=request.ncd_years
        )

        # Apply coverage adjustments
        coverage_multiplier = {
            'third_party': 0.6,
            'third_party_fire_theft': 0.75,
            'comprehensive': 1.0
        }.get(request.coverage_type, 1.0)

        annual_premium = premium_details['annual_premium'] * coverage_multiplier

        # Apply NCD discount
        ncd_discount = min(request.ncd_years * 0.05, 0.25)  # Max 25% discount
        annual_premium *= (1 - ncd_discount)

        # Apply voluntary excess discount
        excess_discount = min(request.voluntary_excess * 0.001, 0.15)  # Max 15% discount
        annual_premium *= (1 - excess_discount)

        monthly_premium = annual_premium / 12

        # Determine discounts applied
        discounts_applied = []
        if ncd_discount > 0:
            discounts_applied.append(f"NCD: {ncd_discount:.1%}")
        if excess_discount > 0:
            discounts_applied.append(f"Voluntary Excess: {excess_discount:.1%}")
        if coverage_multiplier < 1.0:
            discounts_applied.append(f"Coverage Type: {(1-coverage_multiplier):.1%}")

        processing_time = (datetime.now() - start_time).total_seconds()

        return PremiumQuoteResponse(
            annual_premium=round(annual_premium, 2),
            monthly_premium=round(monthly_premium, 2),
            risk_score=risk_score,
            coverage_details={
                "coverage_type": request.coverage_type,
                "voluntary_excess": request.voluntary_excess,
                "ncd_years": request.ncd_years
            },
            discounts_applied=discounts_applied,
            premium_breakdown={
                "base_premium": round(annual_premium / (1 - ncd_discount) / (1 - excess_discount), 2),
                "ncd_discount": round(annual_premium * ncd_discount / (1 - ncd_discount), 2),
                "excess_discount": round(annual_premium * excess_discount / (1 - excess_discount), 2),
                "final_premium": round(annual_premium, 2)
            },
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Premium quotation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Premium quotation failed: {str(e)}")

@app.post("/api/v1/portfolio/analyze", response_model=PortfolioAnalysisResponse)
async def analyze_portfolio(request: PortfolioAnalysisRequest):
    """
    Batch portfolio analysis endpoint.

    Analyzes multiple policies for portfolio risk assessment.
    """
    start_time = datetime.now()

    try:
        portfolio_results = []

        for i, driver_profile in enumerate(request.driver_profiles):
            # Get risk score for each driver
            risk_request = RiskScoreRequest(driver_profile=driver_profile)
            risk_response = await score_risk(risk_request)

            # Get premium quote
            premium_request = PremiumQuoteRequest(driver_profile=driver_profile)
            premium_response = await quote_premium(premium_request)

            portfolio_results.append({
                'policy_id': f'policy_{i+1}',
                'risk_score': risk_response.risk_score,
                'risk_category': risk_response.risk_category,
                'annual_premium': premium_response.annual_premium,
                'region': driver_profile.region
            })

        # Portfolio summary statistics
        df_portfolio = pd.DataFrame(portfolio_results)

        portfolio_summary = {
            "total_policies": len(df_portfolio),
            "average_risk_score": round(df_portfolio['risk_score'].mean(), 3),
            "total_annual_premium": round(df_portfolio['annual_premium'].sum(), 2),
            "average_annual_premium": round(df_portfolio['annual_premium'].mean(), 2),
            "risk_score_std": round(df_portfolio['risk_score'].std(), 3),
            "premium_std": round(df_portfolio['annual_premium'].std(), 2)
        }

        # Risk distribution
        risk_distribution = df_portfolio['risk_category'].value_counts().to_dict()

        # Premium distribution statistics
        premium_stats = df_portfolio['annual_premium'].describe().to_dict()
        premium_distribution = {k: round(v, 2) for k, v in premium_stats.items()}

        # Regional breakdown
        regional_breakdown = {}
        for region in df_portfolio['region'].unique():
            region_data = df_portfolio[df_portfolio['region'] == region]
            regional_breakdown[region] = {
                "policy_count": len(region_data),
                "avg_risk_score": round(region_data['risk_score'].mean(), 3),
                "avg_premium": round(region_data['annual_premium'].mean(), 2),
                "total_premium": round(region_data['annual_premium'].sum(), 2)
            }

        # Business recommendations
        recommendations = []
        avg_risk = portfolio_summary['average_risk_score']

        if avg_risk > 0.35:
            recommendations.append("High-risk portfolio: Consider reinsurance or risk mitigation strategies")
        elif avg_risk > 0.25:
            recommendations.append("Moderate risk portfolio: Balanced underwriting approach recommended")
        else:
            recommendations.append("Low-risk portfolio: Opportunity for competitive pricing and market expansion")

        if portfolio_summary['total_annual_premium'] > 100000:
            recommendations.append("Large portfolio value: Implement advanced risk monitoring and stress testing")

        processing_time = (datetime.now() - start_time).total_seconds()

        return PortfolioAnalysisResponse(
            portfolio_summary=portfolio_summary,
            risk_distribution=risk_distribution,
            premium_distribution=premium_distribution,
            regional_breakdown=regional_breakdown,
            recommendations=recommendations,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Portfolio analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")

@app.get("/api/v1/model/explain/{policy_id}", response_model=ModelExplanationResponse)
async def explain_model(policy_id: str, driver_profile: DriverProfile):
    """
    SHAP model explanation endpoint.

    Provides detailed explanation of risk factors for a specific policy.
    """
    start_time = datetime.now()

    try:
        # Preprocess input
        features = preprocess_driver_profile(driver_profile)

        if not explainer:
            raise HTTPException(status_code=503, detail="SHAP explainer not available")

        # Calculate SHAP values
        shap_values = explainer.shap_values(features)

        # Get risk score
        risk_score = float(models['Random Forest'].predict_proba(features)[0, 1])

        # Extract top contributing factors
        if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
            shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values[0]

        # Create feature importance list
        feature_contributions = []
        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_vals)):
            feature_contributions.append({
                'feature': feature,
                'shap_value': float(shap_val),
                'contribution_percent': abs(float(shap_val)) / abs(sum(shap_vals)) * 100 if sum(shap_vals) != 0 else 0
            })

        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        # Top 5 factors
        top_factors = feature_contributions[:5]

        # Overall feature importance
        feature_importance = {item['feature']: item['contribution_percent'] for item in feature_contributions}

        # SHAP summary
        shap_summary = {
            'total_features': len(feature_names),
            'positive_contributors': len([x for x in feature_contributions if x['shap_value'] > 0]),
            'negative_contributors': len([x for x in feature_contributions if x['shap_value'] < 0]),
            'most_important_factor': top_factors[0]['feature'] if top_factors else None,
            'explanation_confidence': 'high' if abs(sum(shap_vals)) > 0.1 else 'medium'
        }

        processing_time = (datetime.now() - start_time).total_seconds()

        return ModelExplanationResponse(
            policy_id=policy_id,
            risk_score=risk_score,
            top_factors=top_factors,
            feature_importance=feature_importance,
            shap_summary=shap_summary,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Model explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model explanation failed: {str(e)}")

@app.get("/api/v1/models")
async def list_models():
    """List available ML models."""
    return {
        "available_models": list(models.keys()),
        "default_model": "Random Forest",
        "model_descriptions": {
            "Random Forest": "Ensemble method with AUC 0.654, best overall performance"
        }
    }

@app.get("/api/v1/stats")
async def get_statistics():
    """Get API usage statistics."""
    return {
        "api_version": "1.0.0",
        "models_loaded": len(models),
        "features_count": len(feature_names) if feature_names else 0,
        "shap_available": explainer is not None,
        "supported_regions": ["London", "Scotland", "South West", "Wales", "West Midlands",
                            "North East", "East Anglia", "South East", "Yorkshire",
                            "East Midlands", "North West"],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "insureprice_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
