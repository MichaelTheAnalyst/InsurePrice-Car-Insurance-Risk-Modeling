"""
Regulatory Compliance Module for Insurance Pricing
====================================================

Provides capabilities for:
- FCA (UK Financial Conduct Authority) compliance
- GDPR data protection compliance
- Solvency II requirements
- Model Risk Management (SR 11-7 style)
- Fairness monitoring and reporting
- Audit trail management
- Model drift detection

Author: Masood Nazari
Business Intelligence Analyst | Data Science | AI | Clinical Research
Email: M.Nazari@soton.ac.uk
Portfolio: https://michaeltheanalyst.github.io/
LinkedIn: linkedin.com/in/masood-nazari
GitHub: github.com/michaeltheanalyst
Date: December 2025
Project: InsurePrice Car Insurance Risk Modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import uuid


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"


class ProtectedCharacteristic(Enum):
    """Protected characteristics under UK Equality Act 2010."""
    AGE = "age"
    GENDER = "gender"
    DISABILITY = "disability"
    RACE = "race"
    RELIGION = "religion"
    SEXUAL_ORIENTATION = "sexual_orientation"
    PREGNANCY = "pregnancy"
    MARRIAGE = "marriage"


@dataclass
class AuditRecord:
    """Audit trail record for pricing decisions."""
    record_id: str
    timestamp: datetime
    policy_id: str
    customer_id: str
    action: str  # 'quote', 'pricing', 'decision'
    model_version: str
    input_features: Dict[str, Any]
    output: Dict[str, Any]
    explanation: Dict[str, Any]
    user_id: str
    compliance_flags: List[str]
    data_lineage: Dict[str, str]


@dataclass
class FairnessMetric:
    """Fairness metric result."""
    characteristic: str
    group_a: str
    group_b: str
    metric_name: str
    metric_value: float
    threshold: float
    status: ComplianceStatus
    timestamp: datetime


@dataclass
class ModelDriftAlert:
    """Model drift detection alert."""
    alert_id: str
    timestamp: datetime
    metric_name: str
    baseline_value: float
    current_value: float
    drift_percentage: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    recommended_action: str


class RegulatoryComplianceEngine:
    """
    Comprehensive regulatory compliance engine for insurance pricing.
    
    Supports FCA, GDPR, and Solvency II requirements.
    """
    
    def __init__(self):
        """Initialize the compliance engine."""
        self.audit_trail: List[AuditRecord] = []
        self.fairness_metrics: List[FairnessMetric] = []
        self.drift_alerts: List[ModelDriftAlert] = []
        
        # FCA Fairness thresholds
        self.disparate_impact_threshold = 0.8  # 80% rule
        self.demographic_parity_threshold = 0.05  # 5% difference
        
        # Model drift thresholds
        self.drift_thresholds = {
            'auc': 0.05,  # 5% drop triggers alert
            'gini': 0.05,
            'conversion_rate': 0.10,
            'claim_frequency': 0.15,
            'average_premium': 0.10
        }
        
        # Baseline metrics (set during model deployment)
        self.baseline_metrics = {
            'auc': 0.654,
            'gini': 0.308,
            'conversion_rate': 0.12,
            'claim_frequency': 0.122,
            'average_premium': 650
        }
        
        # Model metadata
        self.model_metadata = {
            'model_id': 'RF_v2.1',
            'model_name': 'Random Forest Risk Classifier',
            'version': '2.1.0',
            'deployed_date': datetime(2025, 12, 1),
            'last_validated': datetime(2025, 12, 10),
            'owner': 'Actuarial Team',
            'approver': 'Chief Actuary',
            'risk_tier': 'Tier 1 - High Impact'
        }
        
        # Data sources for lineage
        self.data_sources = {
            'customer_data': {'source': 'CRM System', 'classification': 'PII', 'retention': '7 years'},
            'claims_history': {'source': 'Claims Database', 'classification': 'Sensitive', 'retention': '10 years'},
            'vehicle_data': {'source': 'DVLA API', 'classification': 'Public', 'retention': '3 years'},
            'credit_score': {'source': 'Experian API', 'classification': 'Financial', 'retention': '2 years'},
            'external_risk': {'source': 'Industry Database', 'classification': 'Aggregate', 'retention': '5 years'}
        }
    
    def create_audit_record(self,
                           policy_id: str,
                           customer_id: str,
                           action: str,
                           input_features: Dict[str, Any],
                           output: Dict[str, Any],
                           explanation: Dict[str, Any],
                           user_id: str = "system") -> AuditRecord:
        """Create an audit trail record for a pricing decision."""
        
        # Generate unique record ID
        record_id = f"AUD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        
        # Check for compliance flags
        compliance_flags = self._check_compliance_flags(input_features, output)
        
        # Build data lineage
        data_lineage = self._build_data_lineage(input_features)
        
        record = AuditRecord(
            record_id=record_id,
            timestamp=datetime.now(),
            policy_id=policy_id,
            customer_id=customer_id,
            action=action,
            model_version=self.model_metadata['version'],
            input_features=input_features,
            output=output,
            explanation=explanation,
            user_id=user_id,
            compliance_flags=compliance_flags,
            data_lineage=data_lineage
        )
        
        self.audit_trail.append(record)
        return record
    
    def _check_compliance_flags(self, 
                                input_features: Dict[str, Any], 
                                output: Dict[str, Any]) -> List[str]:
        """Check for potential compliance issues."""
        flags = []
        
        # Check for extreme risk scores
        risk_score = output.get('risk_score', 0)
        if risk_score > 0.8:
            flags.append("HIGH_RISK_SCORE")
        
        # Check for premium outliers
        premium = output.get('premium', 0)
        if premium > 2000:
            flags.append("HIGH_PREMIUM_OUTLIER")
        if premium < 200:
            flags.append("LOW_PREMIUM_OUTLIER")
        
        # Check protected characteristics impact
        age = input_features.get('AGE', 0)
        if age < 25 and premium > 1200:
            flags.append("YOUNG_DRIVER_HIGH_PREMIUM")
        if age > 70 and premium > 1000:
            flags.append("ELDERLY_DRIVER_REVIEW")
        
        # Check for missing explanations
        if not output.get('explanation') and risk_score > 0.5:
            flags.append("MISSING_EXPLANATION_HIGH_RISK")
        
        return flags
    
    def _build_data_lineage(self, input_features: Dict[str, Any]) -> Dict[str, str]:
        """Build data lineage for input features."""
        lineage = {}
        
        feature_to_source = {
            'AGE': 'customer_data',
            'GENDER': 'customer_data',
            'CREDIT_SCORE': 'credit_score',
            'VEHICLE_TYPE': 'vehicle_data',
            'ANNUAL_MILEAGE': 'customer_data',
            'PAST_ACCIDENTS': 'claims_history',
            'SPEEDING_VIOLATIONS': 'external_risk',
            'DUIS': 'external_risk'
        }
        
        for feature in input_features.keys():
            source_key = feature_to_source.get(feature, 'customer_data')
            source_info = self.data_sources.get(source_key, {})
            lineage[feature] = f"{source_info.get('source', 'Unknown')} ({source_info.get('classification', 'N/A')})"
        
        return lineage
    
    def calculate_fairness_metrics(self, 
                                   df: pd.DataFrame,
                                   protected_col: str,
                                   outcome_col: str = 'premium') -> List[FairnessMetric]:
        """
        Calculate comprehensive fairness metrics for a protected characteristic.
        
        Implements:
        - Demographic Parity
        - Disparate Impact Ratio
        - Equalized Odds (where applicable)
        """
        metrics = []
        groups = df[protected_col].unique()
        
        if len(groups) < 2:
            return metrics
        
        # For each pair of groups
        for i, group_a in enumerate(groups):
            for group_b in groups[i+1:]:
                df_a = df[df[protected_col] == group_a]
                df_b = df[df[protected_col] == group_b]
                
                if len(df_a) < 10 or len(df_b) < 10:
                    continue
                
                # 1. Demographic Parity (mean outcome difference)
                mean_a = df_a[outcome_col].mean()
                mean_b = df_b[outcome_col].mean()
                parity_diff = abs(mean_a - mean_b) / max(mean_a, mean_b)
                
                parity_status = ComplianceStatus.COMPLIANT if parity_diff < self.demographic_parity_threshold \
                    else ComplianceStatus.WARNING if parity_diff < self.demographic_parity_threshold * 2 \
                    else ComplianceStatus.NON_COMPLIANT
                
                metrics.append(FairnessMetric(
                    characteristic=protected_col,
                    group_a=str(group_a),
                    group_b=str(group_b),
                    metric_name="Demographic Parity",
                    metric_value=round(parity_diff, 4),
                    threshold=self.demographic_parity_threshold,
                    status=parity_status,
                    timestamp=datetime.now()
                ))
                
                # 2. Disparate Impact Ratio (80% rule)
                # Favorable outcome: lower premium
                favorable_threshold = df[outcome_col].median()
                favorable_a = (df_a[outcome_col] < favorable_threshold).mean()
                favorable_b = (df_b[outcome_col] < favorable_threshold).mean()
                
                if favorable_b > 0:
                    di_ratio = favorable_a / favorable_b
                else:
                    di_ratio = 1.0
                
                # Ensure ratio is <= 1 (flip if needed)
                if di_ratio > 1:
                    di_ratio = 1 / di_ratio
                
                di_status = ComplianceStatus.COMPLIANT if di_ratio >= self.disparate_impact_threshold \
                    else ComplianceStatus.WARNING if di_ratio >= self.disparate_impact_threshold * 0.9 \
                    else ComplianceStatus.NON_COMPLIANT
                
                metrics.append(FairnessMetric(
                    characteristic=protected_col,
                    group_a=str(group_a),
                    group_b=str(group_b),
                    metric_name="Disparate Impact Ratio",
                    metric_value=round(di_ratio, 4),
                    threshold=self.disparate_impact_threshold,
                    status=di_status,
                    timestamp=datetime.now()
                ))
                
                # 3. Statistical Parity Difference
                stat_parity = abs(favorable_a - favorable_b)
                
                sp_status = ComplianceStatus.COMPLIANT if stat_parity < 0.1 \
                    else ComplianceStatus.WARNING if stat_parity < 0.2 \
                    else ComplianceStatus.NON_COMPLIANT
                
                metrics.append(FairnessMetric(
                    characteristic=protected_col,
                    group_a=str(group_a),
                    group_b=str(group_b),
                    metric_name="Statistical Parity Difference",
                    metric_value=round(stat_parity, 4),
                    threshold=0.1,
                    status=sp_status,
                    timestamp=datetime.now()
                ))
        
        self.fairness_metrics.extend(metrics)
        return metrics
    
    def detect_model_drift(self, current_metrics: Dict[str, float]) -> List[ModelDriftAlert]:
        """
        Detect model drift by comparing current metrics to baseline.
        
        Returns alerts for significant drifts.
        """
        alerts = []
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            current_value = current_metrics.get(metric_name)
            
            if current_value is None:
                continue
            
            # Calculate drift percentage
            if baseline_value != 0:
                drift_pct = abs(current_value - baseline_value) / baseline_value
            else:
                drift_pct = abs(current_value) if current_value != 0 else 0
            
            threshold = self.drift_thresholds.get(metric_name, 0.1)
            
            if drift_pct >= threshold:
                # Determine severity
                if drift_pct >= threshold * 3:
                    severity = "critical"
                    action = "Immediate model review and potential retraining required"
                elif drift_pct >= threshold * 2:
                    severity = "high"
                    action = "Schedule model review within 7 days"
                elif drift_pct >= threshold * 1.5:
                    severity = "medium"
                    action = "Monitor closely, schedule review within 30 days"
                else:
                    severity = "low"
                    action = "Continue monitoring, no immediate action required"
                
                alert = ModelDriftAlert(
                    alert_id=f"DFT-{datetime.now().strftime('%Y%m%d')}-{metric_name}",
                    timestamp=datetime.now(),
                    metric_name=metric_name,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    drift_percentage=round(drift_pct * 100, 2),
                    severity=severity,
                    recommended_action=action
                )
                
                alerts.append(alert)
        
        self.drift_alerts.extend(alerts)
        return alerts
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive compliance report."""
        
        # Analyze fairness metrics
        fairness_summary = {
            'total_metrics': len(self.fairness_metrics),
            'compliant': sum(1 for m in self.fairness_metrics if m.status == ComplianceStatus.COMPLIANT),
            'warnings': sum(1 for m in self.fairness_metrics if m.status == ComplianceStatus.WARNING),
            'non_compliant': sum(1 for m in self.fairness_metrics if m.status == ComplianceStatus.NON_COMPLIANT)
        }
        
        # Analyze audit trail
        audit_summary = {
            'total_records': len(self.audit_trail),
            'flagged_records': sum(1 for r in self.audit_trail if r.compliance_flags),
            'flag_breakdown': {}
        }
        
        for record in self.audit_trail:
            for flag in record.compliance_flags:
                audit_summary['flag_breakdown'][flag] = audit_summary['flag_breakdown'].get(flag, 0) + 1
        
        # Analyze drift alerts
        drift_summary = {
            'total_alerts': len(self.drift_alerts),
            'critical': sum(1 for a in self.drift_alerts if a.severity == 'critical'),
            'high': sum(1 for a in self.drift_alerts if a.severity == 'high'),
            'medium': sum(1 for a in self.drift_alerts if a.severity == 'medium'),
            'low': sum(1 for a in self.drift_alerts if a.severity == 'low')
        }
        
        # Overall compliance status
        if fairness_summary['non_compliant'] > 0 or drift_summary['critical'] > 0:
            overall_status = ComplianceStatus.NON_COMPLIANT
        elif fairness_summary['warnings'] > 2 or drift_summary['high'] > 0:
            overall_status = ComplianceStatus.WARNING
        else:
            overall_status = ComplianceStatus.COMPLIANT
        
        return {
            'report_id': f"CMP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'model_metadata': self.model_metadata,
            'overall_status': overall_status.value,
            'fairness_summary': fairness_summary,
            'audit_summary': audit_summary,
            'drift_summary': drift_summary,
            'regulatory_frameworks': ['FCA PRIN', 'GDPR Article 22', 'Solvency II'],
            'next_review_date': (datetime.now() + timedelta(days=90)).isoformat()
        }
    
    def get_model_documentation(self) -> Dict[str, Any]:
        """
        Generate SR 11-7 style model documentation.
        
        SR 11-7 is the Federal Reserve's guidance on model risk management.
        """
        return {
            'document_type': 'Model Risk Management Documentation',
            'standard': 'SR 11-7 / SS1/23 (PRA)',
            'generated_at': datetime.now().isoformat(),
            
            'model_inventory': {
                'model_id': self.model_metadata['model_id'],
                'model_name': self.model_metadata['model_name'],
                'model_type': 'Supervised Classification (Random Forest)',
                'risk_tier': self.model_metadata['risk_tier'],
                'business_use': 'Motor Insurance Risk Scoring and Premium Calculation',
                'materiality': 'High - Directly impacts pricing decisions'
            },
            
            'development': {
                'methodology': 'Ensemble learning with bootstrap aggregation',
                'training_data': 'Historical claims data (2020-2024)',
                'features': ['Age', 'Gender', 'Vehicle Type', 'Mileage', 'Credit Score', 
                            'Driving History', 'Region', 'Coverage Level'],
                'target_variable': 'Claim occurrence (binary)',
                'validation_approach': '5-fold cross-validation with holdout test set',
                'performance_metrics': {
                    'AUC-ROC': 0.654,
                    'Gini Coefficient': 0.308,
                    'Precision': 0.72,
                    'Recall': 0.68
                }
            },
            
            'implementation': {
                'deployment_date': self.model_metadata['deployed_date'].isoformat(),
                'runtime_environment': 'Python 3.11 / FastAPI',
                'integration_points': ['Quote Engine', 'Underwriting System', 'CRM'],
                'fallback_mechanism': 'Manual underwriting referral',
                'monitoring_frequency': 'Daily performance checks, monthly deep review'
            },
            
            'validation': {
                'last_validation': self.model_metadata['last_validated'].isoformat(),
                'validation_type': 'Independent model validation',
                'validator': 'Internal Model Validation Team',
                'findings': 'No material issues identified',
                'next_validation': (self.model_metadata['last_validated'] + timedelta(days=365)).isoformat()
            },
            
            'governance': {
                'model_owner': self.model_metadata['owner'],
                'approver': self.model_metadata['approver'],
                'review_committee': 'Model Risk Committee',
                'escalation_path': 'Chief Actuary → CRO → Board Risk Committee'
            },
            
            'limitations': [
                'Model trained on UK data only - not suitable for international markets',
                'Limited data for very young (<18) and very old (>85) drivers',
                'Does not account for telematics data',
                'Credit score may not be available for all customers'
            ],
            
            'change_log': [
                {'version': '1.0', 'date': '2024-01-15', 'change': 'Initial deployment'},
                {'version': '2.0', 'date': '2024-07-01', 'change': 'Added credit score feature'},
                {'version': '2.1', 'date': '2025-12-01', 'change': 'Retraining with 2024 data'}
            ]
        }
    
    def get_gdpr_data_map(self) -> Dict[str, Any]:
        """Generate GDPR Article 30 compliant data processing record."""
        return {
            'record_type': 'GDPR Article 30 - Records of Processing Activities',
            'controller': 'InsurePrice Ltd',
            'dpo_contact': 'dpo@insureprice.com',
            'generated_at': datetime.now().isoformat(),
            
            'processing_activities': [
                {
                    'activity': 'Risk Assessment',
                    'purpose': 'Calculate insurance risk score for premium pricing',
                    'legal_basis': 'Contract performance (Article 6(1)(b))',
                    'data_subjects': 'Insurance applicants and policyholders',
                    'data_categories': ['Identity', 'Contact', 'Financial', 'Driving History'],
                    'special_categories': 'None processed',
                    'recipients': ['Underwriting System', 'Reinsurers (aggregated)'],
                    'transfers': 'None outside EEA',
                    'retention': '7 years from policy end',
                    'security_measures': ['Encryption at rest', 'TLS in transit', 'Access controls']
                },
                {
                    'activity': 'Automated Decision Making',
                    'purpose': 'Generate insurance quotes without human intervention',
                    'legal_basis': 'Contract performance + Explicit consent',
                    'safeguards': [
                        'Right to obtain human intervention',
                        'Right to express point of view',
                        'Right to contest decision',
                        'Explanation of logic provided on request'
                    ],
                    'profiling': 'Yes - risk profiling based on statistical models'
                }
            ],
            
            'data_subject_rights': {
                'access': 'Automated via customer portal',
                'rectification': 'Via customer service within 30 days',
                'erasure': 'Available after legal retention period',
                'portability': 'JSON/CSV export available',
                'objection': 'Via DPO email, processed within 30 days'
            }
        }


def main():
    """Demonstration of Regulatory Compliance Engine."""
    
    print("📋 REGULATORY COMPLIANCE ENGINE")
    print("=" * 60)
    
    engine = RegulatoryComplianceEngine()
    
    # Simulate some audit records
    print("\n📝 Creating Audit Records...")
    for i in range(5):
        record = engine.create_audit_record(
            policy_id=f"POL-2025-{1000+i}",
            customer_id=f"CUST-{5000+i}",
            action="quote",
            input_features={
                'AGE': 25 + i * 10,
                'GENDER': 'Male' if i % 2 == 0 else 'Female',
                'CREDIT_SCORE': 650 + i * 20,
                'VEHICLE_TYPE': 'Sedan',
                'ANNUAL_MILEAGE': 10000 + i * 2000
            },
            output={
                'risk_score': 0.15 + i * 0.1,
                'premium': 500 + i * 150,
                'explanation': {'top_factor': 'Age'}
            },
            explanation={'shap_values': {'AGE': 0.05}}
        )
        print(f"   Created: {record.record_id} - Flags: {record.compliance_flags}")
    
    # Calculate fairness metrics
    print("\n⚖️ Calculating Fairness Metrics...")
    sample_data = pd.DataFrame({
        'AGE_GROUP': ['18-25'] * 100 + ['26-40'] * 150 + ['41-60'] * 200 + ['60+'] * 50,
        'GENDER': np.random.choice(['Male', 'Female'], 500),
        'premium': np.random.normal(650, 150, 500)
    })
    
    age_metrics = engine.calculate_fairness_metrics(sample_data, 'AGE_GROUP', 'premium')
    gender_metrics = engine.calculate_fairness_metrics(sample_data, 'GENDER', 'premium')
    
    print(f"   Age fairness metrics: {len(age_metrics)}")
    print(f"   Gender fairness metrics: {len(gender_metrics)}")
    
    # Detect model drift
    print("\n📊 Checking for Model Drift...")
    current_metrics = {
        'auc': 0.62,  # Dropped from 0.654
        'gini': 0.29,  # Dropped from 0.308
        'conversion_rate': 0.11,
        'claim_frequency': 0.13,
        'average_premium': 680
    }
    
    alerts = engine.detect_model_drift(current_metrics)
    for alert in alerts:
        print(f"   ⚠️ {alert.severity.upper()}: {alert.metric_name} drifted {alert.drift_percentage}%")
    
    # Generate compliance report
    print("\n📄 Generating Compliance Report...")
    report = engine.generate_compliance_report()
    print(f"   Report ID: {report['report_id']}")
    print(f"   Overall Status: {report['overall_status'].upper()}")
    print(f"   Fairness: {report['fairness_summary']['compliant']}/{report['fairness_summary']['total_metrics']} compliant")
    print(f"   Drift Alerts: {report['drift_summary']['total_alerts']} ({report['drift_summary']['high']} high)")
    
    # Model documentation
    print("\n📚 Model Documentation (SR 11-7)...")
    docs = engine.get_model_documentation()
    print(f"   Model: {docs['model_inventory']['model_name']}")
    print(f"   Risk Tier: {docs['model_inventory']['risk_tier']}")
    print(f"   Last Validated: {docs['validation']['last_validation']}")
    
    print("\n" + "=" * 60)
    print("✅ Regulatory Compliance Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()


