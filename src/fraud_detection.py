"""
InsurePrice Real-Time Fraud Detection Module

Advanced claims fraud detection system featuring:
- Anomaly detection on claim patterns
- Network analysis for connected fraud rings
- NLP analysis on claim descriptions for red flags
- Real-time scoring and alerts
- Integration with claims workflow

UK Insurance Fraud Context:
- Annual fraud cost: £1.2 billion
- 5% improvement = £60 million savings potential
- Focus areas: staged accidents, inflated claims, ghost brokers

Author: Masood Nazari
Business Intelligence Analyst | Data Science | AI | Clinical Research
Email: M.Nazari@soton.ac.uk
Portfolio: https://michaeltheanalyst.github.io/
LinkedIn: linkedin.com/in/masood-nazari
GitHub: github.com/michaeltheanalyst
Date: December 2025
Project: InsurePrice Car Insurance Risk Modeling
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

# Fraud red flag keywords for NLP analysis
FRAUD_KEYWORDS = {
    'high_risk': [
        'whiplash', 'neck pain', 'back injury', 'soft tissue',
        'hit and run', 'unwitnessed', 'total loss', 'write off',
        'cash settlement', 'urgent', 'immediate payment'
    ],
    'medium_risk': [
        'rear ended', 'parking lot', 'low speed', 'minor damage',
        'no witnesses', 'dark', 'night time', 'rain', 'wet road',
        'swerved', 'sudden stop', 'lost control'
    ],
    'suspicious_patterns': [
        'friend', 'family member', 'acquaintance', 'knew the other driver',
        'regular mechanic', 'preferred garage', 'cash only',
        'no receipt', 'estimate only', 'approximate'
    ]
}


class FraudDetectionEngine:
    """
    Comprehensive fraud detection system for car insurance claims.
    
    Implements multiple detection strategies:
    1. Statistical anomaly detection
    2. Network analysis for fraud rings
    3. NLP text analysis for red flags
    4. Behavioral pattern recognition
    """
    
    def __init__(self):
        """Initialize fraud detection engine."""
        self.anomaly_detector = None
        self.text_vectorizer = None
        self.fraud_network = nx.Graph()
        self.claim_history = []
        self.fraud_patterns = defaultdict(list)
        self.scaler = StandardScaler()
        
        print("🔍 FRAUD DETECTION ENGINE INITIALIZED")
        print("=" * 60)
        print("UK Insurance Fraud Context:")
        print("  • Annual fraud cost: £1.2 billion")
        print("  • Target: 5% improvement = £60 million savings")
        print("  • Methods: Anomaly detection, Network analysis, NLP")
        print("=" * 60)
    
    def train_anomaly_detector(self, historical_claims: pd.DataFrame):
        """
        Train anomaly detection model on historical claims data.
        
        Uses Isolation Forest to identify unusual claim patterns.
        """
        print("\n📊 Training Anomaly Detection Model...")
        
        # Select numerical features for anomaly detection
        numerical_features = [
            'claim_amount', 'days_to_report', 'vehicle_age',
            'driver_age', 'annual_mileage', 'previous_claims',
            'policy_age_days', 'time_since_last_claim'
        ]
        
        # Filter available features
        available_features = [f for f in numerical_features if f in historical_claims.columns]
        
        if len(available_features) < 3:
            print("⚠️ Insufficient features for anomaly detection")
            return False
        
        X = historical_claims[available_features].copy()
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=0.05,  # Expected fraud rate ~5%
            random_state=42,
            n_estimators=100
        )
        self.anomaly_detector.fit(X_scaled)
        
        print(f"✅ Anomaly detector trained on {len(X)} claims")
        print(f"   Features used: {available_features}")
        
        return True
    
    def detect_anomalies(self, claim_data: dict) -> dict:
        """
        Detect anomalies in a single claim.
        
        Returns anomaly score and contributing factors.
        """
        if self.anomaly_detector is None:
            return {'anomaly_score': 0.5, 'is_anomaly': False, 'factors': []}
        
        # Prepare features
        features = self._extract_numerical_features(claim_data)
        X = np.array([list(features.values())])
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly score (-1 = anomaly, 1 = normal)
        raw_score = self.anomaly_detector.decision_function(X_scaled)[0]
        prediction = self.anomaly_detector.predict(X_scaled)[0]
        
        # Convert to 0-1 fraud probability (higher = more suspicious)
        fraud_score = 1 - (raw_score + 0.5)  # Normalize
        fraud_score = max(0, min(1, fraud_score))
        
        # Identify contributing factors
        factors = self._identify_anomaly_factors(claim_data, features)
        
        return {
            'anomaly_score': round(fraud_score, 3),
            'is_anomaly': prediction == -1,
            'factors': factors,
            'raw_isolation_score': round(raw_score, 3)
        }
    
    def _extract_numerical_features(self, claim_data: dict) -> dict:
        """Extract numerical features from claim data."""
        return {
            'claim_amount': claim_data.get('claim_amount', 0),
            'days_to_report': claim_data.get('days_to_report', 0),
            'vehicle_age': claim_data.get('vehicle_age', 5),
            'driver_age': claim_data.get('driver_age', 35),
            'annual_mileage': claim_data.get('annual_mileage', 10000),
            'previous_claims': claim_data.get('previous_claims', 0),
            'policy_age_days': claim_data.get('policy_age_days', 365),
            'time_since_last_claim': claim_data.get('time_since_last_claim', 730)
        }
    
    def _identify_anomaly_factors(self, claim_data: dict, features: dict) -> list:
        """Identify factors contributing to anomaly score."""
        factors = []
        
        # High claim amount
        if features['claim_amount'] > 5000:
            factors.append({
                'factor': 'High Claim Amount',
                'value': f"£{features['claim_amount']:,.0f}",
                'risk_level': 'high' if features['claim_amount'] > 10000 else 'medium'
            })
        
        # Quick reporting (potential staged accident)
        if features['days_to_report'] == 0:
            factors.append({
                'factor': 'Same-Day Reporting',
                'value': 'Reported immediately',
                'risk_level': 'medium'
            })
        
        # Late reporting (potential fraud attempt)
        if features['days_to_report'] > 30:
            factors.append({
                'factor': 'Delayed Reporting',
                'value': f"{features['days_to_report']} days",
                'risk_level': 'high'
            })
        
        # New policy claims
        if features['policy_age_days'] < 90:
            factors.append({
                'factor': 'New Policy Claim',
                'value': f"Policy age: {features['policy_age_days']} days",
                'risk_level': 'high'
            })
        
        # Multiple previous claims
        if features['previous_claims'] >= 3:
            factors.append({
                'factor': 'Multiple Previous Claims',
                'value': f"{features['previous_claims']} claims",
                'risk_level': 'high'
            })
        
        # Recent claim history
        if features['time_since_last_claim'] < 180:
            factors.append({
                'factor': 'Recent Previous Claim',
                'value': f"{features['time_since_last_claim']} days ago",
                'risk_level': 'medium'
            })
        
        return factors
    
    def analyze_claim_text(self, claim_description: str) -> dict:
        """
        NLP analysis of claim description for fraud red flags.
        
        Analyzes text for suspicious patterns, keywords, and inconsistencies.
        """
        if not claim_description:
            return {'text_risk_score': 0, 'red_flags': [], 'keyword_matches': {}}
        
        text_lower = claim_description.lower()
        risk_score = 0
        red_flags = []
        keyword_matches = defaultdict(list)
        
        # Check for high-risk keywords
        for keyword in FRAUD_KEYWORDS['high_risk']:
            if keyword in text_lower:
                risk_score += 0.15
                keyword_matches['high_risk'].append(keyword)
                red_flags.append({
                    'type': 'High Risk Keyword',
                    'keyword': keyword,
                    'severity': 'high'
                })
        
        # Check for medium-risk keywords
        for keyword in FRAUD_KEYWORDS['medium_risk']:
            if keyword in text_lower:
                risk_score += 0.08
                keyword_matches['medium_risk'].append(keyword)
                red_flags.append({
                    'type': 'Medium Risk Keyword',
                    'keyword': keyword,
                    'severity': 'medium'
                })
        
        # Check for suspicious patterns
        for pattern in FRAUD_KEYWORDS['suspicious_patterns']:
            if pattern in text_lower:
                risk_score += 0.12
                keyword_matches['suspicious'].append(pattern)
                red_flags.append({
                    'type': 'Suspicious Pattern',
                    'keyword': pattern,
                    'severity': 'high'
                })
        
        # Check for vague language
        vague_patterns = [
            r'\b(approximately|about|around|roughly|maybe)\b',
            r'\b(i think|i believe|possibly|probably)\b',
            r'\b(cant remember|dont recall|not sure)\b'
        ]
        
        for pattern in vague_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                risk_score += 0.05 * len(matches)
                red_flags.append({
                    'type': 'Vague Language',
                    'keyword': matches[0] if matches else 'vague terms',
                    'severity': 'low'
                })
        
        # Check for excessive detail (potential rehearsed story)
        word_count = len(text_lower.split())
        if word_count > 200:
            risk_score += 0.1
            red_flags.append({
                'type': 'Excessive Detail',
                'keyword': f'{word_count} words',
                'severity': 'medium'
            })
        
        # Normalize score
        risk_score = min(1.0, risk_score)
        
        return {
            'text_risk_score': round(risk_score, 3),
            'red_flags': red_flags,
            'keyword_matches': dict(keyword_matches),
            'word_count': word_count
        }
    
    def build_fraud_network(self, claims_data: pd.DataFrame):
        """
        Build network graph to detect connected fraud rings.
        
        Analyzes connections between:
        - Same phone numbers
        - Same addresses
        - Same garage/repair shop
        - Same witnesses
        - Same accident locations
        """
        print("\n🕸️ Building Fraud Network Graph...")
        
        self.fraud_network = nx.Graph()
        
        # Group claims by potential connection points
        connection_fields = ['phone', 'address', 'garage', 'witness_name', 'accident_location']
        
        for field in connection_fields:
            if field in claims_data.columns:
                # Group claims with same value
                groups = claims_data.groupby(field)['claim_id'].apply(list)
                
                for value, claim_ids in groups.items():
                    if len(claim_ids) > 1 and pd.notna(value):
                        # Add edges between connected claims
                        for i, claim1 in enumerate(claim_ids):
                            for claim2 in claim_ids[i+1:]:
                                if self.fraud_network.has_edge(claim1, claim2):
                                    self.fraud_network[claim1][claim2]['connections'].append(field)
                                    self.fraud_network[claim1][claim2]['weight'] += 1
                                else:
                                    self.fraud_network.add_edge(
                                        claim1, claim2,
                                        connections=[field],
                                        weight=1
                                    )
        
        print(f"✅ Network built: {self.fraud_network.number_of_nodes()} claims, {self.fraud_network.number_of_edges()} connections")
        
        return self.fraud_network
    
    def detect_fraud_rings(self, min_cluster_size: int = 3) -> list:
        """
        Identify potential fraud rings from network analysis.
        
        Returns clusters of connected claims that may indicate organized fraud.
        """
        if self.fraud_network.number_of_nodes() == 0:
            return []
        
        fraud_rings = []
        
        # Find connected components
        for component in nx.connected_components(self.fraud_network):
            if len(component) >= min_cluster_size:
                subgraph = self.fraud_network.subgraph(component)
                
                # Calculate ring risk score
                total_weight = sum(data['weight'] for _, _, data in subgraph.edges(data=True))
                connection_types = set()
                for _, _, data in subgraph.edges(data=True):
                    connection_types.update(data['connections'])
                
                risk_score = min(1.0, (total_weight / len(component)) * 0.3 + len(connection_types) * 0.1)
                
                fraud_rings.append({
                    'ring_id': f'ring_{len(fraud_rings) + 1}',
                    'claim_ids': list(component),
                    'size': len(component),
                    'total_connections': total_weight,
                    'connection_types': list(connection_types),
                    'risk_score': round(risk_score, 3),
                    'recommendation': 'Investigate' if risk_score > 0.5 else 'Monitor'
                })
        
        # Sort by risk score
        fraud_rings.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return fraud_rings
    
    def calculate_fraud_score(self, claim_data: dict) -> dict:
        """
        Calculate comprehensive fraud score combining all detection methods.
        
        Returns overall fraud probability with breakdown by detection method.
        """
        # Get individual scores
        anomaly_result = self.detect_anomalies(claim_data)
        text_result = self.analyze_claim_text(claim_data.get('description', ''))
        
        # Calculate network risk (if claim is in network)
        network_risk = 0
        claim_id = claim_data.get('claim_id')
        if claim_id and self.fraud_network.has_node(claim_id):
            neighbors = list(self.fraud_network.neighbors(claim_id))
            if neighbors:
                network_risk = min(1.0, len(neighbors) * 0.15)
        
        # Calculate behavioral score
        behavioral_score = self._calculate_behavioral_score(claim_data)
        
        # Weighted combination
        weights = {
            'anomaly': 0.30,
            'text_analysis': 0.25,
            'network': 0.25,
            'behavioral': 0.20
        }
        
        overall_score = (
            anomaly_result['anomaly_score'] * weights['anomaly'] +
            text_result['text_risk_score'] * weights['text_analysis'] +
            network_risk * weights['network'] +
            behavioral_score * weights['behavioral']
        )
        
        # Determine risk level
        if overall_score >= 0.7:
            risk_level = 'CRITICAL'
            recommendation = 'Immediate investigation required'
        elif overall_score >= 0.5:
            risk_level = 'HIGH'
            recommendation = 'Refer to Special Investigation Unit (SIU)'
        elif overall_score >= 0.3:
            risk_level = 'MEDIUM'
            recommendation = 'Enhanced review recommended'
        else:
            risk_level = 'LOW'
            recommendation = 'Standard processing'
        
        return {
            'overall_fraud_score': round(overall_score, 3),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'component_scores': {
                'anomaly_detection': round(anomaly_result['anomaly_score'], 3),
                'text_analysis': round(text_result['text_risk_score'], 3),
                'network_analysis': round(network_risk, 3),
                'behavioral_analysis': round(behavioral_score, 3)
            },
            'weights_applied': weights,
            'anomaly_factors': anomaly_result['factors'],
            'text_red_flags': text_result['red_flags'],
            'network_connections': len(self.fraud_network.neighbors(claim_id)) if claim_id and self.fraud_network.has_node(claim_id) else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_behavioral_score(self, claim_data: dict) -> float:
        """Calculate behavioral risk score based on claim patterns."""
        score = 0
        
        # Time of accident (late night = higher risk)
        accident_time = claim_data.get('accident_time', '12:00')
        if accident_time:
            try:
                hour = int(accident_time.split(':')[0])
                if 0 <= hour <= 5 or 22 <= hour <= 23:
                    score += 0.15  # Late night accident
            except:
                pass
        
        # Day of week (weekends = slightly higher risk)
        accident_day = claim_data.get('accident_day', 'Monday')
        if accident_day in ['Saturday', 'Sunday']:
            score += 0.08
        
        # Single vehicle accident
        if claim_data.get('single_vehicle', False):
            score += 0.1
        
        # No police report
        if not claim_data.get('police_report', True):
            score += 0.15
        
        # No witnesses
        if claim_data.get('witnesses', 1) == 0:
            score += 0.12
        
        # Third-party injuries (whiplash fraud)
        if claim_data.get('third_party_injuries', 0) > 0:
            score += 0.1
        
        # Cash settlement requested
        if claim_data.get('cash_settlement_requested', False):
            score += 0.2
        
        return min(1.0, score)
    
    def generate_fraud_report(self, claim_data: dict) -> dict:
        """
        Generate comprehensive fraud assessment report for a claim.
        """
        fraud_score = self.calculate_fraud_score(claim_data)
        
        report = {
            'report_id': f"FR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'claim_id': claim_data.get('claim_id', 'UNKNOWN'),
            'generated_at': datetime.now().isoformat(),
            'fraud_assessment': fraud_score,
            'executive_summary': self._generate_executive_summary(fraud_score),
            'detailed_analysis': {
                'anomaly_detection': {
                    'method': 'Isolation Forest',
                    'score': fraud_score['component_scores']['anomaly_detection'],
                    'factors': fraud_score['anomaly_factors']
                },
                'text_analysis': {
                    'method': 'NLP Keyword & Pattern Detection',
                    'score': fraud_score['component_scores']['text_analysis'],
                    'red_flags': fraud_score['text_red_flags']
                },
                'network_analysis': {
                    'method': 'Graph-Based Connection Analysis',
                    'score': fraud_score['component_scores']['network_analysis'],
                    'connections': fraud_score['network_connections']
                },
                'behavioral_analysis': {
                    'method': 'Pattern Recognition',
                    'score': fraud_score['component_scores']['behavioral_analysis']
                }
            },
            'recommended_actions': self._generate_recommended_actions(fraud_score),
            'confidence_level': self._calculate_confidence(fraud_score)
        }
        
        return report
    
    def _generate_executive_summary(self, fraud_score: dict) -> str:
        """Generate executive summary for fraud report."""
        risk_level = fraud_score['risk_level']
        overall_score = fraud_score['overall_fraud_score']
        
        if risk_level == 'CRITICAL':
            return f"CRITICAL ALERT: This claim shows extremely high fraud indicators (score: {overall_score:.1%}). Multiple detection methods flagged suspicious patterns. Immediate SIU investigation strongly recommended before any payment authorization."
        elif risk_level == 'HIGH':
            return f"HIGH RISK: Significant fraud indicators detected (score: {overall_score:.1%}). The claim exhibits patterns consistent with known fraud schemes. Recommend detailed investigation and additional documentation before processing."
        elif risk_level == 'MEDIUM':
            return f"MODERATE RISK: Some unusual patterns detected (score: {overall_score:.1%}). While not conclusive, enhanced verification is advisable. Consider requesting additional evidence or conducting follow-up interviews."
        else:
            return f"LOW RISK: Claim appears legitimate based on available data (score: {overall_score:.1%}). Standard processing procedures apply. Routine verification recommended."
    
    def _generate_recommended_actions(self, fraud_score: dict) -> list:
        """Generate list of recommended actions based on fraud assessment."""
        actions = []
        risk_level = fraud_score['risk_level']
        
        if risk_level in ['CRITICAL', 'HIGH']:
            actions.extend([
                {'priority': 'HIGH', 'action': 'Refer to Special Investigation Unit (SIU)'},
                {'priority': 'HIGH', 'action': 'Request additional documentation and evidence'},
                {'priority': 'HIGH', 'action': 'Conduct claimant interview'},
                {'priority': 'MEDIUM', 'action': 'Verify third-party information independently'}
            ])
        
        if fraud_score['component_scores']['text_analysis'] > 0.3:
            actions.append({
                'priority': 'MEDIUM',
                'action': 'Review claim description for inconsistencies'
            })
        
        if fraud_score['component_scores']['network_analysis'] > 0.2:
            actions.append({
                'priority': 'HIGH',
                'action': 'Investigate connected claims for fraud ring patterns'
            })
        
        if fraud_score['anomaly_factors']:
            actions.append({
                'priority': 'MEDIUM',
                'action': f"Verify flagged anomalies: {', '.join([f['factor'] for f in fraud_score['anomaly_factors'][:3]])}"
            })
        
        if not actions:
            actions.append({
                'priority': 'LOW',
                'action': 'Process claim following standard procedures'
            })
        
        return actions
    
    def _calculate_confidence(self, fraud_score: dict) -> str:
        """Calculate confidence level of fraud assessment."""
        # More detection methods flagging = higher confidence
        flagged_methods = sum(1 for score in fraud_score['component_scores'].values() if score > 0.3)
        
        if flagged_methods >= 3:
            return 'HIGH'
        elif flagged_methods >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'


def demonstrate_fraud_detection():
    """Demonstrate fraud detection capabilities with sample data."""
    
    print("\n" + "=" * 70)
    print("🔍 INSUREPRICE FRAUD DETECTION DEMONSTRATION")
    print("=" * 70)
    
    # Initialize engine
    engine = FraudDetectionEngine()
    
    # Create sample historical claims for training
    print("\n📊 Generating sample claims data for training...")
    
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
    
    # Train anomaly detector
    engine.train_anomaly_detector(historical_claims)
    
    # Test Case 1: Suspicious claim
    print("\n" + "-" * 50)
    print("🚨 TEST CASE 1: Suspicious Claim (High Risk)")
    print("-" * 50)
    
    suspicious_claim = {
        'claim_id': 'TEST-001',
        'claim_amount': 12500,
        'days_to_report': 45,
        'vehicle_age': 8,
        'driver_age': 23,
        'annual_mileage': 25000,
        'previous_claims': 4,
        'policy_age_days': 60,
        'time_since_last_claim': 90,
        'description': 'I was rear ended at low speed in a parking lot at night. No witnesses. I suffered whiplash and severe neck pain. My friend recommended a garage for repairs. I would prefer a cash settlement for immediate payment.',
        'accident_time': '23:30',
        'accident_day': 'Saturday',
        'police_report': False,
        'witnesses': 0,
        'cash_settlement_requested': True,
        'third_party_injuries': 2
    }
    
    report = engine.generate_fraud_report(suspicious_claim)
    
    print(f"\n📋 FRAUD ASSESSMENT REPORT")
    print(f"Claim ID: {report['claim_id']}")
    print(f"Report ID: {report['report_id']}")
    print(f"\n🎯 Overall Fraud Score: {report['fraud_assessment']['overall_fraud_score']:.1%}")
    print(f"Risk Level: {report['fraud_assessment']['risk_level']}")
    print(f"Confidence: {report['confidence_level']}")
    
    print(f"\n📊 Component Scores:")
    for component, score in report['fraud_assessment']['component_scores'].items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {component:20s}: {bar} {score:.1%}")
    
    print(f"\n📝 Executive Summary:")
    print(f"  {report['executive_summary']}")
    
    print(f"\n⚡ Recommended Actions:")
    for action in report['recommended_actions']:
        print(f"  [{action['priority']}] {action['action']}")
    
    # Test Case 2: Legitimate claim
    print("\n" + "-" * 50)
    print("✅ TEST CASE 2: Legitimate Claim (Low Risk)")
    print("-" * 50)
    
    legitimate_claim = {
        'claim_id': 'TEST-002',
        'claim_amount': 2500,
        'days_to_report': 2,
        'vehicle_age': 3,
        'driver_age': 45,
        'annual_mileage': 10000,
        'previous_claims': 0,
        'policy_age_days': 730,
        'time_since_last_claim': 2000,
        'description': 'Another vehicle collided with my car at a junction. Police attended and filed a report. The other driver admitted fault. Two independent witnesses provided statements.',
        'accident_time': '14:30',
        'accident_day': 'Tuesday',
        'police_report': True,
        'witnesses': 2,
        'cash_settlement_requested': False,
        'third_party_injuries': 0
    }
    
    report2 = engine.generate_fraud_report(legitimate_claim)
    
    print(f"\n📋 FRAUD ASSESSMENT REPORT")
    print(f"Claim ID: {report2['claim_id']}")
    print(f"Report ID: {report2['report_id']}")
    print(f"\n🎯 Overall Fraud Score: {report2['fraud_assessment']['overall_fraud_score']:.1%}")
    print(f"Risk Level: {report2['fraud_assessment']['risk_level']}")
    print(f"Confidence: {report2['confidence_level']}")
    
    print(f"\n📊 Component Scores:")
    for component, score in report2['fraud_assessment']['component_scores'].items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {component:20s}: {bar} {score:.1%}")
    
    print(f"\n📝 Executive Summary:")
    print(f"  {report2['executive_summary']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FRAUD DETECTION DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\n💼 Business Value:")
    print("  • UK insurance fraud costs: £1.2 billion annually")
    print("  • 5% fraud detection improvement = £60 million savings")
    print("  • Real-time scoring enables proactive investigation")
    print("  • Multi-method approach reduces false positives")
    
    print("\n🔍 Detection Methods Demonstrated:")
    print("  ✅ Anomaly Detection (Isolation Forest)")
    print("  ✅ NLP Text Analysis (Keyword & Pattern Detection)")
    print("  ✅ Behavioral Analysis (Pattern Recognition)")
    print("  ✅ Network Analysis Framework (Fraud Ring Detection)")
    
    print("\n🚀 Integration Ready:")
    print("  • REST API endpoint: POST /api/v1/fraud/analyze")
    print("  • Real-time claims workflow integration")
    print("  • SIU referral automation")
    print("  • Dashboard monitoring and alerts")
    
    return engine


if __name__ == "__main__":
    engine = demonstrate_fraud_detection()
