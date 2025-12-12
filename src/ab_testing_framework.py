"""
A/B Testing Framework for Insurance Pricing
============================================

Provides capabilities for:
- Price sensitivity analysis per segment
- Conversion rate tracking
- Statistical significance testing
- Experiment management

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
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    experiment_id: str
    name: str
    description: str
    control_price_modifier: float  # e.g., 1.0 for no change
    treatment_price_modifier: float  # e.g., 0.95 for 5% discount
    target_segment: str  # 'all', 'low_risk', 'high_risk', etc.
    start_date: datetime
    end_date: datetime
    min_sample_size: int
    confidence_level: float  # e.g., 0.95


@dataclass
class ExperimentResult:
    """Results from an A/B test experiment."""
    experiment_id: str
    control_conversions: int
    control_total: int
    treatment_conversions: int
    treatment_total: int
    control_revenue: float
    treatment_revenue: float
    p_value: float
    is_significant: bool
    lift: float
    confidence_interval: Tuple[float, float]


class ABTestingFramework:
    """
    A/B Testing Framework for Insurance Pricing Experiments.
    
    Enables data-driven pricing decisions through controlled experiments.
    """
    
    def __init__(self):
        """Initialize the A/B testing framework."""
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, ExperimentResult] = {}
        
        # Segment definitions
        self.segments = {
            'all': {'risk_range': (0, 1), 'description': 'All customers'},
            'low_risk': {'risk_range': (0, 0.2), 'description': 'Low risk customers'},
            'medium_risk': {'risk_range': (0.2, 0.35), 'description': 'Medium risk customers'},
            'high_risk': {'risk_range': (0.35, 0.5), 'description': 'High risk customers'},
            'very_high_risk': {'risk_range': (0.5, 1), 'description': 'Very high risk customers'},
            'young_drivers': {'age_group': '16-25', 'description': 'Young drivers 16-25'},
            'mature_drivers': {'age_group': '40-64', 'description': 'Mature drivers 40-64'},
            'urban': {'regions': ['London', 'West Midlands', 'North West'], 'description': 'Urban areas'},
            'rural': {'regions': ['South West', 'East Anglia', 'Wales'], 'description': 'Rural areas'}
        }
        
        # Base conversion rates by segment (simulated)
        self.base_conversion_rates = {
            'all': 0.12,
            'low_risk': 0.18,
            'medium_risk': 0.14,
            'high_risk': 0.10,
            'very_high_risk': 0.06,
            'young_drivers': 0.08,
            'mature_drivers': 0.16,
            'urban': 0.11,
            'rural': 0.13
        }
        
        # Price elasticity by segment
        self.price_elasticity = {
            'all': -0.8,
            'low_risk': -0.5,  # Less price sensitive
            'medium_risk': -0.8,
            'high_risk': -1.2,  # More price sensitive
            'very_high_risk': -1.5,
            'young_drivers': -1.3,
            'mature_drivers': -0.4,
            'urban': -0.9,
            'rural': -0.6
        }
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B test experiment."""
        self.experiments[config.experiment_id] = config
        return config.experiment_id
    
    def simulate_experiment(self, 
                           experiment_id: str,
                           sample_size: int = 1000,
                           base_premium: float = 650) -> ExperimentResult:
        """
        Simulate an A/B test experiment.
        
        In production, this would track real user behavior.
        For demonstration, we simulate based on price elasticity models.
        """
        config = self.experiments.get(experiment_id)
        if not config:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        segment = config.target_segment
        base_conversion = self.base_conversion_rates.get(segment, 0.12)
        elasticity = self.price_elasticity.get(segment, -0.8)
        
        # Split sample
        control_size = sample_size // 2
        treatment_size = sample_size - control_size
        
        # Control group (no price change)
        control_conversion_rate = base_conversion
        control_premium = base_premium * config.control_price_modifier
        
        # Treatment group (with price change)
        price_change_pct = (config.treatment_price_modifier - 1) * 100
        # Elasticity formula: % change in demand = elasticity * % change in price
        demand_change = elasticity * price_change_pct / 100
        treatment_conversion_rate = base_conversion * (1 + demand_change)
        treatment_conversion_rate = np.clip(treatment_conversion_rate, 0.01, 0.5)
        treatment_premium = base_premium * config.treatment_price_modifier
        
        # Simulate conversions with some randomness
        np.random.seed(42)
        control_conversions = np.random.binomial(control_size, control_conversion_rate)
        treatment_conversions = np.random.binomial(treatment_size, treatment_conversion_rate)
        
        # Calculate revenue
        control_revenue = control_conversions * control_premium
        treatment_revenue = treatment_conversions * treatment_premium
        
        # Statistical significance test (two-proportion z-test)
        p1 = control_conversions / control_size
        p2 = treatment_conversions / treatment_size
        
        # Pooled proportion
        p_pool = (control_conversions + treatment_conversions) / (control_size + treatment_size)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/control_size + 1/treatment_size))
        
        # Z-score
        if se > 0:
            z_score = (p2 - p1) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = 0
            p_value = 1.0
        
        # Confidence interval for the difference
        ci_margin = 1.96 * se
        ci_lower = (p2 - p1) - ci_margin
        ci_upper = (p2 - p1) + ci_margin
        
        # Lift calculation
        lift = ((p2 - p1) / p1 * 100) if p1 > 0 else 0
        
        # Is significant?
        is_significant = p_value < (1 - config.confidence_level)
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            control_conversions=control_conversions,
            control_total=control_size,
            treatment_conversions=treatment_conversions,
            treatment_total=treatment_size,
            control_revenue=round(control_revenue, 2),
            treatment_revenue=round(treatment_revenue, 2),
            p_value=round(p_value, 4),
            is_significant=is_significant,
            lift=round(lift, 2),
            confidence_interval=(round(ci_lower, 4), round(ci_upper, 4))
        )
        
        self.results[experiment_id] = result
        return result
    
    def analyze_price_sensitivity(self, 
                                  segment: str = 'all',
                                  price_range: Tuple[float, float] = (0.8, 1.2),
                                  steps: int = 9,
                                  base_premium: float = 650,
                                  sample_size: int = 1000) -> pd.DataFrame:
        """
        Analyze price sensitivity for a segment.
        
        Returns conversion rates and revenue at different price points.
        """
        base_conversion = self.base_conversion_rates.get(segment, 0.12)
        elasticity = self.price_elasticity.get(segment, -0.8)
        
        price_modifiers = np.linspace(price_range[0], price_range[1], steps)
        results = []
        
        for modifier in price_modifiers:
            price = base_premium * modifier
            price_change_pct = (modifier - 1) * 100
            
            # Calculate expected conversion rate
            demand_change = elasticity * price_change_pct / 100
            conversion_rate = base_conversion * (1 + demand_change)
            conversion_rate = np.clip(conversion_rate, 0.01, 0.5)
            
            # Expected conversions and revenue
            expected_conversions = sample_size * conversion_rate
            expected_revenue = expected_conversions * price
            revenue_per_visitor = expected_revenue / sample_size
            
            results.append({
                'price_modifier': round(modifier, 2),
                'price': round(price, 2),
                'price_change_pct': round(price_change_pct, 1),
                'conversion_rate': round(conversion_rate, 4),
                'expected_conversions': round(expected_conversions, 0),
                'expected_revenue': round(expected_revenue, 2),
                'revenue_per_visitor': round(revenue_per_visitor, 2)
            })
        
        return pd.DataFrame(results)
    
    def calculate_sample_size(self,
                             baseline_conversion: float,
                             minimum_detectable_effect: float,
                             confidence_level: float = 0.95,
                             power: float = 0.8) -> int:
        """
        Calculate required sample size for an A/B test.
        
        Uses standard power analysis for two-proportion test.
        """
        alpha = 1 - confidence_level
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_conversion
        p2 = baseline_conversion * (1 + minimum_detectable_effect)
        
        # Pooled standard deviation
        p_bar = (p1 + p2) / 2
        
        # Sample size per group
        n = (2 * p_bar * (1 - p_bar) * (z_alpha + z_beta)**2) / (p2 - p1)**2
        
        return int(np.ceil(n))
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of an experiment."""
        config = self.experiments.get(experiment_id)
        result = self.results.get(experiment_id)
        
        if not config or not result:
            return {}
        
        control_rate = result.control_conversions / result.control_total
        treatment_rate = result.treatment_conversions / result.treatment_total
        
        revenue_lift = ((result.treatment_revenue - result.control_revenue) / 
                       result.control_revenue * 100) if result.control_revenue > 0 else 0
        
        return {
            'experiment': {
                'id': config.experiment_id,
                'name': config.name,
                'description': config.description,
                'segment': config.target_segment,
                'control_modifier': config.control_price_modifier,
                'treatment_modifier': config.treatment_price_modifier
            },
            'results': {
                'control_conversion_rate': round(control_rate, 4),
                'treatment_conversion_rate': round(treatment_rate, 4),
                'conversion_lift_pct': result.lift,
                'control_revenue': result.control_revenue,
                'treatment_revenue': result.treatment_revenue,
                'revenue_lift_pct': round(revenue_lift, 2),
                'p_value': result.p_value,
                'is_significant': result.is_significant,
                'confidence_interval': result.confidence_interval
            },
            'recommendation': self._get_recommendation(result, config)
        }
    
    def _get_recommendation(self, result: ExperimentResult, config: ExperimentConfig) -> str:
        """Generate a recommendation based on experiment results."""
        if not result.is_significant:
            return "Results not statistically significant. Consider running longer or increasing sample size."
        
        if result.lift > 0 and result.treatment_revenue > result.control_revenue:
            return f"✅ IMPLEMENT: Treatment pricing shows {result.lift:.1f}% conversion lift and higher revenue."
        elif result.lift > 0 and result.treatment_revenue < result.control_revenue:
            return f"⚠️ CAUTION: Higher conversions but lower revenue. Consider profit margin impact."
        elif result.lift < 0 and result.treatment_revenue > result.control_revenue:
            return f"📊 CONSIDER: Lower conversions but higher revenue per customer. Good for capacity constraints."
        else:
            return f"❌ REJECT: Treatment shows {abs(result.lift):.1f}% conversion decrease with lower revenue."


def main():
    """Demonstration of A/B Testing Framework."""
    
    print("🧪 A/B TESTING FRAMEWORK FOR INSURANCE PRICING")
    print("=" * 60)
    
    framework = ABTestingFramework()
    
    # Create experiments
    experiments = [
        ExperimentConfig(
            experiment_id="exp_001",
            name="5% Discount for High-Risk Segment",
            description="Test if 5% discount increases conversions for high-risk customers",
            control_price_modifier=1.0,
            treatment_price_modifier=0.95,
            target_segment="high_risk",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            min_sample_size=500,
            confidence_level=0.95
        ),
        ExperimentConfig(
            experiment_id="exp_002",
            name="10% Premium Increase for Low-Risk",
            description="Test price elasticity of low-risk customers",
            control_price_modifier=1.0,
            treatment_price_modifier=1.10,
            target_segment="low_risk",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            min_sample_size=500,
            confidence_level=0.95
        )
    ]
    
    for exp in experiments:
        framework.create_experiment(exp)
        print(f"\n📋 Experiment: {exp.name}")
        print(f"   Segment: {exp.target_segment}")
        print(f"   Price Change: {(exp.treatment_price_modifier - 1) * 100:+.0f}%")
        
        # Simulate
        result = framework.simulate_experiment(exp.experiment_id, sample_size=2000)
        
        print(f"\n   📊 Results:")
        print(f"   Control: {result.control_conversions}/{result.control_total} = {result.control_conversions/result.control_total:.1%}")
        print(f"   Treatment: {result.treatment_conversions}/{result.treatment_total} = {result.treatment_conversions/result.treatment_total:.1%}")
        print(f"   Lift: {result.lift:+.1f}%")
        print(f"   P-value: {result.p_value:.4f}")
        print(f"   Significant: {'✅ Yes' if result.is_significant else '❌ No'}")
        
        summary = framework.get_experiment_summary(exp.experiment_id)
        print(f"\n   💡 Recommendation:")
        print(f"   {summary['recommendation']}")
    
    # Price sensitivity analysis
    print("\n" + "=" * 60)
    print("📈 PRICE SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    for segment in ['low_risk', 'high_risk']:
        print(f"\n📊 Segment: {segment}")
        sensitivity = framework.analyze_price_sensitivity(segment=segment)
        print(sensitivity[['price_modifier', 'conversion_rate', 'revenue_per_visitor']].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✅ A/B Testing Framework Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()


