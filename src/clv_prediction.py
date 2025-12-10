"""
Customer Lifetime Value (CLV) Prediction Module
================================================

Predicts the lifetime value of insurance customers considering:
- Policy renewal probability
- Cross-sell potential (home, life, pet insurance)
- Claims propensity over time
- Customer acquisition cost

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
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class CustomerProfile:
    """Customer profile for CLV calculation."""
    age_group: str
    income_level: str
    credit_score: float
    years_as_customer: int
    annual_premium: float
    claims_count: int
    risk_score: float
    married: bool
    children: bool
    vehicle_type: str
    region: str


class CLVPredictor:
    """
    Customer Lifetime Value Prediction Engine.
    
    Uses actuarial and behavioral models to predict the total value
    a customer will bring over their relationship with the insurer.
    """
    
    def __init__(self):
        """Initialize CLV predictor with default parameters."""
        
        # Average customer lifespan by segment (years)
        self.avg_customer_lifespan = {
            'high_value': 12,
            'medium_value': 8,
            'low_value': 4,
            'at_risk': 2
        }
        
        # Base renewal rates by age group
        self.base_renewal_rates = {
            '16-25': 0.65,  # Young drivers switch more
            '26-39': 0.78,
            '40-64': 0.85,  # Most loyal
            '65+': 0.80
        }
        
        # Cross-sell conversion rates
        self.cross_sell_rates = {
            'home_insurance': 0.15,
            'life_insurance': 0.08,
            'pet_insurance': 0.05,
            'travel_insurance': 0.12,
            'umbrella_policy': 0.03
        }
        
        # Average product values (annual premium)
        self.product_values = {
            'car_insurance': 650,  # Base product
            'home_insurance': 450,
            'life_insurance': 300,
            'pet_insurance': 180,
            'travel_insurance': 120,
            'umbrella_policy': 200
        }
        
        # Customer acquisition cost by channel
        self.acquisition_costs = {
            'direct': 80,
            'comparison_site': 120,
            'broker': 150,
            'referral': 40,
            'renewal': 20
        }
        
        # Discount rate for NPV calculation
        self.discount_rate = 0.08
        
        # Profit margin on premium
        self.profit_margin = 0.12
    
    def calculate_renewal_probability(self, profile: CustomerProfile) -> float:
        """
        Calculate probability of policy renewal.
        
        Factors:
        - Age group (loyalty patterns)
        - Years as customer (stickiness)
        - Claims history (satisfaction)
        - Credit score (financial stability)
        - Premium competitiveness
        """
        # Base rate from age group
        base_rate = self.base_renewal_rates.get(profile.age_group, 0.75)
        
        # Tenure bonus (longer customers more likely to stay)
        tenure_bonus = min(profile.years_as_customer * 0.02, 0.10)
        
        # Claims penalty (recent claims reduce renewal)
        claims_penalty = min(profile.claims_count * 0.05, 0.15)
        
        # Credit score factor
        credit_factor = (profile.credit_score - 0.5) * 0.1
        
        # Risk score factor (high risk = higher premium = lower renewal)
        risk_penalty = max(0, (profile.risk_score - 0.3) * 0.2)
        
        # Family bonus (families tend to be more stable)
        family_bonus = 0.03 if profile.married else 0
        family_bonus += 0.02 if profile.children else 0
        
        renewal_prob = base_rate + tenure_bonus - claims_penalty + credit_factor - risk_penalty + family_bonus
        
        return np.clip(renewal_prob, 0.3, 0.95)
    
    def calculate_cross_sell_potential(self, profile: CustomerProfile) -> Dict[str, float]:
        """
        Calculate cross-sell potential for each product.
        
        Returns dict of product -> (probability, expected_value)
        """
        cross_sell = {}
        
        for product, base_rate in self.cross_sell_rates.items():
            # Adjust based on customer profile
            adjusted_rate = base_rate
            
            # Income adjustment
            income_multiplier = {
                'poverty': 0.3,
                'working_class': 0.7,
                'middle_class': 1.2,
                'upper_class': 1.8
            }.get(profile.income_level, 1.0)
            
            # Product-specific adjustments
            if product == 'home_insurance':
                # Homeowners more likely
                if profile.age_group in ['40-64', '65+']:
                    adjusted_rate *= 1.5
                if profile.married:
                    adjusted_rate *= 1.3
                    
            elif product == 'life_insurance':
                # Families more likely
                if profile.children:
                    adjusted_rate *= 2.0
                if profile.married:
                    adjusted_rate *= 1.5
                    
            elif product == 'pet_insurance':
                # Younger and families
                if profile.age_group in ['26-39', '40-64']:
                    adjusted_rate *= 1.3
                    
            elif product == 'travel_insurance':
                # Higher income more travel
                adjusted_rate *= income_multiplier
                
            elif product == 'umbrella_policy':
                # High net worth
                if profile.income_level == 'upper_class':
                    adjusted_rate *= 3.0
            
            # Tenure bonus (longer customers more receptive)
            tenure_factor = 1 + min(profile.years_as_customer * 0.05, 0.3)
            
            final_rate = np.clip(adjusted_rate * income_multiplier * tenure_factor, 0, 0.5)
            expected_value = final_rate * self.product_values[product]
            
            cross_sell[product] = {
                'probability': round(final_rate, 3),
                'annual_value': self.product_values[product],
                'expected_value': round(expected_value, 2)
            }
        
        return cross_sell
    
    def calculate_claims_propensity(self, profile: CustomerProfile, years: int = 5) -> List[float]:
        """
        Project claims propensity over time.
        
        Returns list of expected claim costs per year.
        """
        # Base claim frequency from risk score
        base_frequency = profile.risk_score * 0.122  # 12.2% base rate
        
        # Average claim severity
        base_severity = 3500
        
        # Age-based severity adjustment
        age_severity = {
            '16-25': 1.3,  # Young drivers - higher severity
            '26-39': 1.0,
            '40-64': 0.9,
            '65+': 1.1  # Older - more injury claims
        }.get(profile.age_group, 1.0)
        
        yearly_claims = []
        for year in range(years):
            # Risk tends to decrease with tenure (safer driving habits)
            tenure_adjustment = 1 - min(year * 0.02, 0.15)
            
            # Age progression effect
            age_effect = 1 + year * 0.01 if profile.age_group == '65+' else 1 - year * 0.005
            
            expected_claims = (base_frequency * tenure_adjustment * age_effect * 
                             base_severity * age_severity)
            
            yearly_claims.append(round(expected_claims, 2))
        
        return yearly_claims
    
    def calculate_clv(self, profile: CustomerProfile, 
                     acquisition_channel: str = 'direct',
                     projection_years: int = 10) -> Dict[str, Any]:
        """
        Calculate Customer Lifetime Value.
        
        CLV = Sum of discounted future profits - Acquisition Cost
        
        Where future profits = (Premium Revenue - Expected Claims - Expenses) 
                              + Cross-sell Revenue
        """
        # Get renewal probability
        renewal_prob = self.calculate_renewal_probability(profile)
        
        # Get cross-sell potential
        cross_sell = self.calculate_cross_sell_potential(profile)
        
        # Get claims propensity
        claims_propensity = self.calculate_claims_propensity(profile, projection_years)
        
        # Acquisition cost
        cac = self.acquisition_costs.get(acquisition_channel, 100)
        
        # Calculate yearly values
        yearly_values = []
        cumulative_clv = 0
        survival_prob = 1.0
        
        for year in range(projection_years):
            # Probability customer is still active
            survival_prob *= renewal_prob if year > 0 else 1.0
            
            if survival_prob < 0.05:  # Negligible probability
                break
            
            # Core policy revenue
            premium_revenue = profile.annual_premium
            expected_claims = claims_propensity[min(year, len(claims_propensity)-1)]
            expenses = premium_revenue * 0.28  # 28% expense ratio
            
            # Core policy profit
            policy_profit = (premium_revenue - expected_claims - expenses) * survival_prob
            
            # Cross-sell revenue (phased in over time)
            cross_sell_revenue = 0
            if year >= 1:  # Cross-sell starts year 2
                for product, details in cross_sell.items():
                    # Probability of having the product by this year
                    product_prob = 1 - (1 - details['probability']) ** year
                    cross_sell_revenue += product_prob * details['annual_value'] * self.profit_margin
                cross_sell_revenue *= survival_prob
            
            # Total yearly value (before discounting)
            yearly_profit = policy_profit + cross_sell_revenue
            
            # Discount to present value
            discount_factor = 1 / (1 + self.discount_rate) ** year
            pv_profit = yearly_profit * discount_factor
            
            yearly_values.append({
                'year': year + 1,
                'survival_probability': round(survival_prob, 3),
                'premium_revenue': round(premium_revenue * survival_prob, 2),
                'expected_claims': round(expected_claims * survival_prob, 2),
                'policy_profit': round(policy_profit, 2),
                'cross_sell_revenue': round(cross_sell_revenue, 2),
                'total_profit': round(yearly_profit, 2),
                'discount_factor': round(discount_factor, 4),
                'present_value': round(pv_profit, 2)
            })
            
            cumulative_clv += pv_profit
        
        # Final CLV = NPV of profits - acquisition cost
        final_clv = cumulative_clv - cac
        
        # Customer segment classification
        if final_clv >= 1500:
            segment = 'Platinum'
            recommendation = 'Premium service, priority claims, loyalty rewards'
        elif final_clv >= 800:
            segment = 'Gold'
            recommendation = 'Retention focus, cross-sell opportunities'
        elif final_clv >= 400:
            segment = 'Silver'
            recommendation = 'Standard service, efficiency focus'
        else:
            segment = 'Bronze'
            recommendation = 'Cost optimization, consider risk-based pricing'
        
        # Strategic pricing recommendation
        if final_clv > 1000:
            pricing_strategy = 'Accept lower margins to retain - high CLV customer'
            acceptable_discount = min(15, (final_clv - 800) / 100 * 2)
        elif final_clv > 500:
            pricing_strategy = 'Standard pricing - balanced approach'
            acceptable_discount = 5
        else:
            pricing_strategy = 'Risk-adequate pricing essential - low CLV'
            acceptable_discount = 0
        
        return {
            'customer_lifetime_value': round(final_clv, 2),
            'customer_segment': segment,
            'recommendation': recommendation,
            'pricing_strategy': pricing_strategy,
            'acceptable_discount_percent': round(acceptable_discount, 1),
            
            'metrics': {
                'renewal_probability': round(renewal_prob, 3),
                'acquisition_cost': cac,
                'projected_years': len(yearly_values),
                'total_premium_revenue': round(sum(y['premium_revenue'] for y in yearly_values), 2),
                'total_expected_claims': round(sum(y['expected_claims'] for y in yearly_values), 2),
                'total_cross_sell_revenue': round(sum(y['cross_sell_revenue'] for y in yearly_values), 2),
                'gross_clv': round(cumulative_clv, 2),
                'net_clv': round(final_clv, 2)
            },
            
            'cross_sell_potential': cross_sell,
            'yearly_projections': yearly_values
        }
    
    def segment_portfolio(self, customers: List[CustomerProfile]) -> Dict[str, Any]:
        """
        Segment a portfolio of customers by CLV.
        
        Returns portfolio-level CLV analytics.
        """
        results = []
        
        for customer in customers:
            clv_result = self.calculate_clv(customer)
            results.append({
                'clv': clv_result['customer_lifetime_value'],
                'segment': clv_result['customer_segment'],
                'renewal_prob': clv_result['metrics']['renewal_probability']
            })
        
        df = pd.DataFrame(results)
        
        return {
            'total_portfolio_clv': round(df['clv'].sum(), 2),
            'average_clv': round(df['clv'].mean(), 2),
            'median_clv': round(df['clv'].median(), 2),
            'segment_distribution': df['segment'].value_counts().to_dict(),
            'average_renewal_probability': round(df['renewal_prob'].mean(), 3),
            'clv_percentiles': {
                '10th': round(df['clv'].quantile(0.1), 2),
                '25th': round(df['clv'].quantile(0.25), 2),
                '50th': round(df['clv'].quantile(0.5), 2),
                '75th': round(df['clv'].quantile(0.75), 2),
                '90th': round(df['clv'].quantile(0.9), 2)
            }
        }


def main():
    """Demonstration of CLV prediction."""
    
    print("💰 CUSTOMER LIFETIME VALUE (CLV) PREDICTION")
    print("=" * 60)
    
    predictor = CLVPredictor()
    
    # Example customer profiles
    profiles = [
        CustomerProfile(
            age_group='40-64', income_level='upper_class', credit_score=0.85,
            years_as_customer=5, annual_premium=750, claims_count=0,
            risk_score=0.15, married=True, children=True,
            vehicle_type='family_sedan', region='South East'
        ),
        CustomerProfile(
            age_group='26-39', income_level='middle_class', credit_score=0.72,
            years_as_customer=2, annual_premium=680, claims_count=1,
            risk_score=0.25, married=True, children=False,
            vehicle_type='suv', region='London'
        ),
        CustomerProfile(
            age_group='16-25', income_level='working_class', credit_score=0.55,
            years_as_customer=1, annual_premium=1200, claims_count=0,
            risk_score=0.45, married=False, children=False,
            vehicle_type='sports_car', region='North West'
        )
    ]
    
    profile_names = ['Premium Family Customer', 'Urban Professional', 'Young High-Risk']
    
    for name, profile in zip(profile_names, profiles):
        print(f"\n{'='*60}")
        print(f"📊 {name}")
        print(f"{'='*60}")
        
        result = predictor.calculate_clv(profile)
        
        print(f"\n💎 Customer Segment: {result['customer_segment']}")
        print(f"💰 Customer Lifetime Value: £{result['customer_lifetime_value']:,.2f}")
        print(f"📈 Renewal Probability: {result['metrics']['renewal_probability']:.1%}")
        
        print(f"\n📊 CLV Breakdown:")
        print(f"   • Total Premium Revenue: £{result['metrics']['total_premium_revenue']:,.2f}")
        print(f"   • Expected Claims: £{result['metrics']['total_expected_claims']:,.2f}")
        print(f"   • Cross-sell Revenue: £{result['metrics']['total_cross_sell_revenue']:,.2f}")
        print(f"   • Acquisition Cost: £{result['metrics']['acquisition_cost']}")
        
        print(f"\n🎯 Strategic Recommendation:")
        print(f"   {result['recommendation']}")
        print(f"\n💼 Pricing Strategy:")
        print(f"   {result['pricing_strategy']}")
        print(f"   Acceptable Discount: up to {result['acceptable_discount_percent']}%")
        
        print(f"\n🛒 Cross-sell Opportunities:")
        for product, details in result['cross_sell_potential'].items():
            if details['probability'] > 0.05:
                print(f"   • {product.replace('_', ' ').title()}: "
                      f"{details['probability']:.1%} probability, "
                      f"£{details['annual_value']}/year")
    
    print(f"\n{'='*60}")
    print("✅ CLV Prediction Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

