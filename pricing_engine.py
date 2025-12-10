"""
Advanced Car Insurance Pricing Engine

This module provides a comprehensive pricing engine for car insurance that incorporates:
- Actuarial risk modeling based on UK insurance statistics
- Multi-factor risk assessment
- Premium calculation with profit margins and expenses
- Different coverage types and policy options

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
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CarInsurancePricingEngine:
    """
    Advanced pricing engine for car insurance based on actuarial principles.

    The engine uses a combination of frequency and severity modeling to determine
    fair premiums while ensuring profitability.
    """

    def __init__(self, base_rate_gbp: float = 400.0):
        """
        Initialize the pricing engine.

        Args:
            base_rate_gbp: Base annual premium for a standard policy (£400 typical UK average)
        """
        self.base_rate = base_rate_gbp

        # Risk factor multipliers (calibrated to UK market)
        self.age_multipliers = {
            '16-25': 2.8,   # High risk young drivers
            '26-39': 1.2,   # Moderate risk
            '40-64': 1.0,   # Baseline
            '65+': 1.3      # Higher severity but lower frequency
        }

        self.region_multipliers = {
            'London': 1.4,          # Urban congestion, high repair costs
            'South East': 1.1,      # Affluent area, higher vehicle values
            'South West': 0.9,      # Rural, lower congestion
            'East Anglia': 0.85,    # Rural areas
            'West Midlands': 1.3,   # Urban/industrial
            'East Midlands': 1.1,   # Mixed
            'Yorkshire': 1.2,       # Urban density
            'North West': 1.4,      # High urban density
            'North East': 1.5,      # Industrial legacy, weather
            'Wales': 1.1,           # Mixed rural/urban
            'Scotland': 1.3         # Weather, rural factors
        }

        self.vehicle_multipliers = {
            'small_hatchback': 0.85,    # Safer, lower repair costs
            'family_sedan': 1.0,        # Baseline
            'suv': 1.25,                # Higher repair costs, rollover risk
            'sports_car': 1.8,          # Performance, higher repair costs
            'luxury_sedan': 1.5,        # High repair costs
            'mpv': 0.95                 # Family vehicle, careful drivers
        }

        self.experience_multipliers = {
            '0-2y': 2.5,     # Very inexperienced
            '3-5y': 1.8,     # Still learning
            '6-9y': 1.3,     # Developing skills
            '0-9y': 1.4,     # Mixed experience
            '10-19y': 1.1,   # Experienced
            '20-29y': 1.0,   # Very experienced
            '30y+': 0.9      # Expert drivers
        }

        self.safety_multipliers = {
            'basic': 1.2,       # No safety features
            'standard': 1.0,    # Standard safety
            'advanced': 0.85    # Advanced safety systems
        }

        # Coverage type multipliers
        self.coverage_multipliers = {
            'third_party': 0.6,      # Basic coverage
            'third_party_fire_theft': 0.75,  # TPO + fire/theft
            'comprehensive': 1.0     # Full coverage
        }

        # NCD (No Claims Discount) scale
        self.ncd_multipliers = {
            0: 1.0,    # No NCD
            1: 0.95,   # 1 year
            2: 0.90,   # 2 years
            3: 0.85,   # 3 years
            4: 0.80,   # 4 years
            5: 0.75,   # 5+ years
        }

        # Excess levels and their impact
        self.excess_multipliers = {
            0: 1.0,      # No voluntary excess
            100: 0.95,   # £100 excess
            200: 0.90,   # £200 excess
            500: 0.85,   # £500 excess
            1000: 0.80   # £1000 excess
        }

    def calculate_risk_score(self, driver_data: Dict) -> float:
        """
        Calculate comprehensive risk score for a driver.

        Args:
            driver_data: Dictionary containing driver and vehicle information

        Returns:
            Risk score (multiplier relative to baseline)
        """
        risk_score = 1.0

        # Age risk
        age = driver_data.get('AGE', '40-64')
        risk_score *= self.age_multipliers.get(age, 1.0)

        # Regional risk
        region = driver_data.get('REGION', 'South East')
        risk_score *= self.region_multipliers.get(region, 1.0)

        # Vehicle risk
        vehicle = driver_data.get('VEHICLE_TYPE', 'family_sedan')
        risk_score *= self.vehicle_multipliers.get(vehicle, 1.0)

        # Driving experience
        experience = driver_data.get('DRIVING_EXPERIENCE', '20-29y')
        risk_score *= self.experience_multipliers.get(experience, 1.0)

        # Safety rating
        safety = driver_data.get('SAFETY_RATING', 'standard')
        risk_score *= self.safety_multipliers.get(safety, 1.0)

        # Gender factor (males slightly higher risk in younger age groups)
        if driver_data.get('GENDER') == 'male' and age in ['16-25', '26-39']:
            risk_score *= 1.1

        # Mileage factor
        mileage = driver_data.get('ANNUAL_MILEAGE', 8000)
        if mileage > 15000:
            risk_score *= 1.25
        elif mileage > 12000:
            risk_score *= 1.15
        elif mileage < 5000:
            risk_score *= 0.9

        # Credit score factor (higher credit = lower risk)
        credit = driver_data.get('CREDIT_SCORE', 0.6)
        if credit > 0.8:
            risk_score *= 0.9
        elif credit < 0.4:
            risk_score *= 1.2

        # Past incidents
        speeding = driver_data.get('SPEEDING_VIOLATIONS', 0)
        duis = driver_data.get('DUIS', 0)
        accidents = driver_data.get('PAST_ACCIDENTS', 0)

        incident_factor = 1 + (speeding * 0.1) + (duis * 0.3) + (accidents * 0.2)
        risk_score *= min(incident_factor, 2.5)  # Cap at 2.5x

        return risk_score

    def calculate_premium(self,
                         driver_data: Dict,
                         coverage_type: str = 'comprehensive',
                         ncd_years: int = 0,
                         voluntary_excess: int = 0,
                         include_ancillaries: bool = False) -> Dict:
        """
        Calculate comprehensive premium for a driver.

        Args:
            driver_data: Driver and vehicle information
            coverage_type: Type of coverage ('third_party', 'third_party_fire_theft', 'comprehensive')
            ncd_years: Years of no claims discount (0-5)
            voluntary_excess: Voluntary excess amount (£0, £100, £200, £500, £1000)
            include_ancillaries: Whether to include breakdown cover, legal expenses, etc.

        Returns:
            Dictionary with premium breakdown
        """

        # Base risk-adjusted premium
        risk_score = self.calculate_risk_score(driver_data)
        risk_adjusted_premium = self.base_rate * risk_score

        # Apply coverage multiplier
        coverage_multiplier = self.coverage_multipliers.get(coverage_type, 1.0)
        coverage_premium = risk_adjusted_premium * coverage_multiplier

        # Apply NCD discount
        ncd_multiplier = self.ncd_multipliers.get(min(ncd_years, 5), 1.0)
        ncd_adjusted_premium = coverage_premium * ncd_multiplier

        # Apply excess discount
        excess_multiplier = self.excess_multipliers.get(voluntary_excess, 1.0)
        excess_adjusted_premium = ncd_adjusted_premium * excess_multiplier

        # Add profit margin and expenses (typical UK insurer loadings)
        # Profit margin: 8-12%, Expense ratio: 25-30%
        profit_margin = 0.10  # 10%
        expense_loading = 0.28  # 28%

        gross_premium = excess_adjusted_premium / (1 - profit_margin - expense_loading)

        # Add IPT (Insurance Premium Tax) - 12% in UK
        ipt_rate = 0.12
        ipt_amount = gross_premium * ipt_rate
        total_premium = gross_premium + ipt_amount

        # Optional ancillaries
        ancillary_premium = 0
        if include_ancillaries:
            # Breakdown cover (£50), Legal expenses (£25), Windscreen (£15)
            ancillary_premium = 50 + 25 + 15

        final_premium = total_premium + ancillary_premium

        # Calculate monthly premium
        monthly_premium = final_premium / 12

        return {
            'annual_premium': round(final_premium, 2),
            'monthly_premium': round(monthly_premium, 2),
            'risk_score': round(risk_score, 3),
            'breakdown': {
                'base_risk_premium': round(risk_adjusted_premium, 2),
                'coverage_adjusted': round(coverage_premium, 2),
                'ncd_adjusted': round(ncd_adjusted_premium, 2),
                'excess_adjusted': round(excess_adjusted_premium, 2),
                'gross_premium': round(gross_premium, 2),
                'ipt_amount': round(ipt_amount, 2),
                'ancillary_premium': round(ancillary_premium, 2)
            },
            'factors_applied': {
                'coverage_type': coverage_type,
                'ncd_years': ncd_years,
                'voluntary_excess': voluntary_excess,
                'ancillaries_included': include_ancillaries
            }
        }

    def batch_price_policies(self,
                           driver_dataset: pd.DataFrame,
                           coverage_type: str = 'comprehensive',
                           default_ncd: int = 1,
                           default_excess: int = 200) -> pd.DataFrame:
        """
        Calculate premiums for a batch of drivers.

        Args:
            driver_dataset: DataFrame with driver information
            coverage_type: Default coverage type
            default_ncd: Default NCD years
            default_excess: Default voluntary excess

        Returns:
            DataFrame with premium calculations added
        """

        results = []

        for _, driver in driver_dataset.iterrows():
            driver_dict = driver.to_dict()

            # Add default policy options if not present
            policy_options = {
                'coverage_type': coverage_type,
                'ncd_years': default_ncd,
                'voluntary_excess': default_excess,
                'include_ancillaries': False
            }

            premium_result = self.calculate_premium(driver_dict, **policy_options)

            result_row = driver_dict.copy()
            result_row.update({
                'CALCULATED_PREMIUM': premium_result['annual_premium'],
                'MONTHLY_PREMIUM': premium_result['monthly_premium'],
                'RISK_SCORE': premium_result['risk_score'],
                'COVERAGE_TYPE': coverage_type,
                'NCD_YEARS': default_ncd,
                'VOLUNTARY_EXCESS': default_excess
            })

            results.append(result_row)

        return pd.DataFrame(results)

    def analyze_pricing_distribution(self, priced_dataset: pd.DataFrame) -> Dict:
        """
        Analyze the distribution of calculated premiums.

        Args:
            priced_dataset: DataFrame with calculated premiums

        Returns:
            Dictionary with pricing statistics
        """

        premiums = priced_dataset['CALCULATED_PREMIUM']

        return {
            'total_policies': len(premiums),
            'average_premium': round(premiums.mean(), 2),
            'median_premium': round(premiums.median(), 2),
            'min_premium': round(premiums.min(), 2),
            'max_premium': round(premiums.max(), 2),
            'premium_percentiles': {
                '10th': round(np.percentile(premiums, 10), 2),
                '25th': round(np.percentile(premiums, 25), 2),
                '75th': round(np.percentile(premiums, 75), 2),
                '90th': round(np.percentile(premiums, 90), 2),
                '95th': round(np.percentile(premiums, 95), 2)
            },
            'risk_score_distribution': {
                'average_risk': round(priced_dataset['RISK_SCORE'].mean(), 3),
                'high_risk_threshold': round(np.percentile(priced_dataset['RISK_SCORE'], 90), 3),
                'low_risk_threshold': round(np.percentile(priced_dataset['RISK_SCORE'], 10), 3)
            }
        }


def main():
    """
    Demonstrate the pricing engine with sample calculations.
    """
    print("🚗 InsurePrice Car Insurance Pricing Engine")
    print("=" * 50)

    # Initialize pricing engine
    engine = CarInsurancePricingEngine(base_rate_gbp=400.0)

    # Load enhanced synthetic data
    try:
        data_path = r'c:\Users\mn3g24\OneDrive - University of Southampton\Desktop\projects\InsurePrice\InsurePrice\Enhanced_Synthetic_Car_Insurance_Claims.csv'
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} driver records from enhanced synthetic dataset")

        # Sample premium calculations for different risk profiles
        sample_drivers = [
            {'index': 0, 'description': 'Young urban driver (high risk)'},
            {'index': 1000, 'description': 'Middle-aged rural driver (low risk)'},
            {'index': 5000, 'description': 'Senior driver (moderate risk)'}
        ]

        print("\n💰 Sample Premium Calculations:")
        print("-" * 50)

        for sample in sample_drivers:
            driver_data = df.iloc[sample['index']].to_dict()
            premium = engine.calculate_premium(driver_data)

            print(f"\n{sample['description']}:")
            print(f"  Driver: {driver_data['AGE']} {driver_data['GENDER']}, {driver_data['REGION']}")
            print(f"  Vehicle: {driver_data['VEHICLE_TYPE']}")
            print(f"  Risk Score: {premium['risk_score']}")
            print(f"  Annual Premium: £{premium['annual_premium']}")
            print(f"  Monthly Premium: £{premium['monthly_premium']}")

        # Batch pricing for all drivers
        print("\n⚡ Batch Pricing Analysis:")
        print("-" * 50)

        priced_df = engine.batch_price_policies(df.head(1000))  # Price first 1000 for demo
        pricing_stats = engine.analyze_pricing_distribution(priced_df)

        print(f"Sample Size: {pricing_stats['total_policies']} policies")
        print(f"Average Premium: £{pricing_stats['average_premium']}")
        print(f"Premium Range: £{pricing_stats['min_premium']} - £{pricing_stats['max_premium']}")
        print(f"Median Premium: £{pricing_stats['median_premium']}")

        print("\n📊 Premium Distribution:")
        percentiles = pricing_stats['premium_percentiles']
        print(f"  10th percentile: £{percentiles['10th']}")
        print(f"  25th percentile: £{percentiles['25th']}")
        print(f"  75th percentile: £{percentiles['75th']}")
        print(f"  90th percentile: £{percentiles['90th']}")
        print(f"  95th percentile: £{percentiles['95th']}")

        # Risk-based analysis
        print("\n🎯 Risk Analysis:")
        risk_stats = pricing_stats['risk_score_distribution']
        print(f"  Average Risk Score: {risk_stats['average_risk']}")
        print(f"  High Risk Threshold (90th %): {risk_stats['high_risk_threshold']}")
        print(f"  Low Risk Threshold (10th %): {risk_stats['low_risk_threshold']}")

        # Save priced dataset sample
        output_path = r'c:\Users\mn3g24\OneDrive - University of Southampton\Desktop\projects\InsurePrice\InsurePrice\Sample_Priced_Policies.csv'
        priced_df.head(100).to_csv(output_path, index=False)
        print(f"\n💾 Sample priced policies saved to: {output_path}")

    except FileNotFoundError:
        print("❌ Enhanced synthetic dataset not found. Please run generate_data.py first.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
