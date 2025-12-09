"""
Enhanced Synthetic Car Insurance Data Generator

Author: Masood Nazari
Business Intelligence Analyst | Data Science | AI | Clinical Research
Email: M.Nazari@soton.ac.uk
Portfolio: https://michaeltheanalyst.github.io/
LinkedIn: linkedin.com/in/masood-nazari
GitHub: github.com/michaeltheanalyst
Date: December 2025
Project: InsurePrice Car Insurance Risk Modeling
Description: Generates realistic synthetic car insurance data with enhanced features
"""

import pandas as pd
import numpy as np
import random
from scipy import stats

def generate_synthetic_data(num_rows=10000):
    """
    Enhanced synthetic car insurance data generator based on UK statistics.

    Key improvements:
    - Realistic claim frequencies based on UK DfT and ABI data
    - Sophisticated claim severity modeling using mixture distributions
    - Geographic risk factors based on regional statistics
    - Enhanced risk factors including vehicle make/model, safety features
    - More accurate demographic correlations
    """
    np.random.seed(42)
    random.seed(42)

    data = []

    # Enhanced demographics based on UK population statistics
    age_groups = ['16-25', '26-39', '40-64', '65+']
    age_probs = [0.12, 0.35, 0.38, 0.15]  # More realistic UK age distribution

    # UK regions with different risk profiles (based on ABI data)
    regions = ['London', 'South East', 'South West', 'East Anglia', 'West Midlands',
               'East Midlands', 'Yorkshire', 'North West', 'North East', 'Wales', 'Scotland']
    region_risk_factors = {
        'London': 0.85,      # Lower risk due to congestion
        'South East': 1.0,   # Baseline
        'South West': 0.95,  # Rural, slightly lower
        'East Anglia': 0.90, # Rural areas
        'West Midlands': 1.15, # Urban congestion
        'East Midlands': 1.05, # Mixed
        'Yorkshire': 1.10,   # Urban/industrial
        'North West': 1.20,  # Urban density
        'North East': 1.25,  # Industrial legacy
        'Wales': 1.05,       # Mixed rural/urban
        'Scotland': 1.15     # Weather and rural factors
    }

    # Vehicle types with more categories and realistic risk profiles
    vehicle_types = ['small_hatchback', 'family_sedan', 'suv', 'sports_car', 'luxury_sedan', 'mpv']
    vehicle_risk_factors = {
        'small_hatchback': 0.9,   # Safer, more agile
        'family_sedan': 1.0,      # Baseline
        'suv': 1.15,              # Higher center of gravity
        'sports_car': 1.4,        # Performance vehicles
        'luxury_sedan': 1.1,      # Often driven by older drivers
        'mpv': 0.95               # Family vehicles, careful drivers
    }

    # Base claim probabilities based on UK statistics (ABI/DFT data)
    # Young drivers: ~15-20% claim frequency
    # Middle-aged: ~8-10%
    # Seniors: ~6-8% (but higher severity)
    base_claim_freq = {
        '16-25': 0.18,
        '26-39': 0.09,
        '40-64': 0.07,
        '65+': 0.08
    }

    for i in range(num_rows):
        row_id = i + 1

        # 1. Enhanced Demographics
        age = np.random.choice(age_groups, p=age_probs)
        gender = np.random.choice(['male', 'female'], p=[0.49, 0.51])  # UK gender split
        region = np.random.choice(regions)  # Equal probability for simplicity

        # Driving Experience: more sophisticated correlation with age
        if age == '16-25':
            experience = np.random.choice(['0-2y', '3-5y', '6-9y'], p=[0.6, 0.3, 0.1])
        elif age == '26-39':
            experience = np.random.choice(['0-9y', '10-19y'], p=[0.4, 0.6])
        elif age == '40-64':
            experience = np.random.choice(['10-19y', '20-29y'], p=[0.3, 0.7])
        else:
            experience = np.random.choice(['20-29y', '30y+'], p=[0.1, 0.9])

        # 2. Vehicle & Usage
        # Vehicle choice correlated with demographics
        if age == '16-25':
            veh_probs = [0.4, 0.2, 0.1, 0.25, 0.03, 0.02]  # More sports cars for young
        elif age == '26-39':
            veh_probs = [0.25, 0.3, 0.2, 0.15, 0.08, 0.02]
        elif age == '40-64':
            veh_probs = [0.15, 0.35, 0.25, 0.05, 0.15, 0.05]
        else:
            veh_probs = [0.1, 0.4, 0.15, 0.02, 0.28, 0.05]  # More luxury sedans for seniors

        veh_type = np.random.choice(vehicle_types, p=veh_probs)

        # Annual Mileage: correlated with age and region
        base_mileage = 8000  # UK average
        if age == '16-25':
            base_mileage *= 0.8  # Less experienced, less driving
        elif age == '65+':
            base_mileage *= 0.6  # Retired, less driving

        # Regional variation (urban vs rural)
        urban_regions = ['London', 'West Midlands', 'North West']
        if region in urban_regions:
            base_mileage *= 0.7  # Congestion reduces mileage

        mileage = int(np.random.normal(base_mileage, 2500))
        mileage = max(500, min(35000, mileage))

        # 3. Risk Factors
        # Start with base claim probability
        claim_prob = base_claim_freq[age]

        # Apply modifiers
        claim_prob *= region_risk_factors[region]
        claim_prob *= vehicle_risk_factors[veh_type]

        # Gender factor (males slightly higher risk in younger ages)
        if gender == 'male' and age in ['16-25', '26-39']:
            claim_prob *= 1.15
        elif gender == 'male' and age == '40-64':
            claim_prob *= 1.05

        # Mileage factor
        if mileage > 15000:
            claim_prob *= 1.3
        elif mileage > 12000:
            claim_prob *= 1.15

        # Experience factor
        if experience == '0-2y':
            claim_prob *= 2.0  # Inexperienced drivers
        elif experience == '3-5y':
            claim_prob *= 1.3
        elif experience == '6-9y':
            claim_prob *= 1.1

        # Safety features (simplified)
        safety_rating = np.random.choice(['basic', 'standard', 'advanced'], p=[0.3, 0.5, 0.2])
        if safety_rating == 'advanced':
            claim_prob *= 0.85
        elif safety_rating == 'basic':
            claim_prob *= 1.1

        # Cap probability at realistic maximum
        claim_prob = min(0.45, claim_prob)

        # Generate claim outcome
        outcome = 1 if (random.random() < claim_prob) else 0

        # 4. Enhanced Claim Severity Modeling
        claim_amount = 0.0
        if outcome == 1:
            # UK average comprehensive claim ~£3,800 (ABI data)
            # Use mixture of distributions for realistic severity
            # 70% minor claims (£500-£2,000)
            # 25% moderate claims (£2,000-£10,000)
            # 5% major claims (£10,000+)

            claim_type = np.random.choice(['minor', 'moderate', 'major'], p=[0.7, 0.25, 0.05])

            if claim_type == 'minor':
                # Lognormal for smaller claims
                mu, sigma = 6.5, 0.8  # Mean ~£700
                claim_amount = np.random.lognormal(mu, sigma)
                claim_amount = max(300, min(2000, claim_amount))

            elif claim_type == 'moderate':
                # Lognormal for medium claims
                mu, sigma = 7.8, 0.9  # Mean ~£2,500
                claim_amount = np.random.lognormal(mu, sigma)
                claim_amount = max(1500, min(10000, claim_amount))

            else:  # major
                # Pareto/t distribution for large claims
                shape, scale = 2.5, 8000  # Heavy tail for catastrophic claims
                claim_amount = scale * np.random.pareto(shape)
                claim_amount = max(8000, min(50000, claim_amount))

            # Age-based severity adjustments
            if age == '16-25':
                claim_amount *= 1.3  # Young drivers have more severe accidents
            elif age == '65+':
                claim_amount *= 1.2  # Seniors more vulnerable to injury

            # Vehicle type severity
            if veh_type == 'sports_car':
                claim_amount *= 1.2  # Performance vehicles, higher repair costs
            elif veh_type == 'luxury_sedan':
                claim_amount *= 1.3  # Luxury vehicles, higher repair costs

            claim_amount = round(claim_amount, 2)

        # 5. Enhanced demographic and risk factors
        education = np.random.choice(['none', 'high_school', 'university', 'postgraduate'],
                                   p=[0.05, 0.35, 0.45, 0.15])

        income = np.random.choice(['poverty', 'working_class', 'middle_class', 'upper_class'],
                                p=[0.08, 0.32, 0.45, 0.15])

        # Credit score correlated with income and behavior
        if income == 'upper_class':
            credit_score = np.random.beta(8, 2)  # High credit scores
        elif income == 'middle_class':
            credit_score = np.random.beta(6, 3)
        elif income == 'working_class':
            credit_score = np.random.beta(4, 4)
        else:
            credit_score = np.random.beta(2, 6)  # Lower credit scores
        credit_score = round(credit_score, 3)

        vehicle_ownership = 1.0 if random.random() < 0.85 else 0.0  # Most own vehicles
        vehicle_year = np.random.choice(['before_2010', '2010-2015', '2016-2020', 'after_2020'],
                                      p=[0.2, 0.3, 0.3, 0.2])

        # Marital status correlated with age
        if age == '16-25':
            married_prob = 0.1
        elif age == '26-39':
            married_prob = 0.5
        elif age == '40-64':
            married_prob = 0.7
        else:
            married_prob = 0.6

        married = 1.0 if random.random() < married_prob else 0.0
        children = 1.0 if random.random() < (0.6 if married else 0.2) else 0.0

        # Risk behaviors (correlated with demographics)
        base_speeding = 0.8 if age == '16-25' else 0.3
        speeding_violations = np.random.poisson(base_speeding)

        base_dui = 0.05 if age in ['16-25', '26-39'] else 0.02
        duis = np.random.poisson(base_dui)

        base_accidents = 0.4 if age == '16-25' else 0.15
        past_accidents = np.random.poisson(base_accidents)

        # Postal code simulation (simplified)
        postal_code = np.random.choice([10238, 32765, 45678, 54321, 67890, 78901])

        data.append([
            row_id, age, gender, region, experience, education, income, credit_score,
            vehicle_ownership, vehicle_year, married, children, postal_code, float(mileage),
            veh_type, safety_rating, speeding_violations, duis, past_accidents,
            float(outcome), claim_amount
        ])

    columns = [
        'ID', 'AGE', 'GENDER', 'REGION', 'DRIVING_EXPERIENCE', 'EDUCATION', 'INCOME', 'CREDIT_SCORE',
        'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'MARRIED', 'CHILDREN', 'POSTAL_CODE', 'ANNUAL_MILEAGE',
        'VEHICLE_TYPE', 'SAFETY_RATING', 'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS',
        'OUTCOME', 'CLAIM_AMOUNT'
    ]

    df = pd.DataFrame(data, columns=columns)
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    output_path = r'c:\Users\mn3g24\OneDrive - University of Southampton\Desktop\projects\InsurePrice\InsurePrice\Enhanced_Synthetic_Car_Insurance_Claims.csv'
    df.to_csv(output_path, index=False)
    print(f"Enhanced synthetic dataset saved to {output_path}")

    # Comprehensive Validation
    print("\n" + "="*60)
    print("ENHANCED SYNTHETIC DATA VALIDATION")
    print("="*60)

    print(f"Total Records: {len(df):,}")
    print(f"Overall Claim Frequency: {df.OUTCOME.mean():.3f}")
    print(f"Average Claim Amount (claims > £0): £{df[df.CLAIM_AMOUNT > 0]['CLAIM_AMOUNT'].mean():,.2f}")

    print("\nCLAIM FREQUENCY BY DEMOGRAPHICS:")
    print("-" * 40)
    print("By Age Group:")
    age_freq = df.groupby('AGE')['OUTCOME'].mean().sort_values(ascending=False)
    for age, freq in age_freq.items():
        print(f"  {age}: {freq:.3f}")

    print("\nBy Region (Top 5 riskiest):")
    region_freq = df.groupby('REGION')['OUTCOME'].mean().sort_values(ascending=False).head(5)
    for region, freq in region_freq.items():
        print(f"  {region}: {freq:.3f}")

    print("\nBy Vehicle Type:")
    veh_freq = df.groupby('VEHICLE_TYPE')['OUTCOME'].mean().sort_values(ascending=False)
    for veh, freq in veh_freq.items():
        print(f"  {veh}: {freq:.3f}")

    print("\nCLAIM SEVERITY ANALYSIS:")
    print("-" * 40)
    claims_df = df[df.CLAIM_AMOUNT > 0]

    print("Average Claim Amount by Age:")
    age_claims = claims_df.groupby('AGE')['CLAIM_AMOUNT'].mean().sort_values(ascending=False)
    for age, amt in age_claims.items():
        print(f"  {age}: £{amt:,.2f}")

    print("\nAverage Claim Amount by Vehicle Type:")
    veh_claims = claims_df.groupby('VEHICLE_TYPE')['CLAIM_AMOUNT'].mean().sort_values(ascending=False)
    for veh, amt in veh_claims.items():
        print(f"  {veh}: £{amt:,.2f}")

    print("\nClaim Amount Distribution (percentiles):")
    claim_amounts = claims_df['CLAIM_AMOUNT']
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"  {p}th percentile: £{np.percentile(claim_amounts, p):,.2f}")

    print("\nRISK FACTOR CORRELATIONS:")
    print("-" * 40)
    print(".3f")
    print(".3f")
    print(".3f")

    print("\nDATA QUALITY CHECKS:")
    print("-" * 40)
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate IDs: {df.ID.duplicated().sum()}")
    print(".1f")

    print("\nSAMPLE RECORDS:")
    print("-" * 40)
    sample_cols = ['AGE', 'REGION', 'VEHICLE_TYPE', 'ANNUAL_MILEAGE', 'OUTCOME', 'CLAIM_AMOUNT']
    print(df[sample_cols].head(10).to_string(index=False))

    print("\n" + "="*60)
    print("ENHANCED FEATURES ADDED:")
    print("- Geographic risk factors (11 UK regions)")
    print("- Enhanced vehicle types (6 categories)")
    print("- Safety rating factor")
    print("- Mixture distribution claim severity modeling")
    print("- More sophisticated demographic correlations")
    print("- Realistic risk behavior patterns")
    print("="*60)
