"""
Data Exploration and Claim Rate Analysis for InsurePrice

Step 2: Clean & explore the data
- Inspect missing values
- Explore key variables
- Plot claim rates by: Age, Car category, NCD (proxy), Region
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette('husl')

    # Load data
    df = pd.read_csv('Enhanced_Synthetic_Car_Insurance_Claims.csv')

    print('='*60)
    print('STEP 2: DATA CLEANING & EXPLORATION')
    print('='*60)

    print('\n1. DATASET OVERVIEW')
    print('-'*30)
    print(f'Total records: {len(df):,}')
    print(f'Columns: {len(df.columns)}')
    print(f'Features: {list(df.columns)}')

    print('\n2. MISSING VALUES INSPECTION')
    print('-'*30)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(3)
    missing_summary = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
    missing_summary = missing_summary[missing_summary['Missing'] > 0]
    if len(missing_summary) > 0:
        print(missing_summary)
    else:
        print('✅ No missing values found!')

    print('\n3. DATA TYPES & UNIQUE VALUES')
    print('-'*30)
    for col in df.columns:
        dtype = df[col].dtype
        unique = df[col].nunique()
        if dtype in ['object', 'int64', 'float64']:
            sample = df[col].iloc[0] if len(df) > 0 else 'N/A'
            print(f'{col:<20} | {str(dtype):<10} | {unique:<6} | {str(sample)[:30]}')

    print('\n4. BASIC STATISTICS')
    print('-'*30)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe().round(2))

    print('\n5. KEY VARIABLE EXPLORATION')
    print('-'*30)

    # Overall claim rate
    overall_rate = df['OUTCOME'].mean()
    print(f'Overall claim rate: {overall_rate:.3f} ({overall_rate*100:.1f}%)')

    # Categorical variables summary
    categorical_cols = ['AGE', 'GENDER', 'REGION', 'DRIVING_EXPERIENCE', 'VEHICLE_TYPE']

    print('\nCATEGORICAL VARIABLES SUMMARY:')
    for col in categorical_cols:
        print(f'\n{col}:')
        value_counts = df[col].value_counts()
        percentages = (value_counts / len(df) * 100).round(1)
        for val, count, pct in zip(value_counts.index[:5], value_counts.values[:5], percentages.values[:5]):
            print(f'  {val:<15} | {count:>5} ({pct:>4.1f}%)')

    # Create the requested plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🚗 Claim Rate Analysis by Key Risk Factors', fontsize=16, fontweight='bold')

    # 1. Claim Rate by Age
    age_claims = df.groupby('AGE')['OUTCOME'].agg(['mean', 'count', 'sum']).round(4)
    age_claims = age_claims.sort_values('mean', ascending=False)

    bars = axes[0,0].bar(range(len(age_claims)), age_claims['mean'],
                         color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0,0].set_xticks(range(len(age_claims)))
    axes[0,0].set_xticklabels(age_claims.index, rotation=45)
    axes[0,0].set_title('Claim Rate by Age Group', fontweight='bold')
    axes[0,0].set_ylabel('Claim Rate')
    axes[0,0].grid(True, alpha=0.3)

    # Add value labels and sample sizes
    for bar, rate, count, claims in zip(bars, age_claims['mean'], age_claims['count'], age_claims['sum']):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2, height + 0.005,
                      f'{rate:.3f}\n(n={count})', ha='center', va='bottom', fontsize=9)

    # 2. Claim Rate by Vehicle Type (Car Category)
    vehicle_claims = df.groupby('VEHICLE_TYPE')['OUTCOME'].agg(['mean', 'count', 'sum']).round(4)
    vehicle_claims = vehicle_claims.sort_values('mean', ascending=False)

    bars = axes[0,1].bar(range(len(vehicle_claims)), vehicle_claims['mean'],
                         color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0,1].set_xticks(range(len(vehicle_claims)))
    axes[0,1].set_xticklabels(vehicle_claims.index, rotation=45)
    axes[0,1].set_title('Claim Rate by Vehicle Type (Car Category)', fontweight='bold')
    axes[0,1].set_ylabel('Claim Rate')
    axes[0,1].grid(True, alpha=0.3)

    # Add value labels
    for bar, rate, count in zip(bars, vehicle_claims['mean'], vehicle_claims['count']):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2, height + 0.005,
                      f'{rate:.3f}\n(n={count})', ha='center', va='bottom', fontsize=9)

    # 3. Claim Rate by Region
    region_claims = df.groupby('REGION')['OUTCOME'].agg(['mean', 'count', 'sum']).round(4)
    region_claims = region_claims.sort_values('mean', ascending=False)

    bars = axes[1,0].bar(range(len(region_claims)), region_claims['mean'],
                         color='lightgreen', alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1,0].set_xticks(range(len(region_claims)))
    axes[1,0].set_xticklabels(region_claims.index, rotation=45, fontsize=9)
    axes[1,0].set_title('Claim Rate by Region', fontweight='bold')
    axes[1,0].set_ylabel('Claim Rate')
    axes[1,0].grid(True, alpha=0.3)

    # Add value labels
    for bar, rate, count in zip(bars, region_claims['mean'], region_claims['count']):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2, height + 0.005,
                      f'{rate:.3f}\n(n={count})', ha='center', va='bottom', fontsize=8)

    # 4. Claim Rate by Driving Experience (Proxy for NCD)
    exp_claims = df.groupby('DRIVING_EXPERIENCE')['OUTCOME'].agg(['mean', 'count', 'sum']).round(4)
    exp_claims = exp_claims.sort_values('mean', ascending=False)

    bars = axes[1,1].bar(range(len(exp_claims)), exp_claims['mean'],
                         color='orange', alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1,1].set_xticks(range(len(exp_claims)))
    axes[1,1].set_xticklabels(exp_claims.index, rotation=45)
    axes[1,1].set_title('Claim Rate by Driving Experience\n(Proxy for NCD/Risk Experience)', fontweight='bold')
    axes[1,1].set_ylabel('Claim Rate')
    axes[1,1].grid(True, alpha=0.3)

    # Add value labels
    for bar, rate, count in zip(bars, exp_claims['mean'], exp_claims['count']):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2, height + 0.005,
                      f'{rate:.3f}\n(n={count})', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('claim_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print('\n✅ Claim rate plots saved as claim_rate_analysis.png')
    print('\n📊 Key Findings:')
    print(f'• Overall claim rate: {overall_rate*100:.1f}%')
    print(f'• Highest risk age group: {age_claims.index[0]} ({age_claims.iloc[0]["mean"]*100:.1f}%)')
    print(f'• Highest risk vehicle: {vehicle_claims.index[0]} ({vehicle_claims.iloc[0]["mean"]*100:.1f}%)')
    print(f'• Highest risk region: {region_claims.index[0]} ({region_claims.iloc[0]["mean"]*100:.1f}%)')
    print(f'• Highest risk experience: {exp_claims.index[0]} ({exp_claims.iloc[0]["mean"]*100:.1f}%)')

    # Additional insights
    print('\n🔍 Additional Data Insights:')
    print(f'• Young drivers (16-25) are {age_claims.loc["16-25", "mean"]/age_claims.loc["40-64", "mean"]:.1f}x more likely to claim than middle-aged drivers')
    print(f'• Sports cars have {vehicle_claims.loc["sports_car", "mean"]/vehicle_claims.loc["family_sedan", "mean"]:.1f}x higher claim rate than family sedans')
    print(f'• North East has {region_claims.loc["North East", "mean"]/region_claims.loc["South West", "mean"]:.1f}x higher claim rate than South West')
    print(f'• Inexperienced drivers (0-2y) have {exp_claims.loc["0-2y", "mean"]/exp_claims.loc["30y+", "mean"]:.1f}x higher claim rate than very experienced drivers')

    print('\n📝 Notes:')
    print('• NCD (No Claims Discount) variable not available in dataset')
    print('• Using driving experience as proxy for claim history/risk experience')
    print('• All analyses based on 10,000 synthetic but statistically realistic records')
    print('• Claim rates calibrated to UK insurance industry standards')

if __name__ == "__main__":
    main()
