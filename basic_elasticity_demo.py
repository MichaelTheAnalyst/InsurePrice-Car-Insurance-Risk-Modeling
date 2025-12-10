"""
Basic Price Elasticity Demonstration

Shows how customers react to price changes with different elasticities by risk segment.

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
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

def main():
    print("💰 PRICE ELASTICITY SIMULATION - GOLD LEVEL FEATURE")
    print("=" * 60)

    # Load data
    df = pd.read_csv('Sample_Priced_Policies.csv')
    print(f"Loaded {len(df)} customer records")

    # Create segments
    conditions = [
        (df['RISK_SCORE'] <= 0.2),
        (df['RISK_SCORE'] <= 0.3),
        (df['RISK_SCORE'] <= 0.4),
        (df['RISK_SCORE'] > 0.4)
    ]
    choices = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    df['segment'] = np.select(conditions, choices, default='Medium Risk')

    # Price elasticities (more negative = more sensitive)
    elasticities = {
        'Low Risk': -0.3,      # Less sensitive
        'Medium Risk': -0.8,   # Moderately sensitive
        'High Risk': -1.2,     # Very sensitive
        'Very High Risk': -1.5 # Extremely sensitive
    }

    print("\nCustomer Segments & Elasticities:")
    for segment, elasticity in elasticities.items():
        count = (df['segment'] == segment).sum()
        pct = count / len(df) * 100
        print(f"  {segment}: {count} customers ({pct:.1f}%), Elasticity: {elasticity}")

    # Simulate price changes
    price_changes = [-0.3, -0.2, -0.1, 0, 0.05, 0.1, 0.15, 0.2, 0.3]
    results = []

    for price_change_pct in price_changes:
        # Calculate weighted retention across segments
        total_retention = 0
        total_weight = 0

        for segment, elasticity in elasticities.items():
            segment_data = df[df['segment'] == segment]
            if len(segment_data) > 0:
                # Elasticity formula: % change in demand = elasticity * % change in price
                demand_change = elasticity * price_change_pct
                retention = 1 + demand_change
                retention = np.clip(retention, 0.05, 1.0)  # Min 5% retention

                weight = len(segment_data)
                total_retention += retention * weight
                total_weight += weight

        avg_retention = total_retention / total_weight

        # Financial calculations
        avg_premium = df['CALCULATED_PREMIUM'].mean()
        new_premium = avg_premium * (1 + price_change_pct)
        revenue_change = new_premium * avg_retention - avg_premium
        profit_change = revenue_change * 0.12  # 12% profit margin

        results.append({
            'price_change': price_change_pct * 100,
            'retention': avg_retention,
            'profit_change': profit_change
        })

    results_df = pd.DataFrame(results)
    print(f"\nSimulated {len(price_changes)} price change scenarios")

    # Find optimal strategy
    valid_points = results_df[results_df['retention'] > 0.8]
    if len(valid_points) > 0:
        optimal = valid_points.loc[valid_points['profit_change'].idxmax()]
        print("\nOptimal Strategy:")
        print(f"  Price Change: {optimal['price_change']:.0f}%")
        print(f"  Expected Retention: {optimal['retention']:.1%}")
        print(f"  Profit Change: £{optimal['profit_change']:.2f} per customer")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Price Elasticity: Customer Reaction to Price Changes', fontsize=14, fontweight='bold')

    # Retention plot
    ax1.plot(results_df['price_change'], results_df['retention'], 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Retention Target')
    ax1.set_xlabel('Price Change (%)')
    ax1.set_ylabel('Customer Retention Rate')
    ax1.set_title('Customer Retention vs Price Change')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Profit plot
    ax2.plot(results_df['price_change'], results_df['profit_change'], 'g-s', linewidth=2, markersize=6)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Break-even')
    if len(valid_points) > 0:
        ax2.plot(optimal['price_change'], optimal['profit_change'], 'r*', markersize=15, label='Optimal')
    ax2.set_xlabel('Price Change (%)')
    ax2.set_ylabel('Profit Change per Customer (£)')
    ax2.set_title('Profit Impact vs Price Change')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('price_elasticity_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nKey Insights:")
    print("• Low-risk customers: Less sensitive to price changes (elasticity = -0.3)")
    print("• High-risk customers: Very sensitive to price changes (elasticity = -1.5)")
    print("• Optimal strategy: Small price increases maximize profit with minimal churn")
    print("• Revenue optimization requires balancing price vs customer retention")

    print("\n✅ Price Elasticity Simulation Complete!")
    print("This gold-level feature enables data-driven pricing optimization.")

if __name__ == "__main__":
    main()
