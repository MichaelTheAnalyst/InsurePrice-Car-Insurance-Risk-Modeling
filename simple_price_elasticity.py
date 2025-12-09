"""
Simple Price Elasticity Simulation

Models customer reaction to price changes with different elasticity by risk segment.

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
    print("💰 PRICE ELASTICITY SIMULATION (GOLD LEVEL)")
    print("=" * 60)
    print("Modeling customer reaction to price changes")
    print("=" * 60)

    # Load data
    df = pd.read_csv('Sample_Priced_Policies.csv')
    print(f"✅ Loaded {len(df):,} customer records")
    print(".2f")

    # Create customer segments based on risk scores
    conditions = [
        (df['RISK_SCORE'] <= 0.2),
        (df['RISK_SCORE'] <= 0.3),
        (df['RISK_SCORE'] <= 0.4),
        (df['RISK_SCORE'] > 0.4)
    ]
    choices = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    df['segment'] = np.select(conditions, choices, default='Medium Risk')

    # Price elasticities by segment (more negative = more sensitive)
    elasticities = {
        'Low Risk': -0.3,      # Less price sensitive
        'Medium Risk': -0.8,   # Moderately sensitive
        'High Risk': -1.2,     # More sensitive
        'Very High Risk': -1.5 # Very sensitive
    }

    print("\n📊 CUSTOMER SEGMENTS & PRICE ELASTICITIES:")
    print("-" * 50)
    for segment, elasticity in elasticities.items():
        count = (df['segment'] == segment).sum()
        pct = count / len(df) * 100
        print(".1f")

    # Simulate price changes from -30% to +30%
    price_changes = np.arange(-0.3, 0.31, 0.05)

    print("\n📈 SIMULATING PRICE ELASTICITY")
    print(f"Testing {len(price_changes)} price change scenarios...")

    results = []

    for price_change_pct in price_changes:
        # Calculate segment-wise retention rates
        total_weighted_retention = 0
        total_weight = 0

        for segment, elasticity in elasticities.items():
            segment_data = df[df['segment'] == segment]
            if len(segment_data) > 0:
                # Price elasticity formula: %ΔQuantity = Elasticity × %ΔPrice
                demand_change_pct = elasticity * price_change_pct

                # Convert to retention rate (1 + demand change)
                retention_rate = 1 + demand_change_pct
                retention_rate = np.clip(retention_rate, 0.05, 1.0)  # Min 5% retention

                # Weight by segment size
                weight = len(segment_data)
                total_weighted_retention += retention_rate * weight
                total_weight += weight

        # Overall retention rate
        avg_retention = total_weighted_retention / total_weight

        # Calculate financial impact
        avg_premium = df['CALCULATED_PREMIUM'].mean()
        new_avg_premium = avg_premium * (1 + price_change_pct)

        # Revenue change per customer
        revenue_change = new_avg_premium * avg_retention - avg_premium

        # Profit change (assuming 12% profit margin)
        profit_margin = 0.12
        profit_change = revenue_change * profit_margin

        results.append({
            'price_change_pct': price_change_pct,
            'retention_rate': avg_retention,
            'revenue_change': revenue_change,
            'profit_change': profit_change
        })

    results_df = pd.DataFrame(results)

    print("✅ Simulation complete!")
    print(".0f")
    print(".0f")

    # Find optimal pricing strategy
    valid_points = results_df[results_df['retention_rate'] > 0.8]  # Keep retention > 80%
    if len(valid_points) > 0:
        optimal_idx = valid_points['profit_change'].idxmax()
        optimal = valid_points.loc[optimal_idx]

        print("
🏆 OPTIMAL PRICING STRATEGY FOUND:"        print(".1f")
        print(".1f")
        print(".2f")
        print(".2f")
    else:
        print("\n⚠️ No optimal strategy found maintaining 80% retention")

    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('💰 Price Elasticity Simulation - Customer Reaction to Price Changes', fontsize=16, fontweight='bold')

    # Plot 1: Retention vs Price Change
    axes[0].plot(price_changes * 100, results_df['retention_rate'], 'b-o', linewidth=2, markersize=6, label='Customer Retention')
    axes[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Retention Threshold')
    axes[0].set_xlabel('Price Change (%)')
    axes[0].set_ylabel('Customer Retention Rate')
    axes[0].set_title('How Customers React to Price Changes', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Profit Impact vs Price Change
    axes[1].plot(price_changes * 100, results_df['profit_change'], 'g-s', linewidth=2, markersize=6, label='Profit Change')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Break-even')
    if len(valid_points) > 0:
        opt_price_change = optimal['price_change_pct'] * 100
        opt_profit = optimal['profit_change']
        axes[1].plot(opt_price_change, opt_profit, 'r*', markersize=15, label='Optimal Strategy')
    axes[1].set_xlabel('Price Change (%)')
    axes[1].set_ylabel('Profit Change per Customer (£)')
    axes[1].set_title('Profit Optimization with Customer Behavior', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('price_elasticity_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n📊 KEY BUSINESS INSIGHTS:")
    print("-" * 40)
    print("• Price Elasticity varies by risk segment:")
    print("  - Low Risk: -0.3 (less sensitive to price changes)")
    print("  - Medium Risk: -0.8 (moderately sensitive)")
    print("  - High Risk: -1.2 (very sensitive)")
    print("  - Very High Risk: -1.5 (extremely sensitive)")

    print("\\n• Optimal Strategy: Small price increases maximize profit")
    print("  while maintaining high customer retention")

    print("\\n• Revenue Management: Balance between price and retention")
    print("  Too large increases cause significant customer churn")

    print("\\n• Risk-Based Pricing: Different strategies for different segments")
    print("  High-risk customers need more careful price management")

    print("\\n✅ PRICE ELASTICITY SIMULATION COMPLETE")
    print("Generated: price_elasticity_simulation.png")
    print("\\n💡 This gold-level feature enables data-driven pricing optimization!")

if __name__ == "__main__":
    main()
