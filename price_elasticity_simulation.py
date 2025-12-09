"""
Step 12: Price Elasticity Simulation (Gold Level)

Advanced modeling of customer reaction to price changes including:
- Price elasticity of demand analysis
- Customer churn prediction based on price increases
- Revenue optimization with customer behavior
- Customer lifetime value considerations
- Optimal pricing strategy recommendations

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
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')


class PriceElasticitySimulator:
    """
    Gold-level price elasticity simulation for insurance pricing.

    Models how customers react to price changes and optimizes pricing strategy.
    """

    def __init__(self, base_data_path='Sample_Priced_Policies.csv'):
        """
        Initialize price elasticity simulator.

        Args:
            base_data_path: Path to current pricing data
        """
        self.base_data_path = base_data_path
        self.pricing_data = None
        self.customer_segments = None

        print("💰 PRICE ELASTICITY SIMULATOR (GOLD LEVEL)")
        print("=" * 60)
        print("Modeling customer reaction to price changes")
        print("Optimizing revenue while considering customer behavior")
        print("=" * 60)

    def load_pricing_data(self):
        """Load current pricing data and create customer segments."""
        try:
            self.pricing_data = pd.read_csv(self.base_data_path)
            print(f"✅ Loaded {len(self.pricing_data):,} customer pricing records")

            # Create customer segments based on risk profiles
            self._create_customer_segments()
            return True

        except FileNotFoundError:
            print("❌ Pricing data not found. Please run pricing engine first.")
            return False

    def _create_customer_segments(self):
        """Create customer segments with different price sensitivities."""

        df = self.pricing_data.copy()

        # Create risk-based segments
        conditions = [
            (df['RISK_SCORE'] <= 0.2),
            (df['RISK_SCORE'] <= 0.3),
            (df['RISK_SCORE'] <= 0.4),
            (df['RISK_SCORE'] > 0.4)
        ]
        choices = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
        df['risk_segment'] = np.select(conditions, choices, default='Medium Risk')

        # Add price sensitivity factors based on segments
        segment_sensitivities = {
            'Low Risk': {
                'elasticity': -0.3,  # Less price sensitive
                'churn_base_rate': 0.05,
                'clv_multiplier': 1.8,
                'description': 'Price insensitive, loyal customers'
            },
            'Medium Risk': {
                'elasticity': -0.8,  # Moderately sensitive
                'churn_base_rate': 0.08,
                'clv_multiplier': 1.2,
                'description': 'Balanced sensitivity and loyalty'
            },
            'High Risk': {
                'elasticity': -1.2,  # More sensitive
                'churn_base_rate': 0.12,
                'clv_multiplier': 0.9,
                'description': 'Price sensitive, higher churn risk'
            },
            'Very High Risk': {
                'elasticity': -1.5,  # Very sensitive
                'churn_base_rate': 0.18,
                'clv_multiplier': 0.7,
                'description': 'Highly price sensitive, highest churn risk'
            }
        }

        # Apply segment characteristics
        df['price_elasticity'] = df['risk_segment'].map(lambda x: segment_sensitivities[x]['elasticity'])
        df['churn_base_rate'] = df['risk_segment'].map(lambda x: segment_sensitivities[x]['churn_base_rate'])
        df['clv_multiplier'] = df['risk_segment'].map(lambda x: segment_sensitivities[x]['clv_multiplier'])
        df['segment_description'] = df['risk_segment'].map(lambda x: segment_sensitivities[x]['description'])

        # Calculate customer lifetime value (CLV) estimates
        avg_years = 5  # Average customer lifetime
        avg_annual_premium = df['CALCULATED_PREMIUM'].mean()
        df['estimated_clv'] = avg_years * avg_annual_premium * df['clv_multiplier']

        self.customer_segments = df

        print(f"✅ Created {len(df['risk_segment'].unique())} customer segments:")
        for segment in df['risk_segment'].unique():
            count = (df['risk_segment'] == segment).sum()
            pct = count / len(df) * 100
            elasticity = segment_sensitivities[segment]['elasticity']
            churn_rate = segment_sensitivities[segment]['churn_base_rate']
            print(".1f")

    def simulate_price_elasticity(self, price_changes=np.arange(-0.3, 0.31, 0.05)):
        """
        Simulate customer demand response to various price changes.

        Args:
            price_changes: Array of price change percentages (-30% to +30%)
        """

        print("\\n📈 PRICE ELASTICITY SIMULATION")
        print("-" * 50)

        results = []

        for price_change_pct in price_changes:
            price_multiplier = 1 + price_change_pct

            # Calculate segment-wise responses
            segment_results = {}

            for segment in self.customer_segments['risk_segment'].unique():
                segment_data = self.customer_segments[self.customer_segments['risk_segment'] == segment]

                # Calculate demand change using price elasticity
                avg_elasticity = segment_data['price_elasticity'].mean()
                demand_change_pct = avg_elasticity * price_change_pct

                # Ensure demand doesn't go below 0 or above reasonable bounds
                demand_change_pct = np.clip(demand_change_pct, -0.95, 0.2)  # Max 95% drop, 20% increase

                # Calculate retention rate (1 - churn rate)
                retention_rate = 1 + demand_change_pct  # Demand change = retention change
                retention_rate = np.clip(retention_rate, 0.05, 1.0)  # Min 5% retention

                # Calculate revenue impact
                original_premium = segment_data['CALCULATED_PREMIUM'].mean()
                new_premium = original_premium * price_multiplier
                revenue_change = new_premium * retention_rate - original_premium

                # Calculate profit impact (assuming 12% profit margin)
                profit_margin = 0.12
                original_profit = original_premium * profit_margin
                new_profit = new_premium * retention_rate * profit_margin
                profit_change = new_profit - original_profit

                segment_results[segment] = {
                    'original_premium': original_premium,
                    'new_premium': new_premium,
                    'retention_rate': retention_rate,
                    'revenue_change': revenue_change,
                    'profit_change': profit_change,
                    'customer_count': len(segment_data),
                    'elasticity': avg_elasticity
                }

            # Aggregate across all segments
            total_customers = len(self.customer_segments)
            weighted_retention = sum(r['retention_rate'] * r['customer_count'] for r in segment_results.values()) / total_customers
            weighted_revenue_change = sum(r['revenue_change'] * r['customer_count'] for r in segment_results.values()) / total_customers
            weighted_profit_change = sum(r['profit_change'] * r['customer_count'] for r in segment_results.values()) / total_customers

            results.append({
                'price_change_pct': price_change_pct,
                'price_multiplier': price_multiplier,
                'overall_retention': weighted_retention,
                'revenue_change_per_customer': weighted_revenue_change,
                'profit_change_per_customer': weighted_profit_change,
                'segment_results': segment_results
            })

        self.elasticity_results = pd.DataFrame(results)
        print("✅ Simulated price elasticity across"        print(".0f")
        return self.elasticity_results

    def optimize_pricing_strategy(self):
        """
        Find optimal pricing strategy that maximizes profit while maintaining customer retention.
        """

        print("\\n🎯 PRICING OPTIMIZATION ANALYSIS")
        print("-" * 50)

        if self.elasticity_results is None:
            print("❌ Run elasticity simulation first")
            return

        # Find optimal price point
        profit_changes = self.elasticity_results['profit_change_per_customer']
        retention_rates = self.elasticity_results['overall_retention']
        price_changes = self.elasticity_results['price_change_pct']

        # Optimal point: Maximum profit with retention > 80%
        valid_points = self.elasticity_results[retention_rates > 0.8]
        if len(valid_points) > 0:
            optimal_idx = valid_points['profit_change_per_customer'].idxmax()
            optimal_result = valid_points.loc[optimal_idx]

            self.optimal_pricing = {
                'price_change_pct': optimal_result['price_change_pct'],
                'retention_rate': optimal_result['overall_retention'],
                'profit_increase': optimal_result['profit_change_per_customer'],
                'revenue_increase': optimal_result['revenue_change_per_customer']
            }

            print("🏆 OPTIMAL PRICING STRATEGY FOUND:")
            print(".1f")
            print(".1f")
            print(".2f")
            print(".2f")
        else:
            print("⚠️ No optimal strategy found with retention > 80%")
            self.optimal_pricing = None

        return self.optimal_pricing

def main():
    """Run comprehensive price elasticity simulation."""

    print("🚀 Step 12: Price Elasticity Simulation (Gold Level)")
    print("=" * 60)

    # Initialize simulator
    simulator = PriceElasticitySimulator()

    # Load data
    if not simulator.load_pricing_data():
        print("❌ Cannot proceed without pricing data")
        return

    # Run elasticity simulation
    elasticity_results = simulator.simulate_price_elasticity()

    # Find optimal pricing
    optimal_strategy = simulator.optimize_pricing_strategy()

    print("\\n" + "="*60)
    print("✅ Step 12 Complete: Price Elasticity Simulation")
    print("Customer reaction to price changes fully modeled!")
    print("="*60)

if __name__ == "__main__":
    main()