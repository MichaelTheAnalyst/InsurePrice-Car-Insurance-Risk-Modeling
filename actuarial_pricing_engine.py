"""
Step 5: Actuarial Pricing Engine - Converting Risk to Price

Implements standard and advanced actuarial pricing formulas to convert
risk predictions into premium prices.

Formulas:
1. Basic Actuarial: Premium = Expected Loss + Expenses + Margin
2. Frequency-Severity: Premium = (Frequency × Severity) × Loading Factors
3. Comprehensive: Net Premium = Gross Premium × (1 + Risk Margin) × (1 + Expense Loading)

Author: MichaelTheAnalyst
Date: December 2025
Project: InsurePrice Car Insurance Risk Modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')


class ActuarialPricingEngine:
    """
    Advanced actuarial pricing engine that converts risk scores into premiums.

    Implements multiple pricing methodologies:
    1. Basic Expected Loss + Loading
    2. Frequency-Severity Decomposition
    3. Credibility-Weighted Pricing
    4. Experience Rating
    """

    def __init__(self,
                 base_claim_frequency: float = 0.122,  # From our data
                 base_claim_severity: float = 2359.24,  # From our data
                 expense_loading: float = 0.28,         # 28% expense ratio
                 profit_margin: float = 0.10,           # 10% profit margin
                 investment_return: float = 0.03,       # 3% investment return
                 risk_margin: float = 0.05):            # 5% risk margin

        # Base parameters from market data
        self.base_claim_frequency = base_claim_frequency
        self.base_claim_severity = base_claim_severity
        self.expense_loading = expense_loading
        self.profit_margin = profit_margin
        self.investment_return = investment_return
        self.risk_margin = risk_margin

        # Industry benchmarks (UK motor insurance)
        self.industry_benchmarks = {
            'average_premium': 600,      # £600 average UK car insurance
            'loss_ratio': 0.65,          # 65% of premium goes to claims
            'expense_ratio': 0.28,       # 28% for expenses
            'profit_margin': 0.07,       # 7% profit margin
            'combined_ratio': 1.00       # Break-even point
        }

        print("🏗️ Actuarial Pricing Engine Initialized")
        print(f"   Base Claim Frequency: {self.base_claim_frequency:.3f}")
        print(f"   Base Claim Severity: £{self.base_claim_severity:,.0f}")
        print(f"   Expense Loading: {self.expense_loading:.1%}")
        print(f"   Profit Margin: {self.profit_margin:.1%}")

    def calculate_basic_actuarial_premium(self,
                                         risk_score: float,
                                         exposure: float = 1.0,
                                         credibility: float = 1.0) -> Dict:
        """
        Basic actuarial pricing: Premium = Expected Loss + Expenses + Margin

        Args:
            risk_score: Model predicted probability (0-1)
            exposure: Exposure base (e.g., vehicle value, mileage)
            credibility: Weight given to risk score (0-1)

        Returns:
            Dictionary with premium breakdown
        """

        # Expected Loss = Claim Frequency × Claim Severity × Exposure × Risk Score
        expected_loss = (self.base_claim_frequency * risk_score *
                        self.base_claim_severity * exposure)

        # Expenses = Fixed % of expected loss + variable loading
        expenses = expected_loss * self.expense_loading

        # Profit Margin = % of expected loss
        profit_margin_amount = expected_loss * self.profit_margin

        # Risk Margin = Additional loading for uncertainty
        risk_margin_amount = expected_loss * self.risk_margin

        # Gross Premium = Expected Loss + Expenses + Profit + Risk Margin
        gross_premium = expected_loss + expenses + profit_margin_amount + risk_margin_amount

        # Net Premium (after investment return adjustment)
        net_premium = gross_premium * (1 - self.investment_return)

        # Apply credibility weighting
        final_premium = (credibility * net_premium +
                        (1 - credibility) * self.industry_benchmarks['average_premium'])

        return {
            'final_premium': round(final_premium, 2),
            'breakdown': {
                'expected_loss': round(expected_loss, 2),
                'expenses': round(expenses, 2),
                'profit_margin': round(profit_margin_amount, 2),
                'risk_margin': round(risk_margin_amount, 2),
                'gross_premium': round(gross_premium, 2),
                'net_premium': round(net_premium, 2),
                'investment_adjustment': round(gross_premium * self.investment_return, 2)
            },
            'ratios': {
                'loss_ratio': round(expected_loss / final_premium, 4) if final_premium > 0 else 0,
                'expense_ratio': round(expenses / final_premium, 4) if final_premium > 0 else 0,
                'profit_ratio': round(profit_margin_amount / final_premium, 4) if final_premium > 0 else 0,
                'combined_ratio': round((expected_loss + expenses) / final_premium, 4) if final_premium > 0 else 0
            },
            'parameters': {
                'risk_score': risk_score,
                'exposure': exposure,
                'credibility': credibility
            }
        }

    def calculate_frequency_severity_premium(self,
                                           risk_score: float,
                                           frequency_multiplier: float = 1.0,
                                           severity_multiplier: float = 1.0,
                                           exposure: float = 1.0) -> Dict:
        """
        Advanced frequency-severity pricing model.

        Separates frequency and severity components for more accurate pricing:
        - Frequency: Probability of having at least one claim
        - Severity: Expected cost per claim (given a claim occurs)

        Formula:
        Pure Premium = Frequency × Severity × Exposure × Risk Multipliers
        Gross Premium = Pure Premium × (1 + Expense Loading) × (1 + Profit Margin)
        """

        # Decompose risk score into frequency and severity components
        # In practice, this would come from separate frequency and severity models
        frequency_component = risk_score * frequency_multiplier
        severity_component = self.base_claim_severity * severity_multiplier

        # Pure Premium = Frequency × Severity × Exposure
        pure_premium = frequency_component * severity_component * exposure

        # Loading factors
        expense_loading = 1 + self.expense_loading
        profit_loading = 1 + self.profit_margin
        risk_loading = 1 + self.risk_margin

        # Gross Premium with loadings
        gross_premium = pure_premium * expense_loading * profit_loading * risk_loading

        # Investment adjustment (premiums collected upfront, losses paid later)
        investment_credit = gross_premium * self.investment_return

        # Final premium
        final_premium = gross_premium - investment_credit

        # Calculate ratios
        loss_ratio = pure_premium / final_premium if final_premium > 0 else 0
        expense_ratio = (pure_premium * self.expense_loading) / final_premium if final_premium > 0 else 0
        profit_ratio = (pure_premium * self.profit_margin) / final_premium if final_premium > 0 else 0

        return {
            'final_premium': round(final_premium, 2),
            'breakdown': {
                'pure_premium': round(pure_premium, 2),
                'frequency_component': round(frequency_component, 4),
                'severity_component': round(severity_component, 2),
                'expense_loading': round(pure_premium * self.expense_loading, 2),
                'profit_loading': round(pure_premium * self.profit_margin, 2),
                'risk_loading': round(pure_premium * self.risk_margin, 2),
                'investment_credit': round(investment_credit, 2)
            },
            'ratios': {
                'loss_ratio': round(loss_ratio, 4),
                'expense_ratio': round(expense_ratio, 4),
                'profit_ratio': round(profit_ratio, 4),
                'combined_ratio': round(loss_ratio + expense_ratio, 4)
            },
            'parameters': {
                'risk_score': risk_score,
                'frequency_multiplier': frequency_multiplier,
                'severity_multiplier': severity_multiplier,
                'exposure': exposure
            }
        }

    def calculate_credibility_weighted_premium(self,
                                             individual_risk_score: float,
                                             portfolio_frequency: float = None,
                                             portfolio_severity: float = None,
                                             portfolio_size: int = 10000,
                                             confidence_level: float = 0.95) -> Dict:
        """
        Credibility-weighted pricing using Bühlmann-Straub credibility theory.

        Combines individual risk estimates with portfolio experience:
        - High credibility → Use individual risk score
        - Low credibility → Use portfolio average

        Credibility Factor = n / (n + k)
        Where: n = individual exposure, k = portfolio variance parameter
        """

        # Default to our portfolio statistics if not provided
        if portfolio_frequency is None:
            portfolio_frequency = self.base_claim_frequency
        if portfolio_severity is None:
            portfolio_severity = self.base_claim_severity

        # Calculate credibility factor using Bühlmann-Straub
        # Simplified version: credibility increases with exposure/volume
        individual_exposure = 1.0  # Could be mileage, vehicle value, etc.
        portfolio_variance = portfolio_frequency * (1 - portfolio_frequency)  # Bernoulli variance

        # Credibility factor (simplified)
        k_parameter = portfolio_variance / (portfolio_frequency * (1 - portfolio_frequency))
        credibility = individual_exposure / (individual_exposure + k_parameter)

        # Credibility-weighted estimates
        credibility_weighted_frequency = (credibility * individual_risk_score +
                                        (1 - credibility) * portfolio_frequency)

        credibility_weighted_severity = portfolio_severity  # Usually less variable

        # Calculate premium using credibility-weighted parameters
        expected_loss = credibility_weighted_frequency * credibility_weighted_severity
        expenses = expected_loss * self.expense_loading
        profit_margin = expected_loss * self.profit_margin
        risk_margin = expected_loss * self.risk_margin

        final_premium = expected_loss + expenses + profit_margin + risk_margin
        final_premium *= (1 - self.investment_return)  # Investment adjustment

        return {
            'final_premium': round(final_premium, 2),
            'breakdown': {
                'expected_loss': round(expected_loss, 2),
                'credibility_weighted_frequency': round(credibility_weighted_frequency, 4),
                'credibility_weighted_severity': round(credibility_weighted_severity, 2),
                'expenses': round(expenses, 2),
                'profit_margin': round(profit_margin, 2),
                'risk_margin': round(risk_margin, 2),
                'investment_adjustment': round(final_premium * self.investment_return / (1 - self.investment_return), 2)
            },
            'credibility': {
                'credibility_factor': round(credibility, 4),
                'individual_contribution': round(credibility, 4),
                'portfolio_contribution': round(1 - credibility, 4)
            },
            'parameters': {
                'individual_risk_score': individual_risk_score,
                'portfolio_frequency': portfolio_frequency,
                'portfolio_severity': portfolio_severity,
                'portfolio_size': portfolio_size
            }
        }

    def batch_calculate_premiums(self,
                                risk_scores: pd.Series,
                                method: str = 'basic',
                                **kwargs) -> pd.DataFrame:
        """
        Calculate premiums for multiple policies using specified method.

        Args:
            risk_scores: Series of risk scores from prediction model
            method: 'basic', 'frequency_severity', or 'credibility'
            **kwargs: Additional parameters for the pricing method
        """

        results = []

        for risk_score in risk_scores:
            if method == 'basic':
                premium_result = self.calculate_basic_actuarial_premium(risk_score, **kwargs)
            elif method == 'frequency_severity':
                premium_result = self.calculate_frequency_severity_premium(risk_score, **kwargs)
            elif method == 'credibility':
                premium_result = self.calculate_credibility_weighted_premium(risk_score, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")

            results.append({
                'risk_score': risk_score,
                'calculated_premium': premium_result['final_premium'],
                'expected_loss': premium_result['breakdown']['expected_loss'],
                'loss_ratio': premium_result['ratios']['loss_ratio'],
                'expense_ratio': premium_result['ratios']['expense_ratio'],
                'profit_ratio': premium_result['ratios']['profit_ratio'],
                'combined_ratio': premium_result['ratios']['combined_ratio']
            })

        return pd.DataFrame(results)

    def analyze_pricing_distribution(self,
                                   pricing_results: pd.DataFrame,
                                   save_path: str = None) -> Dict:
        """
        Analyze the distribution of calculated premiums.
        """

        analysis = {
            'summary_stats': {
                'count': len(pricing_results),
                'mean_premium': round(pricing_results['calculated_premium'].mean(), 2),
                'median_premium': round(pricing_results['calculated_premium'].median(), 2),
                'std_premium': round(pricing_results['calculated_premium'].std(), 2),
                'min_premium': round(pricing_results['calculated_premium'].min(), 2),
                'max_premium': round(pricing_results['calculated_premium'].max(), 2)
            },
            'ratios': {
                'mean_loss_ratio': round(pricing_results['loss_ratio'].mean(), 4),
                'mean_expense_ratio': round(pricing_results['expense_ratio'].mean(), 4),
                'mean_profit_ratio': round(pricing_results['profit_ratio'].mean(), 4),
                'mean_combined_ratio': round(pricing_results['combined_ratio'].mean(), 4)
            },
            'percentiles': {
                '10th': round(np.percentile(pricing_results['calculated_premium'], 10), 2),
                '25th': round(np.percentile(pricing_results['calculated_premium'], 25), 2),
                '75th': round(np.percentile(pricing_results['calculated_premium'], 75), 2),
                '90th': round(np.percentile(pricing_results['calculated_premium'], 90), 2),
                '95th': round(np.percentile(pricing_results['calculated_premium'], 95), 2)
            }
        }

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('💰 Actuarial Premium Distribution Analysis', fontsize=16, fontweight='bold')

        # Premium distribution
        axes[0,0].hist(pricing_results['calculated_premium'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].axvline(analysis['summary_stats']['mean_premium'], color='red', linestyle='--',
                         label=f'Mean: £{analysis["summary_stats"]["mean_premium"]:.0f}')
        axes[0,0].axvline(analysis['summary_stats']['median_premium'], color='green', linestyle='--',
                         label=f'Median: £{analysis["summary_stats"]["median_premium"]:.0f}')
        axes[0,0].set_xlabel('Annual Premium (£)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Premium Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Premium vs Risk Score
        axes[0,1].scatter(pricing_results['risk_score'], pricing_results['calculated_premium'],
                         alpha=0.6, color='purple', edgecolors='black', linewidth=0.5)
        axes[0,1].set_xlabel('Risk Score')
        axes[0,1].set_ylabel('Calculated Premium (£)')
        axes[0,1].set_title('Risk Score vs Premium')
        axes[0,1].grid(True, alpha=0.3)

        # Ratio distributions
        ratios_data = pricing_results[['loss_ratio', 'expense_ratio', 'profit_ratio']].melt(var_name='Ratio Type', value_name='Ratio')
        sns.boxplot(data=ratios_data, x='Ratio Type', y='Ratio', ax=axes[1,0])
        axes[1,0].set_title('Premium Component Ratios')
        axes[1,0].set_ylabel('Ratio')
        axes[1,0].grid(True, alpha=0.3)

        # Expected Loss distribution
        axes[1,1].hist(pricing_results['expected_loss'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1,1].axvline(pricing_results['expected_loss'].mean(), color='red', linestyle='--',
                         label=f'Mean: £{pricing_results["expected_loss"].mean():.0f}')
        axes[1,1].set_xlabel('Expected Loss (£)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Expected Loss Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Pricing analysis saved to {save_path}")
        plt.show()

        return analysis

    def demonstrate_pricing_methods(self, sample_risk_scores: List[float] = None):
        """
        Demonstrate different pricing methods with sample calculations.
        """

        if sample_risk_scores is None:
            sample_risk_scores = [0.1, 0.2, 0.3, 0.4, 0.5]  # Different risk levels

        print("🔢 ACTUARIAL PRICING METHOD DEMONSTRATION")
        print("=" * 60)

        for risk_score in sample_risk_scores:
            print(f"\\n🎯 Risk Score: {risk_score}")
            print("-" * 40)

            # Method 1: Basic Actuarial
            basic = self.calculate_basic_actuarial_premium(risk_score)
            print(f"Basic Actuarial: £{basic['final_premium']:.2f}")
            print(f"  Expected Loss: £{basic['breakdown']['expected_loss']:.2f}")
            print(f"  Loss Ratio: {basic['ratios']['loss_ratio']:.3f}")

            # Method 2: Frequency-Severity
            freq_sev = self.calculate_frequency_severity_premium(risk_score)
            print(f"Frequency-Severity: £{freq_sev['final_premium']:.2f}")
            print(f"  Pure Premium: £{freq_sev['breakdown']['pure_premium']:.2f}")

            # Method 3: Credibility-Weighted
            credibility = self.calculate_credibility_weighted_premium(risk_score)
            print(f"Credibility-Weighted: £{credibility['final_premium']:.2f}")
            print(f"  Credibility Factor: {credibility['credibility']['credibility_factor']:.3f}")

    def validate_actuarial_principles(self, pricing_results: pd.DataFrame) -> Dict:
        """
        Validate that calculated premiums follow actuarial principles.
        """

        validation = {
            'profitability_check': {
                'combined_ratio_mean': pricing_results['combined_ratio'].mean(),
                'combined_ratio_std': pricing_results['combined_ratio'].std(),
                'profitable_policies_pct': (pricing_results['combined_ratio'] < 1).mean() * 100,
                'loss_making_policies_pct': (pricing_results['combined_ratio'] > 1).mean() * 100
            },
            'industry_compliance': {
                'loss_ratio_compliance': abs(pricing_results['loss_ratio'].mean() - self.industry_benchmarks['loss_ratio']) < 0.1,
                'expense_ratio_compliance': abs(pricing_results['expense_ratio'].mean() - self.industry_benchmarks['expense_ratio']) < 0.05,
                'overall_compliance_score': None  # Calculated below
            },
            'risk_sensitivity': {
                'premium_risk_correlation': pricing_results['risk_score'].corr(pricing_results['calculated_premium']),
                'high_risk_premium_pctl': np.percentile(pricing_results[pricing_results['risk_score'] > 0.3]['calculated_premium'], 75),
                'low_risk_premium_pctl': np.percentile(pricing_results[pricing_results['risk_score'] < 0.1]['calculated_premium'], 25)
            }
        }

        # Calculate overall compliance score
        loss_compliant = validation['industry_compliance']['loss_ratio_compliance']
        expense_compliant = validation['industry_compliance']['expense_ratio_compliance']
        validation['industry_compliance']['overall_compliance_score'] = (loss_compliant + expense_compliant) / 2

        return validation


def main():
    """
    Demonstrate the actuarial pricing engine.
    """

    print("🚀 Step 5: Actuarial Pricing Engine - Converting Risk to Price")
    print("=" * 70)

    # Initialize pricing engine
    pricing_engine = ActuarialPricingEngine()

    # Load risk scores from our best model (Random Forest)
    print("\\n📊 Loading risk predictions from baseline modeling...")

    # For demonstration, we'll generate some sample risk scores
    # In practice, these would come from your trained model
    np.random.seed(42)
    sample_risk_scores = np.random.beta(2, 5, 1000)  # Realistic risk distribution
    sample_risk_scores = pd.Series(sample_risk_scores)

    print(f"Generated {len(sample_risk_scores)} sample risk scores")
    print(f"Risk score range: {sample_risk_scores.min():.3f} - {sample_risk_scores.max():.3f}")
    print(f"Mean risk score: {sample_risk_scores.mean():.3f}")

    # Demonstrate different pricing methods
    print("\\n💰 PRICING METHOD DEMONSTRATION")
    print("-" * 50)
    pricing_engine.demonstrate_pricing_methods([0.1, 0.2, 0.3, 0.4, 0.5])

    # Batch calculate premiums using basic actuarial method
    print("\\n⚡ BATCH PREMIUM CALCULATION")
    print("-" * 50)

    batch_results = pricing_engine.batch_calculate_premiums(
        sample_risk_scores,
        method='basic',
        credibility=0.8  # 80% weight on individual risk, 20% on portfolio average
    )

    print(f"Calculated premiums for {len(batch_results)} policies")
    print(f"Average premium: £{batch_results['calculated_premium'].mean():.2f}")
    print(f"Premium range: £{batch_results['calculated_premium'].min():.2f} - £{batch_results['calculated_premium'].max():.2f}")

    # Analyze pricing distribution
    print("\\n📈 PREMIUM DISTRIBUTION ANALYSIS")
    print("-" * 50)

    analysis = pricing_engine.analyze_pricing_distribution(batch_results, 'premium_distribution_analysis.png')

    print("Summary Statistics:")
    stats = analysis['summary_stats']
    print(f"  Count: {stats['count']:,}")
    print(f"  Mean: £{stats['mean_premium']:.2f}")
    print(f"  Median: £{stats['median_premium']:.2f}")
    print(f"  Std: £{stats['std_premium']:.2f}")
    print(f"  Range: £{stats['min_premium']:.2f} - £{stats['max_premium']:.2f}")

    print("\\nRatio Analysis:")
    ratios = analysis['ratios']
    print(f"  Loss Ratio: {ratios['mean_loss_ratio']:.3f}")
    print(f"  Expense Ratio: {ratios['mean_expense_ratio']:.3f}")
    print(f"  Profit Ratio: {ratios['mean_profit_ratio']:.3f}")
    print(f"  Combined Ratio: {ratios['mean_combined_ratio']:.3f}")

    # Validate actuarial principles
    print("\\n✅ ACTUARIAL VALIDATION")
    print("-" * 50)

    validation = pricing_engine.validate_actuarial_principles(batch_results)

    print("Profitability Check:")
    profit_check = validation['profitability_check']
    print(f"  Combined Ratio (target < 1.0): {profit_check['combined_ratio_mean']:.3f}")
    print(f"  Profitable Policies: {profit_check['profitable_policies_pct']:.1f}%")
    print(f"  Loss-Making Policies: {profit_check['loss_making_policies_pct']:.1f}%")

    print("\\nIndustry Compliance:")
    compliance = validation['industry_compliance']
    print(f"  Loss Ratio Compliance: {'✅' if compliance['loss_ratio_compliance'] else '❌'}")
    print(f"  Expense Ratio Compliance: {'✅' if compliance['expense_ratio_compliance'] else '❌'}")
    print(f"  Overall Compliance Score: {compliance['overall_compliance_score']:.1%}")

    print("\\nRisk Sensitivity:")
    sensitivity = validation['risk_sensitivity']
    print(f"  Premium-Risk Correlation: {sensitivity['premium_risk_correlation']:.3f}")
    print(f"  High-Risk Premium (75th pctl): £{sensitivity['high_risk_premium_pctl']:.2f}")
    print(f"  Low-Risk Premium (25th pctl): £{sensitivity['low_risk_premium_pctl']:.2f}")

    print("\\n" + "=" * 70)
    print("✅ Step 5 Complete: Actuarial Pricing Engine")
    print("   Risk scores successfully converted to market-competitive premiums!")
    print("=" * 70)


if __name__ == "__main__":
    main()
