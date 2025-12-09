"""
Step 3: Baseline Predictive Modeling for Claim Probability

Train and evaluate three baseline models:
- Logistic Regression
- Random Forest
- XGBoost

Metrics: AUC, Gini coefficient, Calibration curve, Feature importance

Author: MichaelTheAnalyst
Date: December 2025
Project: InsurePrice Car Insurance Risk Modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

class InsuranceClaimPredictor:
    """
    Baseline predictive modeling for car insurance claim probability.
    """

    def __init__(self, data_path='Enhanced_Synthetic_Car_Insurance_Claims.csv'):
        """Initialize the predictor with data."""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

    def load_and_preprocess_data(self):
        """Load data and perform preprocessing."""
        print("🔄 Loading and preprocessing data...")

        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df):,} records")

        # Select features for modeling
        # Exclude ID, POSTAL_CODE (too many categories), and CLAIM_AMOUNT (target leakage)
        exclude_cols = ['ID', 'POSTAL_CODE', 'CLAIM_AMOUNT']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols + ['OUTCOME']]

        # Prepare features and target
        X = self.df[feature_cols].copy()
        y = self.df['OUTCOME'].copy()

        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        self.label_encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        self.scaler = StandardScaler()
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])

        self.X = X
        self.y = y
        self.feature_names = list(X.columns)

        print(f"✅ Preprocessed {len(feature_cols)} features")
        print(f"   Categorical variables encoded: {len(categorical_cols)}")
        print(f"   Numerical variables scaled: {len(numerical_cols)}")
        print(f"   Target distribution: {y.mean():.3f} ({y.sum()}/{len(y)}) claims")

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        print("🔄 Splitting data into train/test sets...")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        print("✅ Data split complete:")
        print(f"   Train: {len(self.X_train):,} samples ({len(self.X_train)/len(self.X):.1%})")
        print(f"   Test:  {len(self.X_test):,} samples ({len(self.X_test)/len(self.X):.1%})")
        print(f"   Train claim rate: {self.y_train.mean():.3f}")
        print(f"   Test claim rate:  {self.y_test.mean():.3f}")

    def train_logistic_regression(self):
        """Train logistic regression model."""
        print("\n🔄 Training Logistic Regression...")

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_train, self.y_train)

        self.models['Logistic Regression'] = model
        print("✅ Logistic Regression trained")

    def train_random_forest(self):
        """Train random forest model."""
        print("🔄 Training Random Forest...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)

        self.models['Random Forest'] = model
        print("✅ Random Forest trained")

    def train_xgboost(self):
        """Train XGBoost model."""
        print("🔄 Training XGBoost...")

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)

        self.models['XGBoost'] = model
        print("✅ XGBoost trained")

    def evaluate_model(self, model_name):
        """Evaluate a single model."""
        model = self.models[model_name]

        # Predictions
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Metrics
        auc = roc_auc_score(self.y_test, y_pred_proba)
        gini = 2 * auc - 1

        # Store results
        self.results[model_name] = {
            'auc': auc,
            'gini': gini,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }

        print(f"\n📊 {model_name} Results:")
        print(".4f")
        print(".4f")

        return auc, gini

    def evaluate_all_models(self):
        """Evaluate all trained models."""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)

        for model_name in self.models.keys():
            self.evaluate_model(model_name)

    def plot_roc_curves(self):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))

        for model_name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            auc = result['auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Guessing')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Claim Probability Models', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves_baseline.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_calibration_curves(self):
        """Plot calibration curves for all models."""
        plt.figure(figsize=(10, 8))

        for model_name, result in self.results.items():
            y_true = self.y_test
            y_prob = result['y_pred_proba']

            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')

            plt.plot(prob_pred, prob_true, 'o-', label=model_name, linewidth=2, markersize=6)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfectly Calibrated')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual Probability')
        plt.title('Calibration Curves - Claim Probability Models', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('calibration_curves_baseline.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self):
        """Plot feature importance for tree-based models."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Random Forest feature importance
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            rf_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)

            axes[0].barh(range(len(rf_importance)), rf_importance['importance'])
            axes[0].set_yticks(range(len(rf_importance)))
            axes[0].set_yticklabels(rf_importance['feature'])
            axes[0].set_title('Random Forest - Feature Importance', fontweight='bold')
            axes[0].grid(True, alpha=0.3)

        # XGBoost feature importance
        if 'XGBoost' in self.models:
            xgb_model = self.models['XGBoost']
            xgb_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=True)

            axes[1].barh(range(len(xgb_importance)), xgb_importance['importance'])
            axes[1].set_yticks(range(len(xgb_importance)))
            axes[1].set_yticklabels(xgb_importance['feature'])
            axes[1].set_title('XGBoost - Feature Importance', fontweight='bold')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('feature_importance_baseline.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_logistic_coefficients(self):
        """Plot logistic regression coefficients."""
        if 'Logistic Regression' not in self.models:
            return

        model = self.models['Logistic Regression']
        coefficients = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', ascending=True)

        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(coefficients)), coefficients['coefficient'])
        plt.yticks(range(len(coefficients)), coefficients['feature'])
        plt.title('Logistic Regression - Feature Coefficients', fontweight='bold')
        plt.xlabel('Coefficient Value')
        plt.grid(True, alpha=0.3)

        # Color bars based on positive/negative
        for bar, coef in zip(bars, coefficients['coefficient']):
            if coef > 0:
                bar.set_color('lightcoral')
            else:
                bar.set_color('lightblue')

        plt.tight_layout()
        plt.savefig('logistic_coefficients_baseline.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_model_comparison_table(self):
        """Create a comparison table of all models."""
        print("\n" + "="*60)
        print("BASELINE MODEL COMPARISON")
        print("="*60)

        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'AUC': f"{result['auc']:.4f}",
                'Gini': f"{result['gini']:.4f}"
            })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Find best model
        best_auc_model = max(self.results.items(), key=lambda x: x[1]['auc'])[0]
        best_auc = self.results[best_auc_model]['auc']

        print(f"\n🏆 Best Model: {best_auc_model} (AUC: {best_auc:.4f})")

        return comparison_df

    def run_complete_analysis(self):
        """Run the complete baseline modeling analysis."""
        print("🚀 Starting Step 3: Baseline Predictive Modeling")
        print("="*60)

        # Data preparation
        self.load_and_preprocess_data()
        self.split_data()

        # Model training
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()

        # Model evaluation
        self.evaluate_all_models()

        # Visualizations
        print("\n📊 Generating visualizations...")
        self.plot_roc_curves()
        self.plot_calibration_curves()
        self.plot_feature_importance()
        self.plot_logistic_coefficients()

        # Model comparison
        self.create_model_comparison_table()

        print("\n" + "="*60)
        print("✅ Step 3 Complete: Baseline Modeling Analysis")
        print("="*60)
        print("📁 Generated files:")
        print("  • roc_curves_baseline.png")
        print("  • calibration_curves_baseline.png")
        print("  • feature_importance_baseline.png")
        print("  • logistic_coefficients_baseline.png")

def main():
    """Main function to run baseline modeling."""
    predictor = InsuranceClaimPredictor()
    predictor.run_complete_analysis()

if __name__ == "__main__":
    main()
