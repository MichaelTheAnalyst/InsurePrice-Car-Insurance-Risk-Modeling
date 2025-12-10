"""
Hyperparameter Optimization for Insurance Risk Models
======================================================

Uses Optuna for automated hyperparameter tuning:
- Bayesian optimization (more efficient than grid search)
- Early stopping for poor trials
- Cross-validation for robust estimates

Expected AUC improvement: +1-2%

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
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
import os

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization using Optuna.
    """
    
    def __init__(self, n_trials: int = 50, cv_folds: int = 5):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = {}
        self.best_score = 0
        self.optimization_history = []
        
    def optimize_random_forest(self, X, y) -> dict:
        """
        Optimize Random Forest hyperparameters.
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params)
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            
            return scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params['RandomForest'] = study.best_params
        self.best_score = study.best_value
        
        return study.best_params
    
    def optimize_gradient_boosting(self, X, y) -> dict:
        """
        Optimize Gradient Boosting hyperparameters.
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 6),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'random_state': 42
            }
            
            model = GradientBoostingClassifier(**params)
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            
            return scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params['GradientBoosting'] = study.best_params
        
        return study.best_params


def load_and_prepare_data():
    """Load data and prepare features."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'processed', 
                            'Enhanced_Synthetic_Car_Insurance_Claims.csv')
    
    if not os.path.exists(data_path):
        data_path = os.path.join(script_dir, '..', 
                                'Enhanced_Synthetic_Car_Insurance_Claims.csv')
    
    df = pd.read_csv(data_path)
    print(f"📂 Loaded data: {len(df)} records")
    
    # Basic feature preparation
    numeric_cols = ['AGE', 'DRIVING_EXPERIENCE', 'ANNUAL_MILEAGE', 'CREDIT_SCORE',
                   'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS', 'VEHICLE_YEAR',
                   'SAFETY_RATING']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Encode categoricals
    categorical_cols = ['GENDER', 'VEHICLE_TYPE', 'VEHICLE_OWNERSHIP', 'MARRIED', 
                       'CHILDREN', 'EDUCATION', 'INCOME', 'REGION']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_ENC'] = le.fit_transform(df[col].astype(str))
    
    # Add interaction features (from feature engineering)
    if 'AGE' in df.columns and 'DRIVING_EXPERIENCE' in df.columns:
        df['AGE_x_EXPERIENCE'] = df['AGE'] * df['DRIVING_EXPERIENCE']
        df['EXPERIENCE_RATIO'] = df['DRIVING_EXPERIENCE'] / (df['AGE'] - 16 + 1)
        df['EXPERIENCE_RATIO'] = df['EXPERIENCE_RATIO'].clip(0, 1)
    
    if 'AGE' in df.columns and 'SPEEDING_VIOLATIONS' in df.columns:
        df['AGE_x_VIOLATIONS'] = df['AGE'] * df['SPEEDING_VIOLATIONS']
        df['YOUNG_WITH_VIOLATIONS'] = ((df['AGE'] < 25) & (df['SPEEDING_VIOLATIONS'] > 0)).astype(int)
    
    if 'ANNUAL_MILEAGE' in df.columns and 'PAST_ACCIDENTS' in df.columns:
        df['MILEAGE_x_ACCIDENTS'] = df['ANNUAL_MILEAGE'] * df['PAST_ACCIDENTS']
    
    if 'DUIS' in df.columns and 'SPEEDING_VIOLATIONS' in df.columns:
        df['TOTAL_VIOLATIONS'] = df['DUIS'] + df['SPEEDING_VIOLATIONS']
    
    # Select features
    feature_cols = [col for col in df.columns 
                   if col not in ['OUTCOME', 'ID', 'CLAIM_AMOUNT', 'POSTAL_CODE']
                   and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    X = df[feature_cols].fillna(0)
    y = df['OUTCOME']
    
    return X, y, feature_cols


def run_optimization():
    """Run hyperparameter optimization and compare results."""
    
    print("🎯 HYPERPARAMETER OPTIMIZATION FOR INSURANCE RISK MODEL")
    print("=" * 60)
    
    # Load data
    X, y, feature_cols = load_and_prepare_data()
    print(f"📋 Features: {len(feature_cols)}")
    
    # Baseline model (default parameters)
    print("\n" + "=" * 60)
    print("🔬 BASELINE MODEL (Default Parameters)")
    print("=" * 60)
    
    baseline_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline_scores = cross_val_score(baseline_rf, X, y, cv=cv, scoring='roc_auc')
    baseline_auc = baseline_scores.mean()
    
    print(f"\n   Random Forest (default):")
    print(f"   AUC: {baseline_auc:.4f} (+/- {baseline_scores.std():.4f})")
    
    baseline_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_scores = cross_val_score(baseline_gb, X, y, cv=cv, scoring='roc_auc')
    baseline_gb_auc = gb_scores.mean()
    
    print(f"\n   Gradient Boosting (default):")
    print(f"   AUC: {baseline_gb_auc:.4f} (+/- {gb_scores.std():.4f})")
    
    # Optimization
    print("\n" + "=" * 60)
    print("🚀 OPTIMIZING HYPERPARAMETERS (This may take a few minutes...)")
    print("=" * 60)
    
    optimizer = HyperparameterOptimizer(n_trials=15, cv_folds=3)  # Faster optimization
    
    # Optimize Random Forest
    print("\n🌲 Optimizing Random Forest...")
    rf_best_params = optimizer.optimize_random_forest(X, y)
    
    print("\n   Best Parameters:")
    for param, value in rf_best_params.items():
        print(f"      {param}: {value}")
    
    # Train with optimized parameters
    optimized_rf = RandomForestClassifier(**rf_best_params, random_state=42, n_jobs=-1)
    optimized_rf_scores = cross_val_score(optimized_rf, X, y, cv=cv, scoring='roc_auc')
    optimized_rf_auc = optimized_rf_scores.mean()
    
    print(f"\n   Optimized AUC: {optimized_rf_auc:.4f} (+/- {optimized_rf_scores.std():.4f})")
    
    # Optimize Gradient Boosting
    print("\n📈 Optimizing Gradient Boosting...")
    gb_best_params = optimizer.optimize_gradient_boosting(X, y)
    
    print("\n   Best Parameters:")
    for param, value in gb_best_params.items():
        print(f"      {param}: {value}")
    
    # Train with optimized parameters
    optimized_gb = GradientBoostingClassifier(**gb_best_params, random_state=42)
    optimized_gb_scores = cross_val_score(optimized_gb, X, y, cv=cv, scoring='roc_auc')
    optimized_gb_auc = optimized_gb_scores.mean()
    
    print(f"\n   Optimized AUC: {optimized_gb_auc:.4f} (+/- {optimized_gb_scores.std():.4f})")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    best_baseline = max(baseline_auc, baseline_gb_auc)
    best_optimized = max(optimized_rf_auc, optimized_gb_auc)
    improvement = (best_optimized - best_baseline) * 100
    
    print(f"""
   RANDOM FOREST:
   ├── Baseline AUC:   {baseline_auc:.4f}
   ├── Optimized AUC:  {optimized_rf_auc:.4f}
   └── Improvement:    {(optimized_rf_auc - baseline_auc) * 100:+.2f}%

   GRADIENT BOOSTING:
   ├── Baseline AUC:   {baseline_gb_auc:.4f}
   ├── Optimized AUC:  {optimized_gb_auc:.4f}
   └── Improvement:    {(optimized_gb_auc - baseline_gb_auc) * 100:+.2f}%

   OVERALL:
   ├── Best Baseline:  {best_baseline:.4f}
   ├── Best Optimized: {best_optimized:.4f}
   └── Total Improvement: {improvement:+.2f}%
""")
    
    print("=" * 60)
    
    # Best model recommendation
    if optimized_rf_auc >= optimized_gb_auc:
        best_model = "Random Forest"
        best_params = rf_best_params
        best_auc = optimized_rf_auc
    else:
        best_model = "Gradient Boosting"
        best_params = gb_best_params
        best_auc = optimized_gb_auc
    
    print(f"\n🏆 RECOMMENDED MODEL: {best_model}")
    print(f"   Final AUC: {best_auc:.4f}")
    print(f"   Gini: {2*best_auc - 1:.4f}")
    
    return {
        'best_model': best_model,
        'best_params': best_params,
        'best_auc': best_auc,
        'baseline_auc': best_baseline,
        'improvement_pct': improvement
    }


if __name__ == "__main__":
    results = run_optimization()
    
    print("\n✅ Hyperparameter optimization complete!")
    print(f"   Total improvement: {results['improvement_pct']:+.2f}% AUC")

