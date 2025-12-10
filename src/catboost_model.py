"""
CatBoost Model with Categorical Embeddings
============================================

CatBoost advantages for insurance data:
- Native categorical feature handling (no one-hot encoding needed)
- Ordered boosting reduces overfitting
- Built-in handling of missing values
- Categorical embeddings capture complex relationships

Expected AUC improvement: +2-4%

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
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import warnings
import os

warnings.filterwarnings('ignore')


class CatBoostRiskModel:
    """
    CatBoost-based risk model with categorical embeddings.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.cat_features = []
        
    def prepare_features(self, df: pd.DataFrame):
        """Prepare features with proper categorical handling."""
        df = df.copy()
        
        # Define categorical columns
        categorical_cols = ['GENDER', 'VEHICLE_TYPE', 'VEHICLE_OWNERSHIP', 'MARRIED', 
                           'CHILDREN', 'EDUCATION', 'INCOME', 'REGION']
        
        # Define numerical columns
        numerical_cols = ['AGE', 'DRIVING_EXPERIENCE', 'ANNUAL_MILEAGE', 'CREDIT_SCORE',
                         'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS', 'VEHICLE_YEAR',
                         'SAFETY_RATING', 'POSTAL_CODE']
        
        # Convert numerical columns
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Keep categorical as strings (CatBoost handles them)
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Unknown')
        
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
        
        # Build feature list
        exclude_cols = ['OUTCOME', 'ID', 'CLAIM_AMOUNT', 'CLAIM_COST']
        
        feature_cols = []
        cat_feature_indices = []
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            if col in categorical_cols:
                feature_cols.append(col)
                cat_feature_indices.append(len(feature_cols) - 1)
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                feature_cols.append(col)
        
        self.feature_names = feature_cols
        self.cat_features = cat_feature_indices
        
        return df[feature_cols], cat_feature_indices
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train CatBoost model with categorical embeddings."""
        
        # Create Pool objects (CatBoost's data format)
        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
        
        eval_pool = None
        if X_val is not None:
            eval_pool = Pool(X_val, y_val, cat_features=self.cat_features)
        
        # CatBoost parameters optimized for insurance data
        self.model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            
            # Categorical embedding settings
            one_hot_max_size=10,  # One-hot for small cardinality
            
            # Prevent overfitting
            early_stopping_rounds=50,
            
            # Performance
            task_type='CPU',
            verbose=False
        )
        
        self.model.fit(train_pool, eval_set=eval_pool)
        
        return self
    
    def predict_proba(self, X):
        """Get probability predictions."""
        pool = Pool(X, cat_features=self.cat_features)
        return self.model.predict_proba(pool)[:, 1]
    
    def get_feature_importance(self):
        """Get feature importance from the model."""
        importance = self.model.get_feature_importance()
        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)


def load_data():
    """Load the insurance data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'processed', 
                            'Enhanced_Synthetic_Car_Insurance_Claims.csv')
    
    if not os.path.exists(data_path):
        data_path = os.path.join(script_dir, '..', 
                                'Enhanced_Synthetic_Car_Insurance_Claims.csv')
    
    return pd.read_csv(data_path)


def compare_models():
    """Compare CatBoost with baseline models."""
    
    print("🎯 CATBOOST WITH CATEGORICAL EMBEDDINGS")
    print("=" * 60)
    
    # Load data
    df = load_data()
    print(f"📂 Loaded data: {len(df)} records")
    
    # Prepare CatBoost model
    catboost_model = CatBoostRiskModel()
    X, cat_indices = catboost_model.prepare_features(df)
    y = df['OUTCOME']
    
    print(f"📋 Features: {len(catboost_model.feature_names)}")
    print(f"📋 Categorical features: {len(cat_indices)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_cb, X_val, y_train_cb, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Baseline: Random Forest
    print("\n" + "=" * 60)
    print("🔬 BASELINE: Random Forest (Optimized)")
    print("=" * 60)
    
    # Prepare numeric-only data for RF
    X_train_rf = X_train.copy()
    X_test_rf = X_test.copy()
    
    for col in X_train_rf.columns:
        if X_train_rf[col].dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X_train_rf[col] = le.fit_transform(X_train_rf[col].astype(str))
            X_test_rf[col] = le.transform(X_test_rf[col].astype(str))
    
    rf_model = RandomForestClassifier(
        n_estimators=261,
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_rf, y_train)
    rf_proba = rf_model.predict_proba(X_test_rf)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)
    
    print(f"\n   Random Forest AUC: {rf_auc:.4f}")
    print(f"   Random Forest Gini: {2*rf_auc - 1:.4f}")
    
    # CatBoost
    print("\n" + "=" * 60)
    print("🚀 CATBOOST WITH CATEGORICAL EMBEDDINGS")
    print("=" * 60)
    
    print("\n   Training CatBoost...")
    catboost_model.train(X_train_cb, y_train_cb, X_val, y_val)
    
    cb_proba = catboost_model.predict_proba(X_test)
    cb_auc = roc_auc_score(y_test, cb_proba)
    
    print(f"\n   CatBoost AUC: {cb_auc:.4f}")
    print(f"   CatBoost Gini: {2*cb_auc - 1:.4f}")
    
    # Cross-validation for more robust estimate
    print("\n   Running 5-fold cross-validation...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
        
        cb = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            verbose=False
        )
        
        train_pool = Pool(X_cv_train, y_cv_train, cat_features=cat_indices)
        cb.fit(train_pool)
        
        proba = cb.predict_proba(Pool(X_cv_val, cat_features=cat_indices))[:, 1]
        cv_scores.append(roc_auc_score(y_cv_val, proba))
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"   CV AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    # Feature Importance
    print("\n" + "=" * 60)
    print("📊 TOP 10 FEATURE IMPORTANCES (CatBoost)")
    print("=" * 60)
    
    importance = catboost_model.get_feature_importance()
    print(importance.head(10).to_string(index=False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 IMPROVEMENT SUMMARY")
    print("=" * 60)
    
    improvement = (cb_auc - rf_auc) * 100
    
    print(f"""
   Random Forest (Optimized):  {rf_auc:.4f}
   CatBoost (Categorical):     {cb_auc:.4f}
   
   Improvement:                {improvement:+.2f}%
   
   CatBoost Advantages Used:
   ├── Native categorical handling (no one-hot encoding)
   ├── Ordered boosting (reduces prediction shift)
   ├── Symmetric trees (faster inference)
   └── Automatic feature interactions
""")
    
    print("=" * 60)
    
    if cb_auc > rf_auc:
        print(f"\n🏆 WINNER: CatBoost (AUC: {cb_auc:.4f})")
    else:
        print(f"\n🏆 WINNER: Random Forest (AUC: {rf_auc:.4f})")
    
    return {
        'rf_auc': rf_auc,
        'catboost_auc': cb_auc,
        'improvement_pct': improvement,
        'cv_auc': cv_mean
    }


if __name__ == "__main__":
    # Install catboost if not present
    try:
        import catboost
    except ImportError:
        print("Installing CatBoost...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'catboost', '-q'])
        import catboost
    
    results = compare_models()
    
    print("\n✅ CatBoost model training complete!")

