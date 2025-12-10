"""
Model Training and Persistence Script
=====================================

Trains machine learning models and saves them to disk for production use.
This eliminates the need to retrain models on every API startup.

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
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# Constants
MODEL_DIR = 'models'
DATA_FILE = 'Enhanced_Synthetic_Car_Insurance_Claims.csv'


def ensure_model_directory():
    """Create models directory if it doesn't exist."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"✅ Created directory: {MODEL_DIR}/")
    else:
        print(f"📁 Model directory exists: {MODEL_DIR}/")


def load_and_prepare_data():
    """Load and prepare data for training."""
    print("\n📊 Loading training data...")
    
    df = pd.read_csv(DATA_FILE)
    print(f"   Loaded {len(df):,} records")
    
    # Exclude non-feature columns
    exclude_cols = ['ID', 'POSTAL_CODE', 'CLAIM_AMOUNT']
    feature_cols = [col for col in df.columns if col not in exclude_cols + ['OUTCOME']]
    
    X = df[feature_cols].copy()
    y = df['OUTCOME'].copy()
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Positive class rate: {y.mean():.3f}")
    
    return X, y, feature_cols


def encode_categorical_features(X):
    """Encode categorical features and return encoders."""
    print("\n🔧 Encoding categorical features...")
    
    label_encoders = {}
    X_encoded = X.copy()
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"   {col}: {len(le.classes_)} classes")
    
    return X_encoded, label_encoders


def scale_numerical_features(X):
    """Scale numerical features and return scaler."""
    print("\n📏 Scaling numerical features...")
    
    X_scaled = X.copy()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler = StandardScaler()
    X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    print(f"   Scaled {len(numerical_cols)} numerical columns")
    
    return X_scaled, scaler, numerical_cols


def train_random_forest(X, y):
    """Train Random Forest model."""
    print("\n🌲 Training Random Forest model...")
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    train_auc = roc_auc_score(y_train, rf_model.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, rf_model.predict_proba(X_val)[:, 1])
    
    print(f"   Training AUC: {train_auc:.4f}")
    print(f"   Validation AUC: {val_auc:.4f}")
    
    # Retrain on full data for production
    print("   Retraining on full dataset for production...")
    rf_model.fit(X, y)
    
    return rf_model


def save_artifacts(rf_model, scaler, label_encoders, feature_names, numerical_cols):
    """Save all model artifacts to disk."""
    print("\n💾 Saving model artifacts...")
    
    # Save Random Forest model
    model_path = os.path.join(MODEL_DIR, 'random_forest_model.joblib')
    joblib.dump(rf_model, model_path)
    print(f"   ✅ Saved: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Saved: {scaler_path}")
    
    # Save label encoders
    encoders_path = os.path.join(MODEL_DIR, 'label_encoders.joblib')
    joblib.dump(label_encoders, encoders_path)
    print(f"   ✅ Saved: {encoders_path}")
    
    # Save feature names and metadata
    metadata = {
        'feature_names': feature_names,
        'numerical_cols': numerical_cols,
        'categorical_cols': list(label_encoders.keys()),
        'model_version': '1.0.0',
        'trained_on': pd.Timestamp.now().isoformat()
    }
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.joblib')
    joblib.dump(metadata, metadata_path)
    print(f"   ✅ Saved: {metadata_path}")
    
    return {
        'model': model_path,
        'scaler': scaler_path,
        'encoders': encoders_path,
        'metadata': metadata_path
    }


def verify_saved_models():
    """Verify saved models can be loaded correctly."""
    print("\n🔍 Verifying saved models...")
    
    try:
        # Load all artifacts
        rf_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.joblib'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
        label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.joblib'))
        metadata = joblib.load(os.path.join(MODEL_DIR, 'model_metadata.joblib'))
        
        print(f"   ✅ Random Forest model loaded (n_estimators={rf_model.n_estimators})")
        print(f"   ✅ Scaler loaded")
        print(f"   ✅ Label encoders loaded ({len(label_encoders)} encoders)")
        print(f"   ✅ Metadata loaded (version={metadata['model_version']})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🚀 InsurePrice Model Training Pipeline")
    print("=" * 60)
    
    # Create model directory
    ensure_model_directory()
    
    # Load and prepare data
    X, y, feature_cols = load_and_prepare_data()
    
    # Encode categorical features
    X_encoded, label_encoders = encode_categorical_features(X)
    
    # Scale numerical features
    X_scaled, scaler, numerical_cols = scale_numerical_features(X_encoded)
    
    # Train model
    rf_model = train_random_forest(X_scaled, y)
    
    # Save all artifacts
    saved_files = save_artifacts(
        rf_model=rf_model,
        scaler=scaler,
        label_encoders=label_encoders,
        feature_names=list(X_scaled.columns),
        numerical_cols=numerical_cols
    )
    
    # Verify
    success = verify_saved_models()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ MODEL TRAINING COMPLETE")
        print("=" * 60)
        print("\nSaved artifacts:")
        for name, path in saved_files.items():
            print(f"   • {name}: {path}")
        print("\n💡 The API will now load these pre-trained models on startup")
        print("   instead of retraining, saving 5-15 seconds per startup.")
    else:
        print("❌ MODEL TRAINING FAILED")
        print("=" * 60)
    
    return success


if __name__ == "__main__":
    main()

