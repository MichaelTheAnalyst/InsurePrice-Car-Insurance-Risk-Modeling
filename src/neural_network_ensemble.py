"""
Neural Network Ensemble for Insurance Risk Modeling
=====================================================

OVERVIEW
--------
This module implements an advanced ensemble approach for predicting insurance claim 
risk by combining deep learning with traditional machine learning models. The ensemble
strategy typically outperforms any single model by leveraging the strengths of different
algorithmic approaches.

MODELS INCLUDED
---------------
1. Neural Network with Categorical Embeddings
   - Learns dense representations of categorical features (e.g., vehicle type, region)
   - Captures non-linear interactions between features
   - Deep architecture with batch normalization and dropout for regularization

2. Random Forest
   - Robust tree-based ensemble that handles feature interactions naturally
   - Less prone to overfitting with proper hyperparameter tuning
   - Provides feature importance rankings

3. CatBoost (if available)
   - State-of-the-art gradient boosting algorithm
   - Excellent handling of categorical features
   - Often achieves best individual model performance

4. Meta-Learner (Stacking)
   - Logistic regression that learns optimal weights for combining model predictions
   - Produces final calibrated probability estimates

WHY ENSEMBLE?
-------------
- Different models capture different patterns in the data
- Reduces variance and can reduce bias
- More robust predictions than single models
- Typical AUC improvement: +3-5% over single best model

HOW TO USE
----------
1. Basic usage:
   >>> from neural_network_ensemble import NeuralNetworkEnsemble, load_data
   >>> df = load_data()
   >>> ensemble = NeuralNetworkEnsemble()
   >>> results = ensemble.train_ensemble(df)

2. Run as standalone script:
   $ python neural_network_ensemble.py

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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
import os

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InsuranceEmbeddingNet(nn.Module):
    """
    Neural network with embedding layers for categorical features.
    
    Architecture:
    - Embedding layers for each categorical feature
    - Concatenate embeddings with numerical features
    - Deep fully connected layers with dropout
    - Sigmoid output for binary classification
    """
    
    def __init__(self, num_numerical, cat_dims, embedding_dims, hidden_dims=[128, 64, 32]):
        super(InsuranceEmbeddingNet, self).__init__()
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embed_dim)
            for num_categories, embed_dim in zip(cat_dims, embedding_dims)
        ])
        
        # Calculate total input dimension
        total_embed_dim = sum(embedding_dims)
        input_dim = num_numerical + total_embed_dim
        
        # Build hidden layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.fc_layers = nn.Sequential(*layers)
    
    def forward(self, x_num, x_cat):
        # Get embeddings for categorical features
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        
        # Concatenate embeddings
        x_emb = torch.cat(embeddings, dim=1)
        
        # Concatenate with numerical features
        x = torch.cat([x_num, x_emb], dim=1)
        
        # Forward through FC layers
        return self.fc_layers(x)


class NeuralNetworkEnsemble:
    """
    Ensemble combining neural network with traditional ML models.
    """
    
    def __init__(self):
        self.nn_model = None
        self.rf_model = None
        self.catboost_model = None
        self.meta_model = None
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.num_features = []
        self.cat_features = []
        self.cat_dims = []
        self.embedding_dims = []
        
    def prepare_data(self, df: pd.DataFrame):
        """Prepare features for the ensemble."""
        df = df.copy()
        
        # Define feature types
        self.cat_features = ['GENDER', 'VEHICLE_TYPE', 'VEHICLE_OWNERSHIP', 'MARRIED', 
                            'CHILDREN', 'EDUCATION', 'INCOME', 'REGION']
        
        self.num_features = ['AGE', 'DRIVING_EXPERIENCE', 'ANNUAL_MILEAGE', 'CREDIT_SCORE',
                            'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS', 'VEHICLE_YEAR',
                            'SAFETY_RATING']
        
        # Filter to existing columns
        self.cat_features = [c for c in self.cat_features if c in df.columns]
        self.num_features = [c for c in self.num_features if c in df.columns]
        
        # Convert numerical first
        for col in self.num_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Add engineered features (after numeric conversion)
        if 'AGE' in df.columns and 'DRIVING_EXPERIENCE' in df.columns:
            df['AGE_x_EXPERIENCE'] = df['AGE'] * df['DRIVING_EXPERIENCE']
            df['EXPERIENCE_RATIO'] = df['DRIVING_EXPERIENCE'] / (df['AGE'] - 16 + 1).clip(lower=1)
            df['EXPERIENCE_RATIO'] = df['EXPERIENCE_RATIO'].clip(0, 1)
            self.num_features.extend(['AGE_x_EXPERIENCE', 'EXPERIENCE_RATIO'])
        
        if 'AGE' in df.columns and 'SPEEDING_VIOLATIONS' in df.columns:
            df['AGE_x_VIOLATIONS'] = df['AGE'] * df['SPEEDING_VIOLATIONS']
            self.num_features.append('AGE_x_VIOLATIONS')
        
        if 'DUIS' in df.columns and 'SPEEDING_VIOLATIONS' in df.columns:
            df['TOTAL_VIOLATIONS'] = df['DUIS'] + df['SPEEDING_VIOLATIONS']
            self.num_features.append('TOTAL_VIOLATIONS')
        
        # Encode categorical and track dimensions
        self.cat_dims = []
        self.embedding_dims = []
        
        for col in self.cat_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: x if x in self.label_encoders[col].classes_ 
                                       else self.label_encoders[col].classes_[0])
                df[col] = self.label_encoders[col].transform(df[col])
            
            n_cats = len(self.label_encoders[col].classes_)
            self.cat_dims.append(n_cats)
            # Embedding dimension rule: min(50, max(4, (n_cats + 1) // 2))
            self.embedding_dims.append(min(50, max(4, n_cats // 2 + 1)))
        
        return df
    
    def train_neural_network(self, X_num, X_cat, y, epochs=200, batch_size=128, lr=0.003):
        """Train the neural network model with improved architecture."""
        
        # Convert to tensors
        X_num_tensor = torch.FloatTensor(X_num).to(device)
        X_cat_tensor = torch.LongTensor(X_cat).to(device)
        y_tensor = torch.FloatTensor(y.values).reshape(-1, 1).to(device)
        
        # Create data loader
        dataset = TensorDataset(X_num_tensor, X_cat_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model with deeper architecture
        self.nn_model = InsuranceEmbeddingNet(
            num_numerical=X_num.shape[1],
            cat_dims=self.cat_dims,
            embedding_dims=self.embedding_dims,
            hidden_dims=[256, 128, 64, 32]  # Deeper network
        ).to(device)
        
        # Loss and optimizer with better settings
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(self.nn_model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop with better early stopping
        self.nn_model.train()
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_num, batch_cat, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.nn_model(batch_num, batch_cat)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            scheduler.step()
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = self.nn_model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 30:
                    break
        
        # Restore best model
        if best_state:
            self.nn_model.load_state_dict(best_state)
        
        return self
    
    def predict_nn(self, X_num, X_cat):
        """Get predictions from neural network."""
        self.nn_model.eval()
        with torch.no_grad():
            X_num_tensor = torch.FloatTensor(X_num).to(device)
            X_cat_tensor = torch.LongTensor(X_cat).to(device)
            predictions = self.nn_model(X_num_tensor, X_cat_tensor)
        return predictions.cpu().numpy().flatten()
    
    def train_ensemble(self, df, target_col='OUTCOME'):
        """Train the full ensemble."""
        
        print("🧠 TRAINING NEURAL NETWORK ENSEMBLE")
        print("=" * 60)
        
        # Prepare data
        df = self.prepare_data(df)
        
        X_num = df[self.num_features].values
        X_cat = df[self.cat_features].values
        y = df[target_col]
        
        # Scale numerical features
        X_num = self.scaler.fit_transform(X_num)
        
        # Split data
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
        
        X_num_train, X_num_test = X_num[train_idx], X_num[test_idx]
        X_cat_train, X_cat_test = X_cat[train_idx], X_cat[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 1. Train Neural Network
        print("\n1️⃣ Training Neural Network...")
        self.train_neural_network(X_num_train, X_cat_train, y_train, epochs=100)
        nn_pred_train = self.predict_nn(X_num_train, X_cat_train)
        nn_pred_test = self.predict_nn(X_num_test, X_cat_test)
        nn_auc = roc_auc_score(y_test, nn_pred_test)
        print(f"   Neural Network AUC: {nn_auc:.4f}")
        
        # 2. Train Random Forest
        print("\n2️⃣ Training Random Forest...")
        X_combined_train = np.hstack([X_num_train, X_cat_train])
        X_combined_test = np.hstack([X_num_test, X_cat_test])
        
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_split=10,
            random_state=42, n_jobs=-1
        )
        self.rf_model.fit(X_combined_train, y_train)
        rf_pred_train = self.rf_model.predict_proba(X_combined_train)[:, 1]
        rf_pred_test = self.rf_model.predict_proba(X_combined_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_pred_test)
        print(f"   Random Forest AUC: {rf_auc:.4f}")
        
        # 3. Train CatBoost (if available)
        try:
            from catboost import CatBoostClassifier
            print("\n3️⃣ Training CatBoost...")
            
            self.catboost_model = CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.05,
                random_seed=42, verbose=False
            )
            
            # Train without cat_features since data is already encoded
            self.catboost_model.fit(X_combined_train, y_train)
            
            cb_pred_train = self.catboost_model.predict_proba(X_combined_train)[:, 1]
            cb_pred_test = self.catboost_model.predict_proba(X_combined_test)[:, 1]
            cb_auc = roc_auc_score(y_test, cb_pred_test)
            print(f"   CatBoost AUC: {cb_auc:.4f}")
            
            # Stack predictions
            stack_train = np.column_stack([nn_pred_train, rf_pred_train, cb_pred_train])
            stack_test = np.column_stack([nn_pred_test, rf_pred_test, cb_pred_test])
            
        except ImportError:
            print("\n3️⃣ CatBoost not available, using 2-model ensemble...")
            cb_auc = 0
            stack_train = np.column_stack([nn_pred_train, rf_pred_train])
            stack_test = np.column_stack([nn_pred_test, rf_pred_test])
        
        # 4. Train Meta-learner (Stacking)
        print("\n4️⃣ Training Meta-learner (Stacking)...")
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(stack_train, y_train)
        
        ensemble_pred = self.meta_model.predict_proba(stack_test)[:, 1]
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        print(f"   Ensemble AUC: {ensemble_auc:.4f}")
        
        # Simple average ensemble (often works well)
        if cb_auc > 0:
            avg_pred = (nn_pred_test + rf_pred_test + cb_pred_test) / 3
        else:
            avg_pred = (nn_pred_test + rf_pred_test) / 2
        avg_auc = roc_auc_score(y_test, avg_pred)
        print(f"   Average Ensemble AUC: {avg_auc:.4f}")
        
        # Best ensemble
        best_ensemble_auc = max(ensemble_auc, avg_auc)
        
        return {
            'nn_auc': nn_auc,
            'rf_auc': rf_auc,
            'cb_auc': cb_auc,
            'stacked_ensemble_auc': ensemble_auc,
            'average_ensemble_auc': avg_auc,
            'best_ensemble_auc': best_ensemble_auc,
            'y_test': y_test,
            'ensemble_pred': ensemble_pred if ensemble_auc >= avg_auc else avg_pred
        }


def load_data():
    """Load insurance data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'processed', 
                            'Enhanced_Synthetic_Car_Insurance_Claims.csv')
    
    if not os.path.exists(data_path):
        data_path = os.path.join(script_dir, '..', 
                                'Enhanced_Synthetic_Car_Insurance_Claims.csv')
    
    return pd.read_csv(data_path)


def main():
    """Run neural network ensemble training."""
    
    print("🎯 NEURAL NETWORK ENSEMBLE FOR INSURANCE RISK")
    print("=" * 60)
    print(f"   Device: {device}")
    
    # Load data
    df = load_data()
    print(f"📂 Loaded data: {len(df)} records")
    
    # Train ensemble
    ensemble = NeuralNetworkEnsemble()
    results = ensemble.train_ensemble(df)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ENSEMBLE RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"""
   Individual Models:
   ├── Neural Network:    {results['nn_auc']:.4f}
   ├── Random Forest:     {results['rf_auc']:.4f}
   └── CatBoost:          {results['cb_auc']:.4f}

   Ensemble Methods:
   ├── Stacked Ensemble:  {results['stacked_ensemble_auc']:.4f}
   └── Average Ensemble:  {results['average_ensemble_auc']:.4f}

   🏆 BEST ENSEMBLE AUC:  {results['best_ensemble_auc']:.4f}
   🏆 BEST ENSEMBLE GINI: {2*results['best_ensemble_auc'] - 1:.4f}
""")
    
    # Compare with previous best (CatBoost alone: 0.6176)
    previous_best = 0.6176
    improvement = (results['best_ensemble_auc'] - previous_best) * 100
    
    print("=" * 60)
    print(f"   Previous Best (CatBoost): {previous_best:.4f}")
    print(f"   New Best (Ensemble):      {results['best_ensemble_auc']:.4f}")
    print(f"   Improvement:              {improvement:+.2f}%")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Install torch if needed
    try:
        import torch
    except ImportError:
        print("Installing PyTorch...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'torch', '-q'])
    
    results = main()
    print("\n✅ Neural Network Ensemble training complete!")

