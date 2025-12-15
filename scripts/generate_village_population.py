import pandas as pd
import numpy as np
import os

def generate_village_population(n_users=50000):
    """
    Generates a large synthetic population to populate the 'Village Pools'.
    Includes specific risk factors used for clustering.
    """
    np.random.seed(2025)
    
    # 1. Base IDs
    user_ids = [f"USR_{i:05d}" for i in range(1, n_users + 1)]
    
    # 2. Demographics
    ages = np.random.randint(18, 85, size=n_users)
    annual_mileage = np.random.lognormal(9, 0.6, size=n_users).astype(int) # Mean ~8000
    
    # 3. Telematics / Behavioral Data
    # Night driving: skewed to 0, some high
    night_driving_pct = np.random.beta(1, 5, size=n_users) # mostly low
    
    # Weekend driving: normal around 0.28 (2/7 days)
    weekend_driving_pct = np.random.normal(0.28, 0.1, size=n_users)
    weekend_driving_pct = np.clip(weekend_driving_pct, 0, 1)
    
    # Fatigue Score (mock sensor data): 0 (Fresh) to 10 (Zombie)
    fatigue_score = np.random.gamma(2, 1.5, size=n_users)
    fatigue_score = np.clip(fatigue_score, 0, 10)
    
    # 4. Assign Villages (Pre-calculate for speed)
    villages = []
    for i in range(n_users):
        age = ages[i]
        miles = annual_mileage[i]
        night = night_driving_pct[i]
        
        if age < 25:
            villages.append('young_pros')
        elif miles < 5000:
            villages.append('weekend_warriors')
        elif night > 0.3:
            villages.append('night_owls')
        else:
            villages.append('safe_commuters')
            
    df = pd.DataFrame({
        'user_id': user_ids,
        'age': ages,
        'annual_mileage': annual_mileage,
        'night_driving_pct': np.round(night_driving_pct, 3),
        'weekend_driving_pct': np.round(weekend_driving_pct, 3),
        'fatigue_score': np.round(fatigue_score, 1),
        'assigned_village': villages
    })
    
    return df

if __name__ == "__main__":
    print("Generating 50,000 synthetic villagers...")
    df = generate_village_population()
    
    # Save to data/ directory relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    output_path = os.path.join(data_dir, 'The_Village_Population.csv')
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {len(df):,} records to {output_path}")
    print("\nVillage Distribution:")
    print(df['assigned_village'].value_counts(normalize=True))
