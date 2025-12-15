import pandas as pd
import numpy as np
import os

def generate_eco_data():
    """
    Augments the existing village population with Eco-Driving Telematics.
    Used for the Carbon-to-Credit Gamification prototype.
    """
    print("Loading base population...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    pop_path = os.path.join(data_dir, 'The_Village_Population.csv')
    
    if not os.path.exists(pop_path):
        print("Error: Village Population not found. Run generate_village_population.py first.")
        return

    df_pop = pd.read_csv(pop_path)
    n_users = len(df_pop)
    
    np.random.seed(2030)
    
    # 1. Driving Smoothness (0.0 = Aggressive, 1.0 = Perfect Silk)
    # Correlated with Age: Older drivers tend to be smoother (on average)
    age_factor = (df_pop['age'] - 18) / 80.0
    base_smoothness = np.random.beta(5, 2, size=n_users) # Skewed towards good
    smoothness_score = np.clip(base_smoothness * 0.8 + (age_factor * 0.2), 0.1, 1.0)
    
    # 2. Average RPM (Revolutions Per Minute)
    # Inverse to smoothness. Aggressive drivers hold gears longer.
    # Base idle/cruise is ~2000.
    avg_rpm = 2000 + ((1.0 - smoothness_score) * 1500) + np.random.normal(0, 200, size=n_users)
    avg_rpm = np.clip(avg_rpm, 1200, 5000)
    
    # 3. Idling Percentage (Time spent at 0mph while engine on)
    # City drivers (Young Pros) might idle more.
    idling_pct = np.random.beta(2, 10, size=n_users) # 0.0 to 0.4
    
    # 4. Carbon Credits Wallet (Starting Balance)
    # Random initial distribution of "InsureCoins"
    wallet_balance = np.random.exponential(50, size=n_users).astype(int)
    
    # Create Dataframe
    df_eco = pd.DataFrame({
        'user_id': df_pop['user_id'],
        'avg_rpm': np.round(avg_rpm, 0).astype(int),
        'smoothness_score': np.round(smoothness_score, 2),
        'idling_pct': np.round(idling_pct, 3),
        'carbon_credits_balance': wallet_balance
    })
    
    output_path = os.path.join(data_dir, 'Driver_Eco_Profiles.csv')
    df_eco.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df_eco):,} eco-driving profiles at {output_path}")
    
    # Stats
    green_drivers = len(df_eco[df_eco['avg_rpm'] < 2000])
    print(f"🌱 'Green' Drivers (Avg RPM < 2000): {green_drivers} ({green_drivers/n_users:.1%})")

if __name__ == "__main__":
    generate_eco_data()
