import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_maintenance_data():
    """
    Augments the existing village population with Vehicle Health Telematics.
    Used for the Preventative Maintenance Bond prototype.
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
    
    np.random.seed(2026)
    
    # 1. Brake Pad Wear (%)
    # Correlated with mileage slightly, but mostly random (driving style)
    # High mileage drivers have slightly higher wear
    base_wear = np.random.beta(2, 5, size=n_users) * 100 # 0-100%
    mileage_factor = df_pop['annual_mileage'] / 20000.0
    brake_wear_pct = np.clip(base_wear + (mileage_factor * 20), 0, 100)
    
    # 2. Tyre Tread Depth (mm)
    # New is 8mm, Legal limit 1.6mm
    # Inverse correlation with mileage
    base_tread = np.random.normal(5.0, 1.5, size=n_users)
    tread_depth_mm = base_tread - (mileage_factor * 2.0)
    tread_depth_mm = np.clip(tread_depth_mm, 0.5, 8.0) # Allow some illegal tyres
    
    # 3. Battery Health (State of Health %)
    # 100% is new, <50% needs replace
    battery_health_soh = np.random.normal(85, 10, size=n_users)
    battery_health_soh = np.clip(battery_health_soh, 10, 100)
    
    # 4. Engine Oil Quality (0-100%)
    oil_quality = np.random.uniform(10, 100, size=n_users)
    
    # 5. Last Service Date
    days_ago = np.random.randint(1, 700, size=n_users)
    last_service_dates = [
        (datetime.now() - timedelta(days=int(d))).strftime('%Y-%m-%d')
        for d in days_ago
    ]
    
    # Create Dataframe
    df_maint = pd.DataFrame({
        'user_id': df_pop['user_id'],
        'brake_wear_pct': np.round(brake_wear_pct, 1),
        'tyre_tread_mm': np.round(tread_depth_mm, 2),
        'battery_soh_pct': np.round(battery_health_soh, 1),
        'oil_quality_pct': np.round(oil_quality, 1),
        'last_service_date': last_service_dates
    })
    
    output_path = os.path.join(data_dir, 'Vehicle_Maintenance_Records.csv')
    df_maint.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df_maint):,} vehicle health records at {output_path}")
    
    # Stats
    critical_tyres = len(df_maint[df_maint['tyre_tread_mm'] < 1.6])
    print(f"⚠️ Vehicles with illegal tyres (<1.6mm): {critical_tyres} ({critical_tyres/n_users:.1%})")

if __name__ == "__main__":
    generate_maintenance_data()
