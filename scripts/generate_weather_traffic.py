import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_uk_conditions(year=2025):
    """
    Generates daily weather and traffic baseline conditions for major UK regions.
    Used to seed the Risk Twin simulations with semi-realistic patterns.
    """
    np.random.seed(42)
    
    regions = ['London', 'South East', 'South West', 'East Anglia', 
               'West Midlands', 'North West', 'North East', 'Scotland', 'Wales']
    
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    days = (end_date - start_date).days + 1
    
    data = []
    
    for region in regions:
        # Latitude adjustment for weather (simple mock)
        is_north = region in ['Scotland', 'North East', 'North West']
        
        for d in range(days):
            current_date = start_date + timedelta(days=d)
            month = current_date.month
            
            # 1. Weather Logic
            # Default
            probs = [0.4, 0.4, 0.1, 0.0, 0.1] # Clear, Rain, Fog, Snow, Storm
            
            if month in [12, 1, 2]: # Winter
                probs = [0.2, 0.4, 0.2, 0.15, 0.05] if is_north else [0.25, 0.5, 0.2, 0.03, 0.02]
            elif month in [6, 7, 8]: # Summer
                probs = [0.7, 0.2, 0.05, 0.0, 0.05]
            
            # Normalize just in case
            probs = np.array(probs)
            probs /= probs.sum()
            
            weather = np.random.choice(
                ['Clear', 'Rain', 'Fog', 'Snow', 'Storm'], 
                p=probs
            )
            
            # 2. Traffic Logic
            # Weekdays have higher traffic baseline
            is_weekend = current_date.weekday() >= 5
            
            if is_weekend:
                traffic = np.random.choice(['Light', 'Moderate', 'Heavy'], p=[0.5, 0.4, 0.1])
            else:
                traffic = np.random.choice(['Moderate', 'Heavy', 'Gridlock'], p=[0.2, 0.6, 0.2])
                if region == 'London':
                    traffic = np.random.choice(['Heavy', 'Gridlock'], p=[0.4, 0.6])
            
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'region': region,
                'weather_condition': weather,
                'traffic_baseline': traffic,
                'visibility_meters': np.random.randint(50, 5000) if weather != 'Clear' else 10000
            })
            
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating UK Driving Conditions (365 days x 9 regions)...")
    df = generate_uk_conditions()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    output_path = os.path.join(data_dir, 'UK_Driving_Conditions_2025.csv')
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {len(df):,} records to {output_path}")
    print(df.head())
