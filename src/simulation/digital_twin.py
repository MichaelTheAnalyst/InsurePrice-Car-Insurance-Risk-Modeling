# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple

class DigitalTwinSimulator:
    """
    Simulate a driver's daily commute risk profile using Monte Carlo methods.
    incorporates weather, traffic, and driver-specific factors.
    """
    
    WEATHER_CONDITIONS = ['Clear', 'Rain', 'Fog', 'Snow', 'Storm']
    TRAFFIC_CONDITIONS = ['Light', 'Moderate', 'Heavy', 'Gridlock']
    
    # Risk Multipliers
    WEATHER_RISK = {
        'Clear': 1.0,
        'Rain': 1.3,
        'Fog': 1.8,
        'Snow': 2.5,
        'Storm': 3.0
    }
    
    TRAFFIC_RISK = {
        'Light': 0.8,
        'Moderate': 1.0,
        'Heavy': 1.4,
        'Gridlock': 1.6
    }

    def __init__(self, base_accident_prob_per_mile: float = 0.000005):
        self.base_prob = base_accident_prob_per_mile
        
        # Load Environmental Data
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, '..', '..', 'data', 'UK_Driving_Conditions_2025.csv')
            self.env_data = pd.read_csv(data_path)
        except Exception as e:
            print(f"Warning: Could not lead environment data: {e}")
            self.env_data = None

    def simulate_commute(self, 
                        distance_miles: float, 
                        n_simulations: int = 1000, 
                        driver_fatigue_level: float = 1.0,
                        region: str = 'London') -> pd.DataFrame:
        """
        Run N simulations of the commute using historical environmental data.
        """
        results = []
        
        # Filter data for region if available
        if self.env_data is not None:
            region_data = self.env_data[self.env_data['region'] == region]
            if region_data.empty:
                region_data = self.env_data # Fallback
        
        for i in range(n_simulations):
            # 1. Sample Environment from Real Data
            if self.env_data is not None and not region_data.empty:
                day_cond = region_data.sample(1).iloc[0]
                weather = day_cond['weather_condition']
                traffic = day_cond['traffic_baseline']
                date = day_cond['date']
            else:
                # Fallback to random
                weather = np.random.choice(self.WEATHER_CONDITIONS, p=[0.6, 0.2, 0.1, 0.05, 0.05])
                traffic = np.random.choice(self.TRAFFIC_CONDITIONS, p=[0.2, 0.4, 0.3, 0.1])
                date = "N/A"
            
            # 2. Calculate Specific Risk Multiplier
            w_risk = self.WEATHER_RISK.get(weather, 1.0)
            t_risk = self.TRAFFIC_RISK.get(traffic, 1.0)
            
            # 3. Total Trip Probability of Accident
            # P(Accident) = 1 - (1 - p_mile)^miles
            # Augmented by multipliers
            
            adjusted_prob_per_mile = self.base_prob * w_risk * t_risk * driver_fatigue_level
            trip_accident_prob = 1 - (1 - adjusted_prob_per_mile) ** distance_miles
            
            # 4. Monte Carlo Outcome
            is_accident = random.random() < trip_accident_prob
            
            # 5. Severity Simulation (if accident occurs)
            damage_cost = 0
            if is_accident:
                # Log-normal distribution for cost
                damage_cost = max(500, np.random.lognormal(mean=8, sigma=1.2)) # ~3000 mean
            
            results.append({
                'simulation_id': i,
                'weather': weather,
                'traffic': traffic,
                'risk_multiplier': round(w_risk * t_risk * driver_fatigue_level, 2),
                'accident_prob_percent': trip_accident_prob * 100,
                'accident_occurred': is_accident,
                'estimated_damage': round(damage_cost, 2)
            })
            
        return pd.DataFrame(results)

    def get_summary_stats(self, df_results: pd.DataFrame) -> Dict:
        """Calculate aggregate risk metrics from simulation results."""
        total_runs = len(df_results)
        accidents = df_results[df_results['accident_occurred']]
        
        return {
            'total_simulations': total_runs,
            'accident_count': len(accidents),
            'crash_probability': len(accidents) / total_runs,
            'avg_damage_if_crash': accidents['estimated_damage'].mean() if not accidents.empty else 0,
            'max_simulated_loss': df_results['estimated_damage'].max(),
            'most_dangerous_weather': df_results.groupby('weather')['accident_occurred'].mean().idxmax()
        }
