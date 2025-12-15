# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from typing import Dict, Optional

class CarbonGamificationEngine:
    """
    Gamifies eco-driving by converting low-RPM miles into 'InsureCoin' credits.
    """
    
    def __init__(self):
        # Load Eco Data
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, '..', '..', 'data', 'Driver_Eco_Profiles.csv')
            self.eco_data = pd.read_csv(data_path)
            self.eco_data.set_index('user_id', inplace=True)
        except Exception as e:
            print(f"Warning: Could not load Eco data: {e}")
            self.eco_data = None

    def get_driver_eco_stats(self, user_id: str) -> Dict:
        """Get live eco-telemetry for a user."""
        if self.eco_data is None or user_id not in self.eco_data.index:
            # Fallback mock
            return {
                'avg_rpm': 2450,
                'smoothness_score': 0.72,
                'idling_pct': 0.15,
                'carbon_credits_balance': 450
            }
        
        row = self.eco_data.loc[user_id]
        return row.to_dict()

    def calculate_mining_rate(self, stats: Dict) -> Dict:
        """
        Calculates the 'Mining Hashrate' (Credits per 100 miles).
        Formula: Base * (1 - RPM_Penalty) * Smoothness_Multiplier
        """
        rpm = stats['avg_rpm']
        smoothness = stats['smoothness_score']
        
        # 1. RPM Factor
        # Ideal is < 2000. Penalty scales up to 4000.
        if rpm < 2000:
            rpm_efficiency = 1.0
        elif rpm > 4000:
            rpm_efficiency = 0.0
        else:
            rpm_efficiency = 1.0 - ((rpm - 2000) / 2000)
            
        # 2. Mining Rate calculation
        base_rate = 10.0 # Credits per 100 miles
        mining_rate = base_rate * rpm_efficiency * (0.5 + smoothness)
        
        # 3. Efficiency Tier
        if mining_rate > 12:
            tier = "🍃 Carbon Negative (Elite)"
        elif mining_rate > 8:
            tier = "🌱 Eco-Friendly"
        elif mining_rate > 4:
            tier = "💨 Neutral"
        else:
            tier = "🔥 High Emitter"
            
        return {
            'mining_rate_per_100_miles': round(mining_rate, 2),
            'efficiency_score_pct': round(rpm_efficiency * 100, 1),
            'tier': tier,
            'rpm_gap': max(0, rpm - 2000)
        }
