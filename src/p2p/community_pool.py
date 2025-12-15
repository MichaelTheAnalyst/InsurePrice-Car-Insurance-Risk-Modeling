# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, List

class CommunityCluster:
    """
    Logic to group users into insurance 'Villages' based on similarity.
    """
    
    VILLAGE_TYPES = {
        'safe_commuters': {'name': 'Safe Commuters', 'risk_factor': 0.8},
        'night_owls': {'name': 'Night Shift Workers', 'risk_factor': 1.2},
        'weekend_warriors': {'name': 'Weekend Drivers', 'risk_factor': 0.6},
        'young_pros': {'name': 'Young Professionals', 'risk_factor': 1.1}
    }

    def __init__(self):
        # Load Population Data
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, '..', '..', 'data', 'The_Village_Population.csv')
            self.pop_data = pd.read_csv(data_path)
            self.village_stats_cache = {}
        except Exception as e:
            print(f"Warning: Could not load village population: {e}")
            self.pop_data = None

    def assign_user_to_village(self, user_profile: Dict) -> str:
        """
        Assigns a user to a village based on their profile.
        Matches the logic used in generation script.
        """
        age = user_profile.get('age', 30)
        miles = user_profile.get('annual_mileage', 10000)
        night_pct = user_profile.get('night_driving_percent', 0.0)
        
        if age < 25:
            return 'young_pros'
        elif miles < 5000:
            return 'weekend_warriors'
        elif night_pct > 0.3:
            return 'night_owls'
        else:
            return 'safe_commuters'

    def get_village_stats(self, village_id: str) -> Dict:
        """Get stats from REAL population data."""
        details = self.VILLAGE_TYPES.get(village_id, self.VILLAGE_TYPES['safe_commuters'])
        
        if self.pop_data is None:
            # Fallback Mock
            return {
                'id': village_id,
                'name': details['name'],
                'members': 500,
                'total_pool_value': 250000,
                'risk_factor': details['risk_factor']
            }
        
        # Calculate from Data
        if village_id not in self.village_stats_cache:
            v_data = self.pop_data[self.pop_data['assigned_village'] == village_id]
            members = len(v_data)
            avg_premium = 600 * details['risk_factor'] # Base assumption
            
            self.village_stats_cache[village_id] = {
                'id': village_id,
                'name': details['name'],
                'members': members,
                'total_pool_value': members * avg_premium,
                'risk_factor': details['risk_factor'],
                'avg_fatigue': v_data['fatigue_score'].mean(),
                'avg_mileage': v_data['annual_mileage'].mean()
            }
            
        return self.village_stats_cache[village_id]

class DividendCalculator:
    """
    Calculates the P2P cashback (dividend) at the end of a period.
    """
    
    def calculate_period_performance(self, pool_value: float, claims_paid: float, platform_fee_percent: float = 0.20) -> Dict:
        """
        Determine if the village gets money back.
        
        :param pool_value: Total premiums collected in the pot.
        :param claims_paid: Total money paid out for accidents.
        :param platform_fee_percent: Cut taken by InsurePrice (default 20%).
        """
        platform_fee = pool_value * platform_fee_percent
        insurance_buffer = pool_value * 0.10 # 10% kept for catastrophic reserve
        
        available_pot = pool_value - platform_fee - insurance_buffer
        
        surplus = available_pot - claims_paid
        
        dividend_per_member = 0
        if surplus > 0:
            # We don't know exact member count here but assuming caller knows.
            # Returning total surplus
            pass
            
        return {
            'revenue': pool_value,
            'expenses': {
                'claims': claims_paid,
                'platform_fee': platform_fee,
                'reserve_buffer': insurance_buffer
            },
            'surplus_deficit': surplus,
            'is_profitable': surplus > 0
        }
