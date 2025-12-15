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

    def assign_user_to_village(self, user_profile: Dict) -> str:
        """
        Assigns a user to a village based on their profile.
        (Mock logic for prototype)
        """
        age = user_profile.get('age', 30)
        miles = user_profile.get('annual_mileage', 10000)
        
        if age < 25:
            return 'young_pros'
        elif miles < 5000:
            return 'weekend_warriors'
        elif user_profile.get('night_driving_percent', 0) > 0.3:
            return 'night_owls'
        else:
            return 'safe_commuters'

    def get_village_stats(self, village_id: str) -> Dict:
        """Generated mock stats for a village."""
        details = self.VILLAGE_TYPES.get(village_id, self.VILLAGE_TYPES['safe_commuters'])
        
        # Simulate varying pool sizes
        member_count = np.random.randint(50, 500)
        avg_premium = np.random.randint(400, 800)
        
        return {
            'id': village_id,
            'name': details['name'],
            'members': member_count,
            'total_pool_value': member_count * avg_premium,
            'risk_factor': details['risk_factor']
        }

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
