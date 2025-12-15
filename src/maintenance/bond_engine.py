# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from typing import Dict, Optional

class MaintenanceBondEngine:
    """
    Analyzes IoT vehicle health data to issue 'Preventative Maintenance Bonds'.
    If risk of mechanical failure > threshold, pay for repair immediately.
    """
    
    def __init__(self):
        # Load Maintenance Data
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, '..', '..', 'data', 'Vehicle_Maintenance_Records.csv')
            self.maintenance_data = pd.read_csv(data_path)
            # Create quick lookup
            self.maintenance_data.set_index('user_id', inplace=True)
        except Exception as e:
            print(f"Warning: Could not load maintenance data: {e}")
            self.maintenance_data = None

    def get_vehicle_health(self, user_id: str) -> Dict:
        """Get live telemetry for a user."""
        if self.maintenance_data is None or user_id not in self.maintenance_data.index:
            # Fallback mock for demo if ID not found
            return {
                'brake_wear_pct': 45.0, # Safe
                'tyre_tread_mm': 6.5,   # Good
                'battery_soh_pct': 92.0,
                'last_service_date': '2024-06-15'
            }
        
        row = self.maintenance_data.loc[user_id]
        return row.to_dict()

    def analyze_preventative_risk(self, health_metrics: Dict) -> Dict:
        """
        Determine if immediate intervention is cheaper than accident risk.
        Returns a 'Bond Offer' if risk is critical.
        """
        alerts = []
        risk_score = 0.0 # 0.0 to 1.0 (1.0 = Imminent Failure)
        
        # 1. Tyre Analysis
        tread = health_metrics['tyre_tread_mm']
        if tread < 1.6:
            alerts.append({'component': 'Tyres', 'status': 'CRITICAL', 'msg': 'Legal Limit Breached (<1.6mm)'})
            risk_score += 0.8
        elif tread < 2.5:
            alerts.append({'component': 'Tyres', 'status': 'WARNING', 'msg': 'Safety Compromised (<2.5mm)'})
            risk_score += 0.3

        # 2. Brake Analysis
        wear = health_metrics['brake_wear_pct']
        if wear > 85:
            alerts.append({'component': 'Brakes', 'status': 'CRITICAL', 'msg': 'Pads Worn (>85%)'})
            risk_score += 0.7
        elif wear > 70:
            alerts.append({'component': 'Brakes', 'status': 'WARNING', 'msg': 'Replacement Due Soon'})
            risk_score += 0.2
            
        # 3. Battery
        batt = health_metrics['battery_soh_pct']
        if batt < 40:
             alerts.append({'component': 'Battery', 'status': 'WARNING', 'msg': 'Low Health - May fail in winter'})
             
        # Calculate Bond (Payout) Offer
        # Logic: If Risk > 0.6, we offer 100% repair. If Risk > 0.3, we offer 50%.
        bond_offer_value = 0.0
        
        if risk_score >= 0.6:
            bond_offer_value = 250.0 # Full set of tyres or brakes
        elif risk_score >= 0.3:
            bond_offer_value = 100.0 # Contribution
            
        return {
            'risk_score': min(1.0, risk_score),
            'alerts': alerts,
            'bond_available': bond_offer_value > 0,
            'bond_value': bond_offer_value
        }
