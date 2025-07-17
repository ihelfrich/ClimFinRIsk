#!/usr/bin/env python3
"""
Debug script to isolate and test risk calculation logic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from climfinrisk.data import DataIngestion
from climfinrisk.modeling import RiskEstimation

def debug_synthetic_data():
    """Debug synthetic hazard data generation."""
    print("🔍 Debugging Synthetic Data Generation")
    print("=" * 50)
    
    data_ingestion = DataIngestion()
    
    bbox = {
        'min_lat': 25.0, 'max_lat': 30.0,
        'min_lon': -95.0, 'max_lon': -90.0
    }
    
    hazard_data = data_ingestion._generate_synthetic_hazard_data(
        'flood', 'rcp45', bbox
    )
    
    flood_intensity = hazard_data['flood_intensity'].values
    
    print(f"📊 Flood Intensity Statistics:")
    print(f"   Min: {flood_intensity.min():.3f}")
    print(f"   Max: {flood_intensity.max():.3f}")
    print(f"   Mean: {flood_intensity.mean():.3f}")
    print(f"   Std: {flood_intensity.std():.3f}")
    print(f"   Values > 0.1: {(flood_intensity > 0.1).sum()}/{flood_intensity.size}")
    print(f"   Values > 0.2: {(flood_intensity > 0.2).sum()}/{flood_intensity.size}")
    print(f"   Values > 0.5: {(flood_intensity > 0.5).sum()}/{flood_intensity.size}")
    
    return hazard_data

def debug_vulnerability_curves():
    """Debug vulnerability curve calculations."""
    print("\n🔍 Debugging Vulnerability Curves")
    print("=" * 50)
    
    risk_estimator = RiskEstimation()
    
    test_intensities = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0])
    
    for asset_type in ['residential', 'commercial', 'industrial', 'infrastructure']:
        print(f"\n📈 {asset_type.upper()} Vulnerability:")
        
        vulnerability = risk_estimator._get_vulnerability_curve(asset_type)
        
        for intensity in test_intensities:
            if intensity <= vulnerability['threshold']:
                damage_ratio = 0.0
            else:
                damage_ratio = min(
                    vulnerability['max_damage'],
                    vulnerability['slope'] * (intensity - vulnerability['threshold'])
                )
            print(f"   Intensity {intensity:.2f} → Damage {damage_ratio:.3f}")

def debug_risk_calculation():
    """Debug complete risk calculation pipeline."""
    print("\n🔍 Debugging Complete Risk Calculation")
    print("=" * 50)
    
    data_ingestion = DataIngestion()
    risk_estimator = RiskEstimation()
    
    bbox = {
        'min_lat': 25.0, 'max_lat': 30.0,
        'min_lon': -95.0, 'max_lon': -90.0
    }
    
    climate_data = data_ingestion.load_climate_data(
        hazard_types=['flood'],
        scenarios=['rcp45'],
        bbox=bbox
    )
    
    assets = data_ingestion._generate_synthetic_asset_data(n_assets=10)
    
    print(f"📊 Asset Values:")
    print(f"   Total Portfolio Value: ${assets['value'].sum():,.0f}")
    print(f"   Asset Types: {assets['type'].value_counts().to_dict()}")
    
    risk_estimates = risk_estimator.estimate_portfolio_risk(
        climate_data,
        assets,
        scenarios=['rcp45']
    )
    
    print(f"\n💰 Risk Calculation Results:")
    expected_loss_col = 'expected_loss_rcp45'
    if expected_loss_col in risk_estimates.columns:
        total_loss = risk_estimates[expected_loss_col].sum()
        assets_at_risk = (risk_estimates[expected_loss_col] > 0).sum()
        max_loss = risk_estimates[expected_loss_col].max()
        
        print(f"   Total Expected Loss: ${total_loss:,.0f}")
        print(f"   Assets at Risk: {assets_at_risk}/{len(assets)}")
        print(f"   Maximum Asset Loss: ${max_loss:,.0f}")
        
        if total_loss > 0:
            print("✅ Risk calculation is working correctly!")
        else:
            print("❌ Risk calculation still producing zero values")
            
            print("\n🔍 Detailed Analysis:")
            for idx, row in risk_estimates.iterrows():
                asset_type = assets.loc[idx, 'type']
                asset_value = assets.loc[idx, 'value']
                expected_loss = row[expected_loss_col]
                
                print(f"   Asset {idx}: {asset_type}, Value ${asset_value:,.0f}, Loss ${expected_loss:,.0f}")
    
    return risk_estimates

def main():
    """Run all debug tests."""
    print("🧪 ClimFinRisk Risk Calculation Debug")
    print("=" * 60)
    
    try:
        hazard_data = debug_synthetic_data()
        debug_vulnerability_curves()
        risk_estimates = debug_risk_calculation()
        
        print("\n" + "=" * 60)
        print("🎯 Debug Summary")
        print("=" * 60)
        
        expected_loss_col = 'expected_loss_rcp45'
        if expected_loss_col in risk_estimates.columns:
            total_loss = risk_estimates[expected_loss_col].sum()
            if total_loss > 0:
                print("✅ Risk calculation debug PASSED - producing meaningful values")
            else:
                print("❌ Risk calculation debug FAILED - still producing zero values")
        else:
            print("❌ Risk calculation debug FAILED - missing expected loss column")
            
    except Exception as e:
        print(f"❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
