#!/usr/bin/env python3
"""
Integration test for the ClimFinRisk platform.

Tests that all core modules can be imported and work together.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all core modules can be imported."""
    print("🔍 Testing module imports...")
    
    try:
        from climfinrisk import ClimateRiskModeler
        print("✅ ClimateRiskModeler imported")
        
        from climfinrisk.data import DataIngestion, DataPreprocessor
        from climfinrisk.data.validators import ClimateDataValidator
        print("✅ Data modules imported")
        
        from climfinrisk.modeling import DimensionalityReduction, RiskEstimation, VulnerabilityCurves
        print("✅ Modeling modules imported")
        
        from climfinrisk.geospatial import SpatialAnalyzer
        print("✅ Geospatial modules imported")
        
        from climfinrisk.utils import ConfigManager, DataValidator, Logger
        print("✅ Utility modules imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from climfinrisk.utils import ConfigManager
        config = ConfigManager()
        assert config.get('data.cache_dir') is not None
        print("✅ Configuration management works")
        
        from climfinrisk.data import DataIngestion
        data_ingestion = DataIngestion()
        
        assets = data_ingestion._generate_synthetic_asset_data(n_assets=10)
        assert len(assets) == 10
        assert 'lat' in assets.columns
        assert 'lon' in assets.columns
        print("✅ Synthetic asset data generation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🧪 ClimFinRisk Platform Integration Tests")
    print("=" * 50)
    
    import_success = test_imports()
    functionality_success = test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    total_tests = 2
    passed_tests = sum([import_success, functionality_success])
    
    print(f"✅ Imports: {'PASS' if import_success else 'FAIL'}")
    print(f"✅ Basic Functionality: {'PASS' if functionality_success else 'FAIL'}")
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🌍 ClimFinRisk platform is ready for groundbreaking climate risk analysis!")
        return True
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
