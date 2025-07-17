"""
Data validation utilities for the climate risk modeling platform.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data quality and consistency for climate risk modeling.
    """
    
    @staticmethod
    def validate_asset_data(assets: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate asset portfolio data.
        
        Args:
            assets: Asset DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        required_columns = ['lat', 'lon', 'asset_type', 'value']
        missing_columns = [col for col in required_columns if col not in assets.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        if 'lat' in assets.columns:
            if not assets['lat'].dtype in [np.float64, np.float32, int]:
                issues.append("Latitude column must be numeric")
            elif not assets['lat'].between(-90, 90).all():
                issues.append("Latitude values must be between -90 and 90")
        
        if 'lon' in assets.columns:
            if not assets['lon'].dtype in [np.float64, np.float32, int]:
                issues.append("Longitude column must be numeric")
            elif not assets['lon'].between(-180, 180).all():
                issues.append("Longitude values must be between -180 and 180")
        
        if 'value' in assets.columns:
            if not pd.api.types.is_numeric_dtype(assets['value']):
                issues.append("Asset value column must be numeric")
            elif (assets['value'] < 0).any():
                issues.append("Asset values cannot be negative")
        
        critical_columns = ['lat', 'lon', 'value']
        for col in critical_columns:
            if col in assets.columns and assets[col].isna().any():
                issues.append(f"Missing values found in critical column: {col}")
        
        if 'asset_id' in assets.columns:
            if assets['asset_id'].duplicated().any():
                issues.append("Duplicate asset IDs found")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def validate_climate_data(climate_data: xr.Dataset) -> Tuple[bool, List[str]]:
        """
        Validate climate dataset.
        
        Args:
            climate_data: Climate xarray Dataset to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if len(climate_data.data_vars) == 0:
            issues.append("Climate dataset contains no data variables")
            return False, issues
        
        required_dims = ['time']
        for dim in required_dims:
            if dim not in climate_data.dims:
                issues.append(f"Missing required dimension: {dim}")
        
        spatial_dims = [dim for dim in climate_data.dims 
                       if dim in ['lat', 'lon', 'latitude', 'longitude', 'x', 'y']]
        if len(spatial_dims) < 2:
            issues.append("Insufficient spatial dimensions found")
        
        for var_name, var_data in climate_data.data_vars.items():
            if var_data.isnull().all():
                issues.append(f"Variable {var_name} contains only NaN values")
            
            if np.isinf(var_data.values).any():
                issues.append(f"Variable {var_name} contains infinite values")
            
            if var_name.lower() in ['temperature', 'temp']:
                if (var_data < -100).any() or (var_data > 100).any():
                    issues.append(f"Temperature values in {var_name} outside reasonable range")
            
            elif var_name.lower() in ['precipitation', 'precip']:
                if (var_data < 0).any():
                    issues.append(f"Negative precipitation values in {var_name}")
        
        if 'time' in climate_data.coords:
            time_coord = climate_data.coords['time']
            if not pd.api.types.is_datetime64_any_dtype(time_coord):
                issues.append("Time coordinate is not datetime type")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def validate_risk_estimates(risk_estimates: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate risk estimation results.
        
        Args:
            risk_estimates: Risk estimates DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        risk_columns = [col for col in risk_estimates.columns 
                       if 'expected_loss' in col or 'var_' in col]
        if not risk_columns:
            issues.append("No risk estimate columns found")
        
        for col in risk_columns:
            if col in risk_estimates.columns:
                if (risk_estimates[col] < 0).any():
                    issues.append(f"Negative risk values found in {col}")
                
                if np.isinf(risk_estimates[col]).any():
                    issues.append(f"Infinite risk values found in {col}")
                
                if (risk_estimates[col] > 1e12).any():
                    issues.append(f"Extremely large risk values found in {col}")
        
        var_95_cols = [col for col in risk_estimates.columns if 'var_95' in col]
        var_99_cols = [col for col in risk_estimates.columns if 'var_99' in col]
        
        for var_95_col in var_95_cols:
            scenario = var_95_col.replace('var_95_', '')
            var_99_col = f'var_99_{scenario}'
            
            if var_99_col in risk_estimates.columns:
                inconsistent = risk_estimates[var_99_col] < risk_estimates[var_95_col]
                if inconsistent.any():
                    issues.append(f"VaR_99 < VaR_95 inconsistency in scenario {scenario}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def validate_model_inputs(
        climate_data: xr.Dataset,
        assets: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate inputs for the complete modeling pipeline.
        
        Args:
            climate_data: Climate dataset
            assets: Asset portfolio data
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        all_issues = []
        
        climate_valid, climate_issues = DataValidator.validate_climate_data(climate_data)
        asset_valid, asset_issues = DataValidator.validate_asset_data(assets)
        
        all_issues.extend(climate_issues)
        all_issues.extend(asset_issues)
        
        if climate_valid and asset_valid:
            spatial_issues = DataValidator._check_spatial_alignment(climate_data, assets)
            all_issues.extend(spatial_issues)
        
        if climate_valid:
            temporal_issues = DataValidator._check_temporal_coverage(climate_data, config)
            all_issues.extend(temporal_issues)
        
        is_valid = len(all_issues) == 0
        return is_valid, all_issues
    
    @staticmethod
    def _check_spatial_alignment(
        climate_data: xr.Dataset,
        assets: pd.DataFrame
    ) -> List[str]:
        """Check if asset locations are covered by climate data."""
        issues = []
        
        try:
            if 'lat' in climate_data.coords and 'lon' in climate_data.coords:
                climate_lat_min = float(climate_data.coords['lat'].min())
                climate_lat_max = float(climate_data.coords['lat'].max())
                climate_lon_min = float(climate_data.coords['lon'].min())
                climate_lon_max = float(climate_data.coords['lon'].max())
                
                assets_outside = (
                    (assets['lat'] < climate_lat_min) |
                    (assets['lat'] > climate_lat_max) |
                    (assets['lon'] < climate_lon_min) |
                    (assets['lon'] > climate_lon_max)
                )
                
                if assets_outside.any():
                    n_outside = assets_outside.sum()
                    issues.append(f"{n_outside} assets are outside climate data spatial bounds")
        
        except Exception as e:
            issues.append(f"Could not check spatial alignment: {e}")
        
        return issues
    
    @staticmethod
    def _check_temporal_coverage(
        climate_data: xr.Dataset,
        config: Dict[str, Any]
    ) -> List[str]:
        """Check temporal coverage of climate data."""
        issues = []
        
        try:
            if 'time' in climate_data.coords:
                time_coord = climate_data.coords['time']
                time_span_years = (time_coord.max() - time_coord.min()).values / np.timedelta64(1, 'Y')
                
                required_years = config.get('modeling', {}).get('risk_estimation', {}).get('time_horizon', 30)
                
                if time_span_years < required_years:
                    issues.append(f"Climate data temporal coverage ({time_span_years:.1f} years) "
                                f"is less than required time horizon ({required_years} years)")
        
        except Exception as e:
            issues.append(f"Could not check temporal coverage: {e}")
        
        return issues
    
    @staticmethod
    def generate_validation_report(
        climate_data: xr.Dataset,
        assets: pd.DataFrame,
        risk_estimates: pd.DataFrame = None,
        config: Dict[str, Any] = None
    ) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            climate_data: Climate dataset
            assets: Asset portfolio data
            risk_estimates: Risk estimates (optional)
            config: Configuration (optional)
            
        Returns:
            Validation report as string
        """
        report = "# Data Validation Report\n\n"
        
        climate_valid, climate_issues = DataValidator.validate_climate_data(climate_data)
        report += f"## Climate Data Validation\n"
        report += f"**Status:** {'✅ PASSED' if climate_valid else '❌ FAILED'}\n\n"
        
        if climate_issues:
            report += "**Issues:**\n"
            for issue in climate_issues:
                report += f"- {issue}\n"
        else:
            report += "No issues found.\n"
        report += "\n"
        
        asset_valid, asset_issues = DataValidator.validate_asset_data(assets)
        report += f"## Asset Data Validation\n"
        report += f"**Status:** {'✅ PASSED' if asset_valid else '❌ FAILED'}\n\n"
        
        if asset_issues:
            report += "**Issues:**\n"
            for issue in asset_issues:
                report += f"- {issue}\n"
        else:
            report += "No issues found.\n"
        report += "\n"
        
        if risk_estimates is not None:
            risk_valid, risk_issues = DataValidator.validate_risk_estimates(risk_estimates)
            report += f"## Risk Estimates Validation\n"
            report += f"**Status:** {'✅ PASSED' if risk_valid else '❌ FAILED'}\n\n"
            
            if risk_issues:
                report += "**Issues:**\n"
                for issue in risk_issues:
                    report += f"- {issue}\n"
            else:
                report += "No issues found.\n"
            report += "\n"
        
        overall_valid = climate_valid and asset_valid
        if risk_estimates is not None:
            overall_valid = overall_valid and risk_valid
        
        report += f"## Overall Validation\n"
        report += f"**Status:** {'✅ ALL CHECKS PASSED' if overall_valid else '❌ VALIDATION FAILED'}\n\n"
        
        if overall_valid:
            report += "All data validation checks passed. The data is ready for climate risk modeling.\n"
        else:
            report += "Some validation checks failed. Please address the issues before proceeding.\n"
        
        return report
