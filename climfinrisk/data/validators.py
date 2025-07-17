"""
Data validation utilities specific to climate risk data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


class ClimateDataValidator:
    """
    Specialized validator for climate risk data quality and consistency.
    """
    
    @staticmethod
    def validate_climate_dataset_structure(dataset: xr.Dataset) -> Tuple[bool, List[str]]:
        """
        Validate the structure and format of climate datasets.
        
        Args:
            dataset: Climate xarray Dataset
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        required_dims = ['time']
        for dim in required_dims:
            if dim not in dataset.dims:
                issues.append(f"Missing required dimension: {dim}")
        
        spatial_dims = [dim for dim in dataset.dims 
                       if dim in ['lat', 'lon', 'latitude', 'longitude', 'x', 'y']]
        if len(spatial_dims) < 2:
            issues.append("Insufficient spatial dimensions (need lat/lon or x/y)")
        
        for coord_name, coord in dataset.coords.items():
            if coord_name == 'time':
                if not pd.api.types.is_datetime64_any_dtype(coord):
                    issues.append("Time coordinate must be datetime type")
            elif coord_name in ['lat', 'latitude']:
                if not (-90 <= coord.min() <= coord.max() <= 90):
                    issues.append("Latitude values outside valid range [-90, 90]")
            elif coord_name in ['lon', 'longitude']:
                if not (-180 <= coord.min() <= coord.max() <= 360):
                    issues.append("Longitude values outside valid range [-180, 360]")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def validate_climate_variable_quality(
        dataset: xr.Dataset,
        variable_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate quality of specific climate variables.
        
        Args:
            dataset: Climate dataset
            variable_name: Name of variable to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if variable_name not in dataset.data_vars:
            issues.append(f"Variable {variable_name} not found in dataset")
            return False, issues
        
        var_data = dataset[variable_name]
        
        if var_data.isnull().all():
            issues.append(f"Variable {variable_name} contains only NaN values")
        
        if np.isinf(var_data.values).any():
            issues.append(f"Variable {variable_name} contains infinite values")
        
        if variable_name.lower() in ['temperature', 'temp', 'tas']:
            if (var_data < -100).any() or (var_data > 100).any():
                issues.append(f"Temperature values outside reasonable range [-100, 100]°C")
        
        elif variable_name.lower() in ['precipitation', 'precip', 'pr']:
            if (var_data < 0).any():
                issues.append(f"Negative precipitation values found")
        
        elif variable_name.lower() in ['wind_speed', 'wind', 'ws']:
            if (var_data < 0).any():
                issues.append(f"Negative wind speed values found")
            if (var_data > 200).any():  # 200 m/s is extreme
                issues.append(f"Extremely high wind speed values (>200 m/s)")
        
        elif variable_name.lower() in ['pressure', 'slp', 'mslp']:
            if (var_data < 800).any() or (var_data > 1100).any():
                issues.append(f"Pressure values outside reasonable range [800, 1100] hPa")
        
        missing_pct = (var_data.isnull().sum() / var_data.size) * 100
        if missing_pct > 50:
            issues.append(f"Variable {variable_name} has {missing_pct:.1f}% missing data")
        elif missing_pct > 20:
            issues.append(f"Variable {variable_name} has {missing_pct:.1f}% missing data (warning)")
        
        is_valid = len(issues) == 0
        return is_valid, issues
