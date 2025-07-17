"""
Advanced data preprocessing module for climate risk modeling.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import warnings

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Advanced data preprocessing for climate risk modeling.
    
    Features:
    - Multi-dimensional data alignment and interpolation
    - Advanced missing data imputation
    - Temporal and spatial standardization
    - Quality control and outlier detection
    """
    
    def __init__(self, config=None):
        """
        Initialize data preprocessor.
        
        Args:
            config: Configuration object with preprocessing parameters
        """
        self.config = config or {}
        self.scalers = {}
        self.imputers = {}
        
    def preprocess_climate_data(
        self,
        climate_data: xr.Dataset,
        target_resolution: Optional[Dict[str, float]] = None,
        temporal_aggregation: str = 'monthly'
    ) -> xr.Dataset:
        """
        Comprehensive preprocessing of climate data.
        
        Args:
            climate_data: Raw climate dataset
            target_resolution: Target spatial resolution (lat, lon in degrees)
            temporal_aggregation: Temporal aggregation method
            
        Returns:
            Preprocessed climate dataset
        """
        logger.info("Starting comprehensive climate data preprocessing")
        
        processed_data = climate_data.copy()
        
        processed_data = self._quality_control(processed_data)
        
        if target_resolution:
            processed_data = self._regrid_data(processed_data, target_resolution)
        
        processed_data = self._temporal_aggregation(processed_data, temporal_aggregation)
        
        processed_data = self._impute_missing_data(processed_data)
        
        processed_data = self._standardize_variables(processed_data)
        
        processed_data = self._add_derived_variables(processed_data)
        
        logger.info("Climate data preprocessing completed")
        return processed_data
    
    def align_asset_climate_data(
        self,
        climate_data: xr.Dataset,
        assets: pd.DataFrame
    ) -> xr.Dataset:
        """
        Align climate data with asset locations.
        
        Args:
            climate_data: Climate dataset
            assets: Asset portfolio data
            
        Returns:
            Climate data aligned to asset locations
        """
        logger.info("Aligning climate data with asset locations")
        
        asset_lats = assets['lat'].values
        asset_lons = assets['lon'].values
        asset_ids = assets.index.values
        
        try:
            aligned_data = climate_data.interp(
                lat=xr.DataArray(asset_lats, dims='asset_id'),
                lon=xr.DataArray(asset_lons, dims='asset_id'),
                method='linear'
            )
            
            aligned_data = aligned_data.assign_coords(asset_id=asset_ids)
            
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}. Using nearest neighbor.")
            aligned_data = climate_data.sel(
                lat=xr.DataArray(asset_lats, dims='asset_id'),
                lon=xr.DataArray(asset_lons, dims='asset_id'),
                method='nearest'
            )
            aligned_data = aligned_data.assign_coords(asset_id=asset_ids)
        
        return aligned_data
    
    def create_feature_matrix(
        self,
        climate_data: xr.Dataset,
        assets: pd.DataFrame,
        include_lags: bool = True,
        lag_periods: List[int] = [1, 3, 6, 12]
    ) -> pd.DataFrame:
        """
        Create feature matrix for machine learning models.
        
        Args:
            climate_data: Preprocessed climate data
            assets: Asset portfolio data
            include_lags: Whether to include lagged features
            lag_periods: Number of periods to lag
            
        Returns:
            Feature matrix DataFrame
        """
        logger.info("Creating feature matrix for ML models")
        
        aligned_data = self.align_asset_climate_data(climate_data, assets)
        
        feature_df = aligned_data.to_dataframe().reset_index()
        
        asset_features = assets.copy()
        feature_df = feature_df.merge(
            asset_features, 
            left_on='asset_id', 
            right_index=True, 
            how='left'
        )
        
        if 'time' in feature_df.columns:
            feature_df = self._add_temporal_features(feature_df)
        
        if include_lags and 'time' in feature_df.columns:
            feature_df = self._add_lagged_features(feature_df, lag_periods)
        
        feature_df = self._add_interaction_features(feature_df)
        
        return feature_df
    
    def _quality_control(self, data: xr.Dataset) -> xr.Dataset:
        """Apply quality control checks and remove outliers."""
        logger.info("Applying quality control checks")
        
        processed_data = data.copy()
        
        for var_name, var_data in data.data_vars.items():
            mean_val = var_data.mean()
            std_val = var_data.std()
            
            lower_bound = mean_val - 5 * std_val
            upper_bound = mean_val + 5 * std_val
            
            outlier_mask = (var_data < lower_bound) | (var_data > upper_bound)
            processed_data[var_name] = var_data.where(~outlier_mask)
            
            n_outliers = outlier_mask.sum().values
            total_points = var_data.size
            outlier_pct = (n_outliers / total_points) * 100
            
            if outlier_pct > 0.1:  # More than 0.1% outliers
                logger.warning(f"Variable {var_name}: {outlier_pct:.2f}% outliers detected and masked")
        
        return processed_data
    
    def _regrid_data(
        self,
        data: xr.Dataset,
        target_resolution: Dict[str, float]
    ) -> xr.Dataset:
        """Regrid data to target spatial resolution."""
        logger.info(f"Regridding data to resolution: {target_resolution}")
        
        try:
            if 'lat' in data.coords and 'lon' in data.coords:
                current_lat = data.coords['lat']
                current_lon = data.coords['lon']
                
                lat_min, lat_max = float(current_lat.min()), float(current_lat.max())
                lon_min, lon_max = float(current_lon.min()), float(current_lon.max())
                
                new_lat = np.arange(lat_min, lat_max + target_resolution['lat'], target_resolution['lat'])
                new_lon = np.arange(lon_min, lon_max + target_resolution['lon'], target_resolution['lon'])
                
                regridded_data = data.interp(lat=new_lat, lon=new_lon, method='linear')
                
                return regridded_data
            else:
                logger.warning("No lat/lon coordinates found for regridding")
                return data
                
        except Exception as e:
            logger.error(f"Regridding failed: {e}")
            return data
    
    def _temporal_aggregation(
        self,
        data: xr.Dataset,
        aggregation: str
    ) -> xr.Dataset:
        """Aggregate data temporally."""
        if 'time' not in data.dims:
            return data
        
        logger.info(f"Applying temporal aggregation: {aggregation}")
        
        try:
            if aggregation == 'monthly':
                aggregated = data.resample(time='1M').mean()
            elif aggregation == 'seasonal':
                aggregated = data.resample(time='QS').mean()
            elif aggregation == 'annual':
                aggregated = data.resample(time='1Y').mean()
            elif aggregation == 'daily':
                aggregated = data.resample(time='1D').mean()
            else:
                logger.warning(f"Unknown aggregation method: {aggregation}")
                aggregated = data
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Temporal aggregation failed: {e}")
            return data
    
    def _impute_missing_data(self, data: xr.Dataset) -> xr.Dataset:
        """Impute missing data using advanced methods."""
        logger.info("Imputing missing data")
        
        processed_data = data.copy()
        
        for var_name, var_data in data.data_vars.items():
            missing_pct = (var_data.isnull().sum() / var_data.size) * 100
            
            if missing_pct > 0:
                logger.info(f"Variable {var_name}: {missing_pct:.2f}% missing data")
                
                if missing_pct < 20:  # Use interpolation for small gaps
                    if 'time' in var_data.dims:
                        interpolated = var_data.interpolate_na(dim='time', method='linear')
                    else:
                        interpolated = var_data
                    
                    if 'lat' in var_data.dims and 'lon' in var_data.dims:
                        interpolated = interpolated.interpolate_na(dim='lat', method='linear')
                        interpolated = interpolated.interpolate_na(dim='lon', method='linear')
                    
                    processed_data[var_name] = interpolated
                    
                elif missing_pct < 50:  # Use mean imputation for moderate gaps
                    mean_value = var_data.mean()
                    processed_data[var_name] = var_data.fillna(mean_value)
                    
                else:  # Too much missing data
                    logger.warning(f"Variable {var_name} has {missing_pct:.2f}% missing data - consider removing")
        
        return processed_data
    
    def _standardize_variables(self, data: xr.Dataset) -> xr.Dataset:
        """Standardize variables for consistent scaling."""
        logger.info("Standardizing variables")
        
        processed_data = data.copy()
        
        for var_name, var_data in data.data_vars.items():
            if var_data.dtype == bool or var_data.max() <= 1:
                continue
            
            mean_val = var_data.mean()
            std_val = var_data.std()
            
            if std_val > 0:
                standardized = (var_data - mean_val) / std_val
                processed_data[var_name] = standardized
                
                self.scalers[var_name] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'method': 'zscore'
                }
        
        return processed_data
    
    def _add_derived_variables(self, data: xr.Dataset) -> xr.Dataset:
        """Add derived climate variables."""
        logger.info("Adding derived climate variables")
        
        processed_data = data.copy()
        
        if 'temperature' in data.data_vars:
            temp = data['temperature']
            
            processed_data['heating_degree_days'] = np.maximum(0, 18.3 - temp)
            processed_data['cooling_degree_days'] = np.maximum(0, temp - 18.3)
            
            if 'time' in temp.dims:
                processed_data['temp_max'] = temp.resample(time='1M').max()
                processed_data['temp_min'] = temp.resample(time='1M').min()
                processed_data['temp_range'] = processed_data['temp_max'] - processed_data['temp_min']
        
        if 'precipitation' in data.data_vars:
            precip = data['precipitation']
            
            processed_data['precip_intensity'] = precip.where(precip > 0).mean()
            
            if 'time' in precip.dims:
                dry_days = (precip < 1.0).astype(int)
                processed_data['dry_spell_indicator'] = dry_days
        
        if 'wind_speed' in data.data_vars:
            wind = data['wind_speed']
            
            processed_data['wind_power'] = wind ** 3
            
            processed_data['extreme_wind'] = (wind > wind.quantile(0.95)).astype(int)
        
        return processed_data
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to DataFrame."""
        if 'time' not in df.columns:
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        df['quarter'] = df['time'].dt.quarter
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        return df
    
    def _add_lagged_features(
        self,
        df: pd.DataFrame,
        lag_periods: List[int]
    ) -> pd.DataFrame:
        """Add lagged features for time series analysis."""
        if 'time' not in df.columns or 'asset_id' not in df.columns:
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        climate_cols = [col for col in numeric_cols 
                       if col not in ['asset_id', 'year', 'month', 'day_of_year', 'quarter']]
        
        df_sorted = df.sort_values(['asset_id', 'time'])
        
        for col in climate_cols:
            for lag in lag_periods:
                lag_col = f'{col}_lag_{lag}'
                df_sorted[lag_col] = df_sorted.groupby('asset_id')[col].shift(lag)
        
        return df_sorted
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between climate variables."""
        interaction_pairs = [
            ('temperature', 'precipitation'),
            ('temperature', 'wind_speed'),
            ('precipitation', 'wind_speed')
        ]
        
        for var1, var2 in interaction_pairs:
            if var1 in df.columns and var2 in df.columns:
                df[f'{var1}_x_{var2}'] = df[var1] * df[var2]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[f'{var1}_div_{var2}'] = df[var1] / (df[var2] + 1e-8)
        
        return df
    
    def inverse_transform(
        self,
        data: xr.Dataset,
        variable_name: str
    ) -> xr.Dataset:
        """Inverse transform standardized data back to original scale."""
        if variable_name not in self.scalers:
            logger.warning(f"No scaler found for variable {variable_name}")
            return data
        
        scaler_info = self.scalers[variable_name]
        
        if scaler_info['method'] == 'zscore':
            original_data = data[variable_name] * scaler_info['std'] + scaler_info['mean']
            data[variable_name] = original_data
        
        return data
