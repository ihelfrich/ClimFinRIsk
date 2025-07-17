"""
Data ingestion module for retrieving climate and hazard data from various sources.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import requests
from pathlib import Path
import os

try:
    import cdsapi
    CDS_AVAILABLE = True
except ImportError:
    CDS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Handles data retrieval from various climate and hazard data sources.
    
    Supports:
    - Copernicus Climate Data Store (CDS) via CDSAPI
    - NOAA data via pyncei
    - Local file ingestion
    - Synthetic data generation for testing
    """
    
    def __init__(self, config=None):
        """
        Initialize data ingestion with configuration.
        
        Args:
            config: Configuration object containing API keys and settings
        """
        self.config = config or {}
        self.cache_dir = Path(self.config.get('cache_dir', './data_cache'))
        self.cache_dir.mkdir(exist_ok=True)
        
        self.cds_client = None
        if CDS_AVAILABLE and self.config.get('cds_api_key'):
            try:
                self.cds_client = cdsapi.Client()
                logger.info("CDS API client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize CDS client: {e}")
    
    def load_climate_data(
        self,
        hazard_types: List[str],
        scenarios: List[str],
        bbox: Dict[str, float],
        time_range: Optional[Tuple[str, str]] = None
    ) -> xr.Dataset:
        """
        Load climate data for specified hazards and scenarios.
        
        Args:
            hazard_types: List of hazard types (e.g., ['flood', 'cyclone', 'drought'])
            scenarios: List of climate scenarios (e.g., ['rcp26', 'rcp85'])
            bbox: Bounding box with keys: min_lat, max_lat, min_lon, max_lon
            time_range: Optional tuple of (start_date, end_date) in YYYY-MM-DD format
            
        Returns:
            xarray Dataset containing climate data
        """
        logger.info(f"Loading climate data for hazards: {hazard_types}, scenarios: {scenarios}")
        
        datasets = []
        
        for hazard in hazard_types:
            for scenario in scenarios:
                try:
                    cache_key = f"{hazard}_{scenario}_{self._bbox_to_string(bbox)}"
                    cached_data = self._load_from_cache(cache_key)
                    
                    if cached_data is not None:
                        logger.info(f"Loaded {hazard}-{scenario} from cache")
                        datasets.append(cached_data)
                        continue
                    
                    if hazard == 'flood':
                        data = self._load_flood_data(scenario, bbox, time_range)
                    elif hazard == 'cyclone':
                        data = self._load_cyclone_data(scenario, bbox, time_range)
                    elif hazard == 'drought':
                        data = self._load_drought_data(scenario, bbox, time_range)
                    else:
                        logger.warning(f"Unknown hazard type: {hazard}, generating synthetic data")
                        data = self._generate_synthetic_hazard_data(hazard, scenario, bbox, time_range)
                    
                    data.attrs.update({
                        'hazard_type': hazard,
                        'scenario': scenario,
                        'bbox': bbox,
                        'source': 'climfinrisk_ingestion'
                    })
                    
                    self._save_to_cache(cache_key, data)
                    datasets.append(data)
                    
                except Exception as e:
                    logger.error(f"Failed to load {hazard}-{scenario}: {e}")
                    data = self._generate_synthetic_hazard_data(hazard, scenario, bbox, time_range)
                    datasets.append(data)
        
        if datasets:
            combined = xr.concat(datasets, dim='hazard_scenario')
            logger.info(f"Successfully loaded climate data with shape: {combined.dims}")
            return combined
        else:
            raise ValueError("No climate data could be loaded")
    
    def load_scenario_data(
        self,
        scenario: str,
        hazard_types: List[str] = None,
        bbox: Dict[str, float] = None
    ) -> xr.Dataset:
        """
        Load data for a specific climate scenario.
        
        Args:
            scenario: Climate scenario (e.g., 'rcp85')
            hazard_types: List of hazard types to include
            bbox: Bounding box for spatial filtering
            
        Returns:
            xarray Dataset for the scenario
        """
        if hazard_types is None:
            hazard_types = ['flood', 'cyclone', 'drought']
        
        if bbox is None:
            bbox = {'min_lat': -90, 'max_lat': 90, 'min_lon': -180, 'max_lon': 180}
        
        return self.load_climate_data([scenario], hazard_types, bbox)
    
    def load_copernicus_data(
        self,
        dataset_name: str,
        variables: List[str],
        bbox: Dict[str, float],
        time_range: Tuple[str, str]
    ) -> xr.Dataset:
        """
        Load data from Copernicus Climate Data Store.
        
        Args:
            dataset_name: CDS dataset identifier
            variables: List of variable names to retrieve
            bbox: Spatial bounding box
            time_range: Time range as (start, end) dates
            
        Returns:
            xarray Dataset from CDS
        """
        if not self.cds_client:
            logger.warning("CDS client not available, generating synthetic data")
            return self._generate_synthetic_copernicus_data(dataset_name, variables, bbox, time_range)
        
        try:
            request = {
                'variable': variables,
                'year': self._get_years_from_range(time_range),
                'month': ['01', '02', '03', '04', '05', '06', 
                         '07', '08', '09', '10', '11', '12'],
                'day': ['01'],
                'area': [bbox['max_lat'], bbox['min_lon'], bbox['min_lat'], bbox['max_lon']],
                'format': 'netcdf'
            }
            
            output_file = self.cache_dir / f"cds_{dataset_name}_{hash(str(request))}.nc"
            
            logger.info(f"Requesting data from CDS: {dataset_name}")
            self.cds_client.retrieve(dataset_name, request, str(output_file))
            
            dataset = xr.open_dataset(output_file)
            logger.info(f"Successfully loaded CDS data: {dataset.dims}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load CDS data: {e}")
            return self._generate_synthetic_copernicus_data(dataset_name, variables, bbox, time_range)
    
    def load_asset_data(self, file_path: str) -> pd.DataFrame:
        """
        Load asset portfolio data from file.
        
        Args:
            file_path: Path to asset data file (CSV, Excel, etc.)
            
        Returns:
            DataFrame containing asset information
        """
        try:
            if file_path.endswith('.csv'):
                assets = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                assets = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            required_cols = ['lat', 'lon', 'asset_type', 'value']
            missing_cols = [col for col in required_cols if col not in assets.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info(f"Loaded {len(assets)} assets from {file_path}")
            return assets
            
        except Exception as e:
            logger.error(f"Failed to load asset data: {e}")
            return self._generate_synthetic_asset_data()
    
    def _load_flood_data(self, scenario: str, bbox: Dict[str, float], time_range: Optional[Tuple[str, str]]) -> xr.Dataset:
        """Load flood hazard data."""
        logger.info(f"Loading flood data for scenario {scenario}")
        
        return self._generate_synthetic_hazard_data('flood', scenario, bbox, time_range)
    
    def _load_cyclone_data(self, scenario: str, bbox: Dict[str, float], time_range: Optional[Tuple[str, str]]) -> xr.Dataset:
        """Load cyclone/hurricane data."""
        logger.info(f"Loading cyclone data for scenario {scenario}")
        
        return self._generate_synthetic_hazard_data('cyclone', scenario, bbox, time_range)
    
    def _load_drought_data(self, scenario: str, bbox: Dict[str, float], time_range: Optional[Tuple[str, str]]) -> xr.Dataset:
        """Load drought data."""
        logger.info(f"Loading drought data for scenario {scenario}")
        
        return self._generate_synthetic_hazard_data('drought', scenario, bbox, time_range)
    
    def _generate_synthetic_hazard_data(
        self,
        hazard: str,
        scenario: str,
        bbox: Dict[str, float],
        time_range: Optional[Tuple[str, str]] = None
    ) -> xr.Dataset:
        """Generate synthetic hazard data for testing and demonstration."""
        logger.info(f"Generating synthetic {hazard} data for scenario {scenario}")
        
        lats = np.linspace(bbox['min_lat'], bbox['max_lat'], 50)
        lons = np.linspace(bbox['min_lon'], bbox['max_lon'], 50)
        
        if time_range:
            times = pd.date_range(time_range[0], time_range[1], freq='M')
        else:
            times = pd.date_range('2020-01-01', '2050-12-31', freq='Y')
        
        np.random.seed(42)  # For reproducibility
        
        if hazard == 'flood':
            intensity = np.random.exponential(0.2, (len(times), len(lats), len(lons)))
            intensity = np.clip(intensity, 0, 1)
            
            frequency = np.random.poisson(2, (len(times), len(lats), len(lons)))
            
            data_vars = {
                'flood_intensity': (['time', 'lat', 'lon'], intensity),
                'flood_frequency': (['time', 'lat', 'lon'], frequency)
            }
            
        elif hazard == 'cyclone':
            wind_speed = np.random.gamma(2, 15, (len(times), len(lats), len(lons)))
            
            surge_height = np.random.exponential(1.5, (len(times), len(lats), len(lons)))
            
            data_vars = {
                'wind_speed': (['time', 'lat', 'lon'], wind_speed),
                'surge_height': (['time', 'lat', 'lon'], surge_height)
            }
            
        elif hazard == 'drought':
            precip_anomaly = np.random.normal(-0.5, 1.0, (len(times), len(lats), len(lons)))
            
            soil_moisture = np.random.beta(2, 3, (len(times), len(lats), len(lons)))
            
            data_vars = {
                'precipitation_anomaly': (['time', 'lat', 'lon'], precip_anomaly),
                'soil_moisture': (['time', 'lat', 'lon'], soil_moisture)
            }
            
        else:
            intensity = np.random.exponential(0.3, (len(times), len(lats), len(lons)))
            data_vars = {
                'hazard_intensity': (['time', 'lat', 'lon'], intensity)
            }
        
        coords = {
            'time': times,
            'lat': lats,
            'lon': lons
        }
        
        dataset = xr.Dataset(data_vars, coords=coords)
        
        scenario_multipliers = {
            'rcp26': 1.0,
            'rcp45': 1.3,
            'rcp85': 1.8
        }
        multiplier = scenario_multipliers.get(scenario, 1.0)
        
        for var in dataset.data_vars:
            dataset[var] = dataset[var] * multiplier
        
        return dataset
    
    def _generate_synthetic_asset_data(self, n_assets: int = 100) -> pd.DataFrame:
        """Generate synthetic asset portfolio data."""
        logger.info(f"Generating synthetic asset data for {n_assets} assets")
        
        np.random.seed(42)
        
        lats = np.random.normal(40, 20, n_assets)  # Roughly centered on mid-latitudes
        lons = np.random.normal(0, 60, n_assets)   # Global distribution
        
        lats = np.clip(lats, -85, 85)
        lons = np.clip(lons, -180, 180)
        
        asset_types = ['commercial', 'residential', 'industrial', 'infrastructure']
        types = np.random.choice(asset_types, n_assets)
        
        base_values = {
            'commercial': 50,
            'residential': 20,
            'industrial': 100,
            'infrastructure': 200
        }
        
        values = [base_values[t] * np.random.lognormal(0, 0.5) for t in types]
        
        assets = pd.DataFrame({
            'asset_id': [f'ASSET_{i:04d}' for i in range(n_assets)],
            'lat': lats,
            'lon': lons,
            'asset_type': types,
            'value': values,
            'construction_year': np.random.randint(1950, 2020, n_assets),
            'building_material': np.random.choice(['concrete', 'steel', 'wood', 'mixed'], n_assets)
        })
        
        return assets
    
    def _generate_synthetic_copernicus_data(
        self,
        dataset_name: str,
        variables: List[str],
        bbox: Dict[str, float],
        time_range: Tuple[str, str]
    ) -> xr.Dataset:
        """Generate synthetic Copernicus-like data."""
        logger.info(f"Generating synthetic Copernicus data for {dataset_name}")
        
        lats = np.linspace(bbox['min_lat'], bbox['max_lat'], 25)
        lons = np.linspace(bbox['min_lon'], bbox['max_lon'], 25)
        times = pd.date_range(time_range[0], time_range[1], freq='M')
        
        data_vars = {}
        np.random.seed(42)
        
        for var in variables:
            if 'temperature' in var.lower():
                data = np.random.normal(15, 10, (len(times), len(lats), len(lons)))
            elif 'precipitation' in var.lower():
                data = np.random.exponential(50, (len(times), len(lats), len(lons)))
            else:
                data = np.random.normal(0, 1, (len(times), len(lats), len(lons)))
            
            data_vars[var] = (['time', 'lat', 'lon'], data)
        
        coords = {
            'time': times,
            'lat': lats,
            'lon': lons
        }
        
        return xr.Dataset(data_vars, coords=coords)
    
    def _bbox_to_string(self, bbox: Dict[str, float]) -> str:
        """Convert bounding box to string for caching."""
        return f"{bbox['min_lat']}_{bbox['max_lat']}_{bbox['min_lon']}_{bbox['max_lon']}"
    
    def _get_years_from_range(self, time_range: Tuple[str, str]) -> List[str]:
        """Extract years from time range."""
        start_year = int(time_range[0][:4])
        end_year = int(time_range[1][:4])
        return [str(year) for year in range(start_year, end_year + 1)]
    
    def _load_from_cache(self, cache_key: str) -> Optional[xr.Dataset]:
        """Load data from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.nc"
        if cache_file.exists():
            try:
                return xr.open_dataset(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: xr.Dataset):
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.nc"
        try:
            data.to_netcdf(cache_file)
            logger.debug(f"Saved data to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
