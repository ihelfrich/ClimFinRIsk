"""
Real-world data source loaders for geospatial climate risk analysis.

This module provides efficient loaders for various open-source geospatial datasets
including population, nightlights, roads, and land cover data.
"""

import os
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from urllib.parse import urlencode
import time
import zipfile
import rasterio
from rasterio.warp import transform_bounds
import warnings

logger = logging.getLogger(__name__)


class BaseDataLoader:
    """Base class for all real-world data loaders."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, filename: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / filename
        
    def _is_cached(self, filename: str) -> bool:
        """Check if data is already cached."""
        return self._get_cache_path(filename).exists()


class WorldPopDataLoader(BaseDataLoader):
    """
    Loader for WorldPop population density data.
    
    WorldPop provides high-resolution population distribution datasets
    for demographic and development applications.
    """
    
    BASE_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj"
    
    def __init__(self, cache_dir: str = "data/cache/worldpop"):
        super().__init__(cache_dir)
        
    def load_population_data(
        self, 
        country_code: str, 
        year: int = 2020,
        bbox: Optional[Dict[str, float]] = None
    ) -> xr.DataArray:
        """
        Load population density data for a specific country and year.
        
        Args:
            country_code: ISO 3-letter country code (e.g., 'USA', 'GBR')
            year: Year of data (2000-2020)
            bbox: Optional bounding box to clip data
            
        Returns:
            Population density as xarray DataArray
        """
        filename = f"worldpop_{country_code}_{year}.tif"
        cache_path = self._get_cache_path(filename)
        
        if not self._is_cached(filename):
            logger.info(f"Downloading WorldPop data for {country_code} {year}")
            self._download_population_data(country_code, year, cache_path)
        
        with rasterio.open(cache_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            
        coords = self._get_coordinates(data.shape, transform)
        
        pop_data = xr.DataArray(
            data,
            coords=coords,
            dims=['lat', 'lon'],
            attrs={
                'units': 'people per pixel',
                'source': 'WorldPop',
                'year': year,
                'country': country_code,
                'crs': str(crs)
            }
        )
        
        if bbox:
            pop_data = self._clip_to_bbox(pop_data, bbox)
            
        return pop_data
    
    def _download_population_data(self, country_code: str, year: int, output_path: Path):
        """Download population data from WorldPop."""
        url = f"{self.BASE_URL}/{year}/{country_code.upper()}/{country_code.lower()}_ppp_{year}_1km_Aggregated_UNadj.tif"
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Downloaded WorldPop data to {output_path}")
            
        except requests.RequestException as e:
            logger.error(f"Failed to download WorldPop data: {e}")
            raise
    
    def _get_coordinates(self, shape: Tuple[int, int], transform) -> Dict[str, np.ndarray]:
        """Generate coordinate arrays from raster transform."""
        height, width = shape
        
        lons = np.array([transform * (i, 0) for i in range(width)])[:, 0]
        lats = np.array([transform * (0, j) for j in range(height)])[:, 1]
        
        return {'lat': lats, 'lon': lons}
    
    def _clip_to_bbox(self, data: xr.DataArray, bbox: Dict[str, float]) -> xr.DataArray:
        """Clip data to bounding box."""
        return data.sel(
            lat=slice(bbox['min_lat'], bbox['max_lat']),
            lon=slice(bbox['min_lon'], bbox['max_lon'])
        )


class VIIRSNightlightsLoader(BaseDataLoader):
    """
    Loader for VIIRS nighttime lights data from NOAA.
    
    VIIRS provides monthly and annual nighttime lights composites
    useful for economic activity and urbanization analysis.
    """
    
    BASE_URL = "https://eogdata.mines.edu/nighttime_light/annual/v21"
    
    def __init__(self, cache_dir: str = "data/cache/viirs"):
        super().__init__(cache_dir)
        
    def load_nightlights_data(
        self, 
        year: int = 2022,
        bbox: Optional[Dict[str, float]] = None
    ) -> xr.DataArray:
        """
        Load VIIRS nighttime lights data.
        
        Args:
            year: Year of data (2012-2022)
            bbox: Optional bounding box to clip data
            
        Returns:
            Nighttime lights as xarray DataArray
        """
        filename = f"viirs_nightlights_{year}.tif"
        cache_path = self._get_cache_path(filename)
        
        if not self._is_cached(filename):
            logger.info(f"Downloading VIIRS nightlights data for {year}")
            self._download_nightlights_data(year, cache_path)
        
        with rasterio.open(cache_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            
        coords = self._get_coordinates(data.shape, transform)
        
        lights_data = xr.DataArray(
            data,
            coords=coords,
            dims=['lat', 'lon'],
            attrs={
                'units': 'nanoWatts/cm²/sr',
                'source': 'VIIRS/NOAA',
                'year': year,
                'crs': str(crs)
            }
        )
        
        if bbox:
            lights_data = self._clip_to_bbox(lights_data, bbox)
            
        return lights_data
    
    def _download_nightlights_data(self, year: int, output_path: Path):
        """Download nightlights data from NOAA."""
        url = f"{self.BASE_URL}/{year}/VNL_v21_npp_{year}_global_vcmslcfg_c202205302300.average_masked.tif.gz"
        
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            compressed_path = output_path.with_suffix('.tif.gz')
            with open(compressed_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            import gzip
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            compressed_path.unlink()
            logger.info(f"Downloaded VIIRS nightlights data to {output_path}")
            
        except requests.RequestException as e:
            logger.error(f"Failed to download VIIRS data: {e}")
            raise
    
    def _get_coordinates(self, shape: Tuple[int, int], transform) -> Dict[str, np.ndarray]:
        """Generate coordinate arrays from raster transform."""
        height, width = shape
        
        lons = np.array([transform * (i, 0) for i in range(width)])[:, 0]
        lats = np.array([transform * (0, j) for j in range(height)])[:, 1]
        
        return {'lat': lats, 'lon': lons}
    
    def _clip_to_bbox(self, data: xr.DataArray, bbox: Dict[str, float]) -> xr.DataArray:
        """Clip data to bounding box."""
        return data.sel(
            lat=slice(bbox['min_lat'], bbox['max_lat']),
            lon=slice(bbox['min_lon'], bbox['max_lon'])
        )


class OpenStreetMapLoader(BaseDataLoader):
    """
    Loader for OpenStreetMap roads and infrastructure data.
    
    Uses the Overpass API to query OSM data for roads, buildings,
    and other infrastructure elements.
    """
    
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    
    def __init__(self, cache_dir: str = "data/cache/osm"):
        super().__init__(cache_dir)
        
    def load_roads_data(
        self, 
        bbox: Dict[str, float],
        road_types: List[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Load roads data from OpenStreetMap.
        
        Args:
            bbox: Bounding box for data query
            road_types: List of road types to include (e.g., ['primary', 'secondary'])
            
        Returns:
            Roads as GeoDataFrame
        """
        if road_types is None:
            road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']
            
        bbox_str = f"{bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']}"
        filename = f"osm_roads_{hash(bbox_str)}_{hash(str(road_types))}.geojson"
        cache_path = self._get_cache_path(filename)
        
        if not self._is_cached(filename):
            logger.info(f"Downloading OSM roads data for bbox {bbox_str}")
            self._download_roads_data(bbox, road_types, cache_path)
        
        return gpd.read_file(cache_path)
    
    def _download_roads_data(
        self, 
        bbox: Dict[str, float], 
        road_types: List[str], 
        output_path: Path
    ):
        """Download roads data from Overpass API."""
        bbox_str = f"{bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']}"
        
        highway_filter = '|'.join(road_types)
        query = f"""
        [out:json][timeout:25];
        (
          way["highway"~"^({highway_filter})$"]({bbox_str});
        );
        out geom;
        """
        
        try:
            response = requests.post(
                self.OVERPASS_URL,
                data={'data': query},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            features = []
            for element in data.get('elements', []):
                if element['type'] == 'way' and 'geometry' in element:
                    coords = [(node['lon'], node['lat']) for node in element['geometry']]
                    
                    feature = {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'LineString',
                            'coordinates': coords
                        },
                        'properties': {
                            'highway': element.get('tags', {}).get('highway'),
                            'name': element.get('tags', {}).get('name'),
                            'osm_id': element.get('id')
                        }
                    }
                    features.append(feature)
            
            geojson = {
                'type': 'FeatureCollection',
                'features': features
            }
            
            import json
            with open(output_path, 'w') as f:
                json.dump(geojson, f)
                
            logger.info(f"Downloaded OSM roads data to {output_path}")
            
        except requests.RequestException as e:
            logger.error(f"Failed to download OSM data: {e}")
            raise


class CopernicusLandCoverLoader(BaseDataLoader):
    """
    Loader for Copernicus Global Land Cover data.
    
    Provides annual global land cover maps at 100m resolution
    from the Copernicus Land Monitoring Service.
    """
    
    BASE_URL = "https://zenodo.org/record/3939050/files"
    
    def __init__(self, cache_dir: str = "data/cache/copernicus"):
        super().__init__(cache_dir)
        
    def load_land_cover_data(
        self, 
        year: int = 2019,
        bbox: Optional[Dict[str, float]] = None
    ) -> xr.DataArray:
        """
        Load Copernicus Global Land Cover data.
        
        Args:
            year: Year of data (2015-2019)
            bbox: Optional bounding box to clip data
            
        Returns:
            Land cover as xarray DataArray
        """
        filename = f"copernicus_landcover_{year}.tif"
        cache_path = self._get_cache_path(filename)
        
        if not self._is_cached(filename):
            logger.info(f"Downloading Copernicus land cover data for {year}")
            self._download_land_cover_data(year, cache_path)
        
        with rasterio.open(cache_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            
        coords = self._get_coordinates(data.shape, transform)
        
        landcover_data = xr.DataArray(
            data,
            coords=coords,
            dims=['lat', 'lon'],
            attrs={
                'units': 'land cover class',
                'source': 'Copernicus Global Land Service',
                'year': year,
                'crs': str(crs)
            }
        )
        
        if bbox:
            landcover_data = self._clip_to_bbox(landcover_data, bbox)
            
        return landcover_data
    
    def _download_land_cover_data(self, year: int, output_path: Path):
        """Download land cover data from Copernicus."""
        url = f"{self.BASE_URL}/PROBAV_LC100_global_v3.0.1_{year}-nrt_Discrete-Classification-map_EPSG-4326.tif"
        
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Downloaded Copernicus land cover data to {output_path}")
            
        except requests.RequestException as e:
            logger.error(f"Failed to download Copernicus data: {e}")
            raise
    
    def _get_coordinates(self, shape: Tuple[int, int], transform) -> Dict[str, np.ndarray]:
        """Generate coordinate arrays from raster transform."""
        height, width = shape
        
        lons = np.array([transform * (i, 0) for i in range(width)])[:, 0]
        lats = np.array([transform * (0, j) for j in range(height)])[:, 1]
        
        return {'lat': lats, 'lon': lons}
    
    def _clip_to_bbox(self, data: xr.DataArray, bbox: Dict[str, float]) -> xr.DataArray:
        """Clip data to bounding box."""
        return data.sel(
            lat=slice(bbox['min_lat'], bbox['max_lat']),
            lon=slice(bbox['min_lon'], bbox['max_lon'])
        )


class RealWorldDataManager:
    """
    Manager class for coordinating multiple real-world data sources.
    
    Provides a unified interface for loading and integrating data from
    multiple sources with efficient caching and storage.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.worldpop = WorldPopDataLoader(str(self.cache_dir / "worldpop"))
        self.viirs = VIIRSNightlightsLoader(str(self.cache_dir / "viirs"))
        self.osm = OpenStreetMapLoader(str(self.cache_dir / "osm"))
        self.copernicus = CopernicusLandCoverLoader(str(self.cache_dir / "copernicus"))
        
    def load_integrated_dataset(
        self,
        bbox: Dict[str, float],
        year: int = 2020,
        country_code: Optional[str] = None
    ) -> Dict[str, Union[xr.DataArray, gpd.GeoDataFrame]]:
        """
        Load integrated dataset from multiple sources.
        
        Args:
            bbox: Bounding box for data
            year: Year for temporal data
            country_code: Country code for population data
            
        Returns:
            Dictionary containing all loaded datasets
        """
        logger.info("Loading integrated real-world dataset")
        
        datasets = {}
        
        try:
            if country_code:
                datasets['population'] = self.worldpop.load_population_data(
                    country_code, year, bbox
                )
                logger.info("✅ Population data loaded")
        except Exception as e:
            logger.warning(f"Failed to load population data: {e}")
            
        try:
            datasets['nightlights'] = self.viirs.load_nightlights_data(year, bbox)
            logger.info("✅ Nightlights data loaded")
        except Exception as e:
            logger.warning(f"Failed to load nightlights data: {e}")
            
        try:
            datasets['roads'] = self.osm.load_roads_data(bbox)
            logger.info("✅ Roads data loaded")
        except Exception as e:
            logger.warning(f"Failed to load roads data: {e}")
            
        try:
            datasets['land_cover'] = self.copernicus.load_land_cover_data(year, bbox)
            logger.info("✅ Land cover data loaded")
        except Exception as e:
            logger.warning(f"Failed to load land cover data: {e}")
            
        logger.info(f"Integrated dataset loaded with {len(datasets)} data sources")
        return datasets
    
    def save_to_efficient_format(
        self,
        datasets: Dict[str, Union[xr.DataArray, gpd.GeoDataFrame]],
        output_dir: str
    ):
        """
        Save datasets in efficient formats (GeoParquet, Zarr).
        
        Args:
            datasets: Dictionary of datasets to save
            output_dir: Output directory for saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, data in datasets.items():
            if isinstance(data, xr.DataArray):
                zarr_path = output_path / f"{name}.zarr"
                data.to_zarr(zarr_path, mode='w')
                logger.info(f"Saved {name} to Zarr format: {zarr_path}")
                
            elif isinstance(data, gpd.GeoDataFrame):
                parquet_path = output_path / f"{name}.parquet"
                data.to_parquet(parquet_path)
                logger.info(f"Saved {name} to GeoParquet format: {parquet_path}")
