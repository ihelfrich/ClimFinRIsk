# Open Source Climate and Geospatial Data Sources

## Climate and Weather Data

### 1. Copernicus Climate Data Store (CDS)
- **URL**: https://cds.climate.copernicus.eu/
- **API**: Climate Data Store API (cdsapi)
- **Data Types**: Temperature, precipitation, wind, humidity, climate projections
- **Coverage**: Global, high resolution
- **Access**: Free registration required
- **Python Package**: `cdsapi`

### 2. NOAA Climate Data
- **URL**: https://www.ncei.noaa.gov/
- **API**: NOAA Climate Data Online API
- **Data Types**: Historical weather, climate normals, extreme events
- **Coverage**: Global, focus on US
- **Access**: Free with API key
- **Python Package**: `pyncei`

### 3. NASA Giovanni
- **URL**: https://giovanni.gsfc.nasa.gov/giovanni/
- **API**: OPeNDAP/REST services
- **Data Types**: Satellite-based climate data, precipitation, temperature
- **Coverage**: Global satellite coverage
- **Access**: Free
- **Python Package**: `pydap`, `xarray`

### 4. European Centre for Medium-Range Weather Forecasts (ECMWF)
- **URL**: https://www.ecmwf.int/
- **API**: ECMWF Web API
- **Data Types**: Weather forecasts, reanalysis data (ERA5)
- **Coverage**: Global
- **Access**: Free for research
- **Python Package**: `ecmwf-api-client`

## Hazard-Specific Data

### 5. Global Flood Database
- **URL**: https://global-flood-database.cloudtostreet.ai/
- **Data Types**: Historical flood extents, flood frequency
- **Coverage**: Global
- **Access**: Open access
- **Format**: Cloud Optimized GeoTIFF

### 6. USGS Earthquake Hazards Program
- **URL**: https://earthquake.usgs.gov/
- **API**: USGS Earthquake API
- **Data Types**: Earthquake catalogs, hazard maps
- **Coverage**: Global
- **Access**: Free
- **Python Package**: `obspy`

### 7. NOAA Storm Events Database
- **URL**: https://www.ncdc.noaa.gov/stormevents/
- **Data Types**: Storm tracks, intensity, damage reports
- **Coverage**: United States
- **Access**: Free download
- **Format**: CSV, Shapefile

## Geospatial and Demographic Data

### 8. WorldPop
- **URL**: https://www.worldpop.org/
- **API**: WorldPop API
- **Data Types**: Population density, demographics, urban growth
- **Coverage**: Global, 100m resolution
- **Access**: Open access
- **Format**: GeoTIFF

### 9. VIIRS Nighttime Lights
- **URL**: https://eogdata.mines.edu/nighttime_light/
- **Data Types**: Nighttime lights, economic activity indicators
- **Coverage**: Global, monthly/annual composites
- **Access**: Free download
- **Format**: GeoTIFF

### 10. OpenStreetMap
- **URL**: https://www.openstreetmap.org/
- **API**: Overpass API, Nominatim
- **Data Types**: Roads, buildings, infrastructure, land use
- **Coverage**: Global, crowd-sourced
- **Access**: Open access
- **Python Package**: `osmnx`, `overpy`

### 11. Copernicus Land Monitoring Service
- **URL**: https://land.copernicus.eu/
- **Data Types**: Land cover, land use change, urban atlas
- **Coverage**: Global and European focus
- **Access**: Free registration
- **Format**: GeoTIFF, Vector

### 12. NASA MODIS Land Cover
- **URL**: https://modis.gsfc.nasa.gov/data/dataprod/mod12.php
- **Data Types**: Global land cover classification
- **Coverage**: Global, 500m resolution
- **Access**: Free
- **Format**: HDF, GeoTIFF

## Economic and Financial Data

### 13. World Bank Open Data
- **URL**: https://data.worldbank.org/
- **API**: World Bank API
- **Data Types**: GDP, economic indicators, development data
- **Coverage**: Country-level global
- **Access**: Open access
- **Python Package**: `wbdata`

### 14. OECD Data
- **URL**: https://data.oecd.org/
- **API**: OECD API
- **Data Types**: Economic statistics, environmental indicators
- **Coverage**: OECD countries
- **Access**: Open access
- **Python Package**: `pandas-datareader`

### 15. Federal Reserve Economic Data (FRED)
- **URL**: https://fred.stlouisfed.org/
- **API**: FRED API
- **Data Types**: Economic time series, financial indicators
- **Coverage**: United States focus
- **Access**: Free with API key
- **Python Package**: `fredapi`

## Elevation and Topography

### 16. NASA SRTM Digital Elevation Model
- **URL**: https://www2.jpl.nasa.gov/srtm/
- **Data Types**: Digital elevation models, topography
- **Coverage**: Global, 30m resolution
- **Access**: Free
- **Format**: GeoTIFF

### 17. USGS National Elevation Dataset
- **URL**: https://www.usgs.gov/core-science-systems/ngp/3dep
- **Data Types**: High-resolution elevation data
- **Coverage**: United States
- **Access**: Free
- **Format**: Various

## Ocean and Coastal Data

### 18. NOAA Sea Level Rise Viewer
- **URL**: https://coast.noaa.gov/slr/
- **Data Types**: Sea level rise projections, coastal vulnerability
- **Coverage**: US coastlines
- **Access**: Free
- **Format**: Web services, downloads

### 19. Global Self-consistent, Hierarchical, High-resolution Geography (GSHHG)
- **URL**: https://www.soest.hawaii.edu/pwessel/gshhg/
- **Data Types**: Coastlines, political boundaries
- **Coverage**: Global
- **Access**: Open access
- **Format**: Shapefile

## Satellite Imagery

### 20. Landsat (USGS/NASA)
- **URL**: https://landsat.gsfc.nasa.gov/
- **API**: USGS Earth Explorer API
- **Data Types**: Multispectral satellite imagery
- **Coverage**: Global, 30m resolution
- **Access**: Free
- **Python Package**: `landsatxplore`

### 21. Sentinel Hub
- **URL**: https://www.sentinel-hub.com/
- **API**: Sentinel Hub API
- **Data Types**: Sentinel-1/2 satellite data
- **Coverage**: Global
- **Access**: Free tier available
- **Python Package**: `sentinelhub`

## Climate Projections and Scenarios

### 22. CMIP6 Climate Model Data
- **URL**: https://esgf-node.llnl.gov/projects/cmip6/
- **Data Types**: Climate model projections, scenarios
- **Coverage**: Global climate models
- **Access**: Free registration
- **Format**: NetCDF

### 23. WorldClim
- **URL**: https://www.worldclim.org/
- **Data Types**: Climate surfaces, future climate projections
- **Coverage**: Global, 1km resolution
- **Access**: Free download
- **Format**: GeoTIFF

### 24. Climate Impact Lab
- **URL**: https://www.impactlab.org/
- **Data Types**: Climate impact projections, economic damages
- **Coverage**: Global
- **Access**: Open access datasets
- **Format**: Various

## Data Integration Platforms

### 25. Google Earth Engine
- **URL**: https://earthengine.google.com/
- **API**: Earth Engine API
- **Data Types**: Petabyte-scale geospatial analysis platform
- **Coverage**: Global satellite and climate data
- **Access**: Free for research
- **Python Package**: `earthengine-api`

### 26. Microsoft Planetary Computer
- **URL**: https://planetarycomputer.microsoft.com/
- **API**: STAC API
- **Data Types**: Curated geospatial datasets
- **Coverage**: Global
- **Access**: Free
- **Python Package**: `planetary-computer`

### 27. AWS Open Data
- **URL**: https://aws.amazon.com/opendata/
- **Data Types**: Climate, satellite, and geospatial datasets
- **Coverage**: Various
- **Access**: Free (pay for compute)
- **Format**: Cloud-optimized formats

## API Integration Best Practices

### Authentication
- Most APIs require registration and API keys
- Store credentials securely using environment variables
- Respect rate limits and usage quotas

### Data Formats
- **Raster Data**: GeoTIFF, NetCDF, HDF
- **Vector Data**: Shapefile, GeoJSON, GeoParquet
- **Time Series**: CSV, JSON, Parquet

### Python Libraries for Integration
```python
# Climate data
import cdsapi
import xarray as xr
import pyncei

# Geospatial data
import geopandas as gpd
import rasterio
import folium

# API clients
import requests
import earthengine as ee
```

### Efficient Data Storage
- **Zarr**: For large multidimensional arrays
- **GeoParquet**: For vector geospatial data
- **Cloud Optimized GeoTIFF**: For raster data
- **Dask**: For parallel processing of large datasets

This comprehensive list provides access to the most valuable open-source datasets for climate risk modeling and geospatial analysis.
