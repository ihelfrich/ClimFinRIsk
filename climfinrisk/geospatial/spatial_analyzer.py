"""
Spatial analysis module for climate risk mapping and geospatial analytics.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point, Polygon, LineString
import folium
from folium import plugins
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class SpatialAnalyzer:
    """
    Geospatial analysis for climate risk assessment.
    
    Provides:
    - Asset-level spatial risk mapping
    - Geographic risk clustering and hotspot identification
    - Interactive map generation
    - Spatial network analysis for risk propagation
    """
    
    def __init__(self, config=None):
        """
        Initialize spatial analyzer with configuration.
        
        Args:
            config: Configuration object with mapping parameters
        """
        self.config = config or {}
        self.risk_maps = {}
        try:
            from ..data.real_world_sources import RealWorldDataManager
            self.data_manager = RealWorldDataManager()
        except ImportError:
            self.data_manager = None
        
    def analyze_spatial_risk(
        self,
        risk_estimates: pd.DataFrame,
        assets: pd.DataFrame,
        output_dir: str = "outputs"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive spatial risk analysis.
        
        Args:
            risk_estimates: Risk estimates for each asset
            assets: Asset portfolio with geographic information
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing spatial analysis results
        """
        logger.info("Starting spatial risk analysis")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        spatial_data = self._prepare_spatial_data(risk_estimates, assets)
        
        gdf = self._create_geodataframe(spatial_data)
        
        results = {}
        
        logger.info("Identifying risk hotspots")
        hotspots = self._identify_risk_hotspots(gdf)
        results['hotspots'] = hotspots
        
        logger.info("Performing spatial clustering")
        clusters = self._perform_spatial_clustering(gdf)
        results['clusters'] = clusters
        
        logger.info("Generating risk maps")
        maps = self._generate_risk_maps(gdf, output_path)
        results['maps'] = maps
        
        logger.info("Calculating spatial statistics")
        spatial_stats = self._calculate_spatial_statistics(gdf)
        results['spatial_statistics'] = spatial_stats
        
        logger.info("Creating interactive dashboard")
        dashboard = self._create_interactive_dashboard(gdf, output_path)
        results['dashboard'] = dashboard
        
        logger.info("Spatial risk analysis completed")
        return results
    
    def create_risk_map(
        self,
        risk_data: pd.DataFrame,
        risk_column: str = 'expected_loss',
        map_type: str = 'choropleth',
        scenario: str = 'rcp85',
        output_path: str = None
    ) -> str:
        """
        Create a risk map for visualization.
        
        Args:
            risk_data: DataFrame with risk estimates and coordinates
            map_type: Type of map ('choropleth', 'heatmap', 'bubble')
            scenario: Climate scenario to visualize
            output_path: Path to save the map
            
        Returns:
            Path to generated map file
        """
        logger.info(f"Creating {map_type} risk map for scenario {scenario}")
        
        if risk_column not in risk_data.columns:
            scenario_column = f'expected_loss_{scenario}'
            if scenario_column in risk_data.columns:
                risk_column = scenario_column
            else:
                risk_column = 'expected_loss'  # Fallback
        
        center_lat = risk_data['lat'].mean()
        center_lon = risk_data['lon'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        if map_type == 'heatmap':
            heat_data = [[row['lat'], row['lon'], row[risk_column]] 
                        for idx, row in risk_data.iterrows() if not pd.isna(row[risk_column])]
            
            plugins.HeatMap(heat_data).add_to(m)
            
        elif map_type == 'bubble':
            max_risk = risk_data[risk_column].max()
            if max_risk == 0 or pd.isna(max_risk):
                max_risk = 1.0  # Avoid division by zero
            
            for idx, row in risk_data.iterrows():
                if pd.isna(row[risk_column]):
                    continue
                
                risk_value = row[risk_column] if not pd.isna(row[risk_column]) else 0
                radius = max(5, (risk_value / max_risk) * 50)
                
                color = self._get_risk_color(row[risk_column], risk_data[risk_column])
                
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=radius,
                    popup=f"Asset: {row.get('asset_id', 'Unknown')}<br>"
                          f"Risk: ${row[risk_column]:,.0f}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6
                ).add_to(m)
        
        else:  # choropleth or default
            for idx, row in risk_data.iterrows():
                if pd.isna(row[risk_column]):
                    continue
                
                color = self._get_risk_color(row[risk_column], risk_data[risk_column])
                
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    popup=f"Asset: {row.get('asset_id', 'Unknown')}<br>"
                          f"Type: {row.get('asset_type', 'Unknown')}<br>"
                          f"Risk: ${row[risk_column]:,.0f}",
                    icon=folium.Icon(color=color)
                ).add_to(m)
        
        self._add_map_legend(m, risk_data[risk_column])
        
        if output_path is None:
            output_path = f"risk_map_{map_type}_{scenario}.html"
        
        m.save(output_path)
        logger.info(f"Risk map saved to {output_path}")
        
        return output_path
    
    def _prepare_spatial_data(
        self,
        risk_estimates: pd.DataFrame,
        assets: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine risk estimates with asset spatial data."""
        if 'asset_id' in risk_estimates.columns and 'asset_id' in assets.columns:
            spatial_data = pd.merge(assets, risk_estimates, on='asset_id', how='left')
        else:
            spatial_data = pd.concat([assets, risk_estimates], axis=1)
        
        required_cols = ['lat', 'lon']
        for col in required_cols:
            if col not in spatial_data.columns:
                logger.warning(f"Missing required column: {col}")
                spatial_data[col] = 0.0
        
        return spatial_data
    
    def _create_geodataframe(self, spatial_data: pd.DataFrame) -> gpd.GeoDataFrame:
        """Create GeoDataFrame from spatial data."""
        geometry = [Point(lon, lat) for lon, lat in zip(spatial_data['lon'], spatial_data['lat'])]
        
        gdf = gpd.GeoDataFrame(spatial_data, geometry=geometry, crs='EPSG:4326')
        
        return gdf
    
    def _identify_risk_hotspots(
        self,
        gdf: gpd.GeoDataFrame,
        risk_column: str = 'expected_loss_rcp85'
    ) -> Dict[str, Any]:
        """Identify geographic risk hotspots."""
        hotspots = {}
        
        risk_values = gdf[risk_column].dropna()
        if len(risk_values) == 0:
            return hotspots
        
        high_risk_threshold = risk_values.quantile(0.9)
        medium_risk_threshold = risk_values.quantile(0.7)
        
        gdf['risk_category'] = 'Low'
        gdf.loc[gdf[risk_column] >= medium_risk_threshold, 'risk_category'] = 'Medium'
        gdf.loc[gdf[risk_column] >= high_risk_threshold, 'risk_category'] = 'High'
        
        risk_counts = gdf['risk_category'].value_counts()
        
        hotspots = {
            'high_risk_assets': gdf[gdf['risk_category'] == 'High'],
            'medium_risk_assets': gdf[gdf['risk_category'] == 'Medium'],
            'risk_counts': risk_counts.to_dict(),
            'thresholds': {
                'high_risk': high_risk_threshold,
                'medium_risk': medium_risk_threshold
            }
        }
        
        return hotspots
    
    def _perform_spatial_clustering(
        self,
        gdf: gpd.GeoDataFrame,
        n_clusters: int = 5
    ) -> Dict[str, Any]:
        """Perform spatial clustering of assets."""
        try:
            from sklearn.cluster import KMeans
            
            coords = np.array([[point.x, point.y] for point in gdf.geometry])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(coords)
            
            gdf['cluster'] = cluster_labels
            
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                cluster_data = gdf[gdf['cluster'] == cluster_id]
                
                cluster_stats[cluster_id] = {
                    'n_assets': len(cluster_data),
                    'mean_risk': cluster_data['expected_loss_rcp85'].mean(),
                    'total_risk': cluster_data['expected_loss_rcp85'].sum(),
                    'center': {
                        'lat': cluster_data['lat'].mean(),
                        'lon': cluster_data['lon'].mean()
                    }
                }
            
            return {
                'cluster_labels': cluster_labels,
                'cluster_centers': kmeans.cluster_centers_,
                'cluster_statistics': cluster_stats,
                'gdf_with_clusters': gdf
            }
            
        except ImportError:
            logger.warning("Scikit-learn not available for clustering")
            return {}
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {}
    
    def _generate_risk_maps(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: Path
    ) -> Dict[str, str]:
        """Generate various types of risk maps."""
        maps = {}
        
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        df['lat'] = [point.y for point in gdf.geometry]
        df['lon'] = [point.x for point in gdf.geometry]
        
        map_types = ['heatmap', 'bubble', 'choropleth']
        scenarios = ['rcp26', 'rcp45', 'rcp85']
        
        for map_type in map_types:
            for scenario in scenarios:
                try:
                    map_file = output_path / f"risk_map_{map_type}_{scenario}.html"
                    self.create_risk_map(
                        df, 
                        map_type=map_type, 
                        scenario=scenario,
                        output_path=str(map_file)
                    )
                    maps[f"{map_type}_{scenario}"] = str(map_file)
                except Exception as e:
                    logger.warning(f"Failed to create {map_type} map for {scenario}: {e}")
        
        return maps
    
    def _calculate_spatial_statistics(self, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Calculate spatial statistics for risk distribution."""
        stats = {}
        
        stats['geographic_extent'] = {
            'min_lat': gdf['lat'].min(),
            'max_lat': gdf['lat'].max(),
            'min_lon': gdf['lon'].min(),
            'max_lon': gdf['lon'].max()
        }
        
        stats['risk_by_region'] = self._analyze_risk_by_region(gdf)
        
        stats['risk_concentration'] = self._calculate_risk_concentration(gdf)
        
        return stats
    
    def _analyze_risk_by_region(self, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Analyze risk distribution by geographic regions."""
        center_lat = gdf['lat'].median()
        center_lon = gdf['lon'].median()
        
        gdf['region'] = 'Unknown'
        gdf.loc[(gdf['lat'] >= center_lat) & (gdf['lon'] >= center_lon), 'region'] = 'Northeast'
        gdf.loc[(gdf['lat'] >= center_lat) & (gdf['lon'] < center_lon), 'region'] = 'Northwest'
        gdf.loc[(gdf['lat'] < center_lat) & (gdf['lon'] >= center_lon), 'region'] = 'Southeast'
        gdf.loc[(gdf['lat'] < center_lat) & (gdf['lon'] < center_lon), 'region'] = 'Southwest'
        
        regional_stats = {}
        for region in gdf['region'].unique():
            region_data = gdf[gdf['region'] == region]
            
            regional_stats[region] = {
                'n_assets': len(region_data),
                'total_risk': region_data['expected_loss_rcp85'].sum(),
                'mean_risk': region_data['expected_loss_rcp85'].mean(),
                'max_risk': region_data['expected_loss_rcp85'].max()
            }
        
        return regional_stats
    
    def _calculate_risk_concentration(self, gdf: gpd.GeoDataFrame) -> Dict[str, float]:
        """Calculate risk concentration metrics."""
        risk_values = gdf['expected_loss_rcp85'].dropna()
        
        if len(risk_values) == 0:
            return {}
        
        sorted_risks = np.sort(risk_values)
        n = len(sorted_risks)
        cumsum = np.cumsum(sorted_risks)
        if cumsum[-1] == 0 or pd.isna(cumsum[-1]):
            gini = 0.0
        else:
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        risk_shares = risk_values / risk_values.sum()
        hhi = np.sum(risk_shares ** 2)
        
        return {
            'gini_coefficient': gini,
            'herfindahl_hirschman_index': hhi,
            'top_10_percent_share': (risk_values.nlargest(int(0.1 * len(risk_values))).sum() / risk_values.sum()) if risk_values.sum() > 0 else 0.0
        }
    
    def _create_interactive_dashboard(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: Path
    ) -> str:
        """Create an interactive risk dashboard."""
        dashboard_file = output_path / "risk_dashboard.html"
        
        center_lat = gdf['lat'].mean()
        center_lon = gdf['lon'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        scenarios = ['rcp26', 'rcp45', 'rcp85']
        colors = ['green', 'orange', 'red']
        
        for scenario, color in zip(scenarios, colors):
            risk_column = f'expected_loss_{scenario}'
            if risk_column in gdf.columns:
                feature_group = folium.FeatureGroup(name=f'{scenario.upper()} Scenario')
                
                for idx, row in gdf.iterrows():
                    if pd.isna(row[risk_column]):
                        continue
                    
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=max(3, (row[risk_column] / max(gdf[risk_column].max(), 1.0)) * 20) if not pd.isna(row[risk_column]) else 3,
                        popup=f"Asset: {row.get('asset_id', 'Unknown')}<br>"
                              f"Scenario: {scenario.upper()}<br>"
                              f"Risk: ${row[risk_column]:,.0f}",
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.6
                    ).add_to(feature_group)
                
                feature_group.add_to(m)
        
        folium.LayerControl().add_to(m)
        
        m.save(str(dashboard_file))
        logger.info(f"Interactive dashboard saved to {dashboard_file}")
        
        return str(dashboard_file)
    
    def calculate_spatial_statistics(self, assets_data, risk_column='expected_loss'):
        """Calculate spatial statistics for risk distribution"""
        gdf = self._create_geodataframe(assets_data)
        return self._calculate_risk_concentration(gdf)
    
    def identify_risk_clusters(self, assets_data, risk_column='expected_loss', n_clusters=5):
        """Identify spatial risk clusters"""
        gdf = self._create_geodataframe(assets_data)
        return self._perform_spatial_clustering(gdf, n_clusters)
    
    def create_interactive_dashboard(self, assets_data, risk_column='expected_loss'):
        """Create interactive risk dashboard"""
        from pathlib import Path
        output_path = Path("../outputs/demo_results")
        output_path.mkdir(parents=True, exist_ok=True)
        return self._create_interactive_dashboard(assets_data, output_path)

    def _get_risk_color(self, risk_value: float, risk_series: pd.Series) -> str:
        """Get color based on risk level."""
        if pd.isna(risk_value):
            return 'gray'
        
        low_threshold = risk_series.quantile(0.33)
        high_threshold = risk_series.quantile(0.67)
        
        if risk_value <= low_threshold:
            return 'green'
        elif risk_value <= high_threshold:
            return 'orange'
        else:
            return 'red'
    
    def _add_map_legend(self, folium_map: folium.Map, risk_series: pd.Series):
        """Add legend to folium map."""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Risk Level</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> Low</p>
    
    def _get_asset_bbox(self, assets: pd.DataFrame) -> Dict[str, float]:
        """Extract bounding box from asset coordinates."""
        return {
            'min_lat': assets['lat'].min() - 0.1,
            'max_lat': assets['lat'].max() + 0.1,
            'min_lon': assets['lon'].min() - 0.1,
            'max_lon': assets['lon'].max() + 0.1
        }
    
    def _load_real_world_data(self, bbox: Dict[str, float]) -> Dict[str, Any]:
        """Load real-world geospatial data for the analysis area."""
        logger.info("Loading real-world geospatial data")
        
        if self.data_manager:
            try:
                datasets = self.data_manager.load_integrated_dataset(
                    bbox=bbox,
                    year=2020,
                    country_code='USA'
                )
                logger.info(f"Loaded {len(datasets)} real-world datasets")
                return datasets
            except Exception as e:
                logger.warning(f"Failed to load some real-world data: {e}")
        
        return self._create_synthetic_real_world_data(bbox)
    
    def _create_synthetic_real_world_data(self, bbox: Dict[str, float]) -> Dict[str, Any]:
        """Create synthetic real-world data for demonstration."""
        logger.info("Creating synthetic real-world data for demonstration")
        
        lat_range = np.linspace(bbox['min_lat'], bbox['max_lat'], 50)
        lon_range = np.linspace(bbox['min_lon'], bbox['max_lon'], 50)
        
        population = xr.DataArray(
            np.random.exponential(100, (50, 50)),
            coords={'lat': lat_range, 'lon': lon_range},
            dims=['lat', 'lon'],
            attrs={'units': 'people per km²', 'source': 'Synthetic WorldPop'}
        )
        
        nightlights = xr.DataArray(
            np.random.gamma(2, 5, (50, 50)),
            coords={'lat': lat_range, 'lon': lon_range},
            dims=['lat', 'lon'],
            attrs={'units': 'nanoWatts/cm²/sr', 'source': 'Synthetic VIIRS'}
        )
        
        land_cover = xr.DataArray(
            np.random.choice([10, 20, 30, 40, 50], (50, 50)),
            coords={'lat': lat_range, 'lon': lon_range},
            dims=['lat', 'lon'],
            attrs={'units': 'land cover class', 'source': 'Synthetic Copernicus'}
        )
        
        n_roads = 20
        road_coords = []
        for _ in range(n_roads):
            start_lat = np.random.uniform(bbox['min_lat'], bbox['max_lat'])
            start_lon = np.random.uniform(bbox['min_lon'], bbox['max_lon'])
            end_lat = start_lat + np.random.uniform(-0.05, 0.05)
            end_lon = start_lon + np.random.uniform(-0.05, 0.05)
            road_coords.append([[start_lon, start_lat], [end_lon, end_lat]])
        
        roads_gdf = gpd.GeoDataFrame({
            'highway': np.random.choice(['primary', 'secondary', 'tertiary'], n_roads),
            'geometry': [LineString(coords) for coords in road_coords]
        })
        
        return {
            'population': population,
            'nightlights': nightlights,
            'land_cover': land_cover,
            'roads': roads_gdf
        }
    
    def _enhance_with_real_world_data(
        self, 
        gdf: gpd.GeoDataFrame, 
        real_world_datasets: Dict[str, Any]
    ) -> gpd.GeoDataFrame:
        """Enhance asset data with real-world geospatial information."""
        logger.info("Enhancing asset data with real-world information")
        
        enhanced_gdf = gdf.copy()
        
        if 'population' in real_world_datasets:
            pop_data = real_world_datasets['population']
            enhanced_gdf['population_density'] = enhanced_gdf.apply(
                lambda row: self._sample_raster_at_point(pop_data, row.geometry.x, row.geometry.y),
                axis=1
            )
        
        if 'nightlights' in real_world_datasets:
            lights_data = real_world_datasets['nightlights']
            enhanced_gdf['nightlight_intensity'] = enhanced_gdf.apply(
                lambda row: self._sample_raster_at_point(lights_data, row.geometry.x, row.geometry.y),
                axis=1
            )
        
        if 'land_cover' in real_world_datasets:
            lc_data = real_world_datasets['land_cover']
            enhanced_gdf['land_cover_class'] = enhanced_gdf.apply(
                lambda row: self._sample_raster_at_point(lc_data, row.geometry.x, row.geometry.y),
                axis=1
            )
        
        if 'roads' in real_world_datasets:
            roads_gdf = real_world_datasets['roads']
            enhanced_gdf['distance_to_road'] = enhanced_gdf.apply(
                lambda row: self._calculate_distance_to_nearest_road(row.geometry, roads_gdf),
                axis=1
            )
        
        enhanced_gdf['urban_index'] = self._calculate_urban_index(enhanced_gdf)
        enhanced_gdf['exposure_multiplier'] = self._calculate_exposure_multiplier(enhanced_gdf)
        
        return enhanced_gdf
    
    def _sample_raster_at_point(self, raster: xr.DataArray, lon: float, lat: float) -> float:
        """Sample raster value at a specific point."""
        try:
            value = raster.sel(lat=lat, lon=lon, method='nearest').values
            return float(value) if not np.isnan(value) else 0.0
        except:
            return 0.0
    
    def _calculate_distance_to_nearest_road(self, point, roads_gdf: gpd.GeoDataFrame) -> float:
        """Calculate distance to nearest road."""
        if roads_gdf.empty:
            return 1000.0
        
        try:
            distances = roads_gdf.geometry.distance(point)
            return float(distances.min()) * 111000
        except:
            return 1000.0
    
    def _calculate_urban_index(self, gdf: gpd.GeoDataFrame) -> pd.Series:
        """Calculate urban development index from multiple indicators."""
        urban_index = pd.Series(0.0, index=gdf.index)
        
        if 'population_density' in gdf.columns:
            pop_norm = (gdf['population_density'] - gdf['population_density'].min()) / \
                      (gdf['population_density'].max() - gdf['population_density'].min() + 1e-6)
            urban_index += 0.4 * pop_norm
        
        if 'nightlight_intensity' in gdf.columns:
            lights_norm = (gdf['nightlight_intensity'] - gdf['nightlight_intensity'].min()) / \
                         (gdf['nightlight_intensity'].max() - gdf['nightlight_intensity'].min() + 1e-6)
            urban_index += 0.3 * lights_norm
        
        if 'distance_to_road' in gdf.columns:
            road_proximity = 1 / (1 + gdf['distance_to_road'] / 1000)
            urban_index += 0.3 * road_proximity
        
        return urban_index.fillna(0.0)
    
    def _calculate_exposure_multiplier(self, gdf: gpd.GeoDataFrame) -> pd.Series:
        """Calculate exposure multiplier based on real-world factors."""
        multiplier = pd.Series(1.0, index=gdf.index)
        
        if 'urban_index' in gdf.columns:
            multiplier *= (1 + 0.5 * gdf['urban_index'])
        
        if 'land_cover_class' in gdf.columns:
            lc_multipliers = {10: 1.2, 20: 1.5, 30: 0.8, 40: 0.9, 50: 1.0}
            multiplier *= gdf['land_cover_class'].map(lc_multipliers).fillna(1.0)
        
        return multiplier
    
    def _create_enhanced_risk_map(
        self,
        gdf: gpd.GeoDataFrame,
        real_world_datasets: Dict[str, Any],
        output_path: str
    ) -> str:
        """Create enhanced interactive risk map with real-world data layers."""
        logger.info("Creating enhanced interactive risk map")
        
        center_lat = gdf.geometry.y.mean()
        center_lon = gdf.geometry.x.mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        
        if 'roads' in real_world_datasets:
            self._add_roads_layer(m, real_world_datasets['roads'])
        
        risk_columns = [col for col in gdf.columns if 'expected_loss' in col]
        if risk_columns:
            primary_risk = risk_columns[0]
            
            for idx, row in gdf.iterrows():
                risk_value = row[primary_risk]
                
                if risk_value > 0:
                    color = self._get_risk_color(risk_value, gdf[primary_risk].max())
                    
                    popup_text = f"""
                    <b>Asset ID:</b> {row.get('asset_id', idx)}<br>
                    <b>Risk Value:</b> ${risk_value:,.0f}<br>
                    <b>Asset Type:</b> {row.get('type', 'Unknown')}<br>
                    <b>Population Density:</b> {row.get('population_density', 0):.1f}<br>
                    <b>Nightlight Intensity:</b> {row.get('nightlight_intensity', 0):.1f}<br>
                    <b>Urban Index:</b> {row.get('urban_index', 0):.2f}<br>
                    <b>Distance to Road:</b> {row.get('distance_to_road', 0):.0f}m
                    """
                    
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=max(5, min(20, risk_value / gdf[primary_risk].max() * 20)),
                        popup=folium.Popup(popup_text, max_width=300),
                        color=color,
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=2
                    ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        legend_html = self._create_map_legend()
        m.get_root().html.add_child(folium.Element(legend_html))
        
        m.save(output_path)
        logger.info(f"Enhanced risk map saved to {output_path}")
        
        return output_path
    
    def _add_roads_layer(self, m: folium.Map, roads_gdf: gpd.GeoDataFrame):
        """Add roads layer to map."""
        try:
            road_colors = {
                'motorway': '#FF0000',
                'trunk': '#FF4500',
                'primary': '#FFA500',
                'secondary': '#FFFF00',
                'tertiary': '#90EE90'
            }
            
            for idx, road in roads_gdf.iterrows():
                coords = [[point[1], point[0]] for point in road.geometry.coords]
                color = road_colors.get(road.get('highway', 'tertiary'), '#90EE90')
                
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=2,
                    opacity=0.8,
                    popup=f"Road: {road.get('highway', 'Unknown')}"
                ).add_to(m)
                
        except Exception as e:
            logger.warning(f"Could not add roads layer: {e}")
    
    def _create_live_dashboard(
        self,
        gdf: gpd.GeoDataFrame,
        real_world_datasets: Dict[str, Any],
        hotspots: Dict[str, Any],
        clusters: Dict[str, Any],
        output_path: str
    ) -> str:
        """Create comprehensive live dashboard with real-world data integration."""
        logger.info("Creating live risk dashboard with real-world data")
        
        center_lat = gdf.geometry.y.mean()
        center_lon = gdf.geometry.x.mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        folium.TileLayer('Stamen Terrain', name='Terrain').add_to(m)
        
        if 'roads' in real_world_datasets:
            self._add_roads_layer(m, real_world_datasets['roads'])
        
        risk_columns = [col for col in gdf.columns if 'expected_loss' in col]
        
        for scenario_idx, risk_col in enumerate(risk_columns):
            scenario_name = risk_col.replace('expected_loss_', '').upper()
            
            feature_group = folium.FeatureGroup(name=f'Risk - {scenario_name}')
            
            for idx, row in gdf.iterrows():
                risk_value = row[risk_col]
                
                if risk_value > 0:
                    color = self._get_risk_color(risk_value, gdf[risk_col].max())
                    
                    popup_text = f"""
                    <div style="font-family: Arial, sans-serif;">
                    <h4>Asset Risk Profile</h4>
                    <b>Asset ID:</b> {row.get('asset_id', idx)}<br>
                    <b>Scenario:</b> {scenario_name}<br>
                    <b>Expected Loss:</b> ${risk_value:,.0f}<br>
                    <b>Asset Type:</b> {row.get('type', 'Unknown')}<br>
                    <b>Population Density:</b> {row.get('population_density', 0):.1f}<br>
                    <b>Nightlight Intensity:</b> {row.get('nightlight_intensity', 0):.1f}<br>
                    <b>Urban Index:</b> {row.get('urban_index', 0):.2f}<br>
                    <b>Distance to Road:</b> {row.get('distance_to_road', 0):.0f}m<br>
                    <b>Exposure Multiplier:</b> {row.get('exposure_multiplier', 1):.2f}
                    </div>
                    """
                    
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=max(5, min(20, risk_value / gdf[risk_col].max() * 20)),
                        popup=folium.Popup(popup_text, max_width=350),
                        color=color,
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=2
                    ).add_to(feature_group)
            
            feature_group.add_to(m)
        
        folium.LayerControl().add_to(m)
        
        legend_html = self._create_enhanced_legend()
        m.get_root().html.add_child(folium.Element(legend_html))
        
        m.save(output_path)
        logger.info(f"Live dashboard saved to {output_path}")
        
        return output_path
    
    def _create_enhanced_legend(self) -> str:
        """Create enhanced legend for live dashboard."""
        return """
        <div style='position: fixed; 
                    bottom: 50px; left: 50px; width: 300px; height: 200px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px'>
        <h4>Climate Risk Dashboard</h4>
        <p><b>Risk Levels:</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> Low Risk</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Medium Risk</p>
        <p><i class="fa fa-circle" style="color:orange"></i> High Risk</p>
        <p><i class="fa fa-circle" style="color:red"></i> Critical Risk</p>
        <p><b>Data Sources:</b> WorldPop, VIIRS, OSM, Copernicus</p>
        </div>
        """
    
    def _create_map_legend(self) -> str:
        """Create map legend."""
        return """
        <div style='position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px'>
        <h4>Risk Map Legend</h4>
        <p><i class="fa fa-circle" style="color:green"></i> Low Risk</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Medium Risk</p>
        <p><i class="fa fa-circle" style="color:orange"></i> High Risk</p>
        <p><i class="fa fa-circle" style="color:red"></i> Critical Risk</p>
        </div>
        """

        <p><i class="fa fa-circle" style="color:orange"></i> Medium</p>
        <p><i class="fa fa-circle" style="color:red"></i> High</p>
        </div>
        '''
        
        folium_map.get_root().html.add_child(folium.Element(legend_html))
