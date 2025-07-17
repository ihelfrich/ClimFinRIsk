"""
Spatial analysis module for climate risk mapping and geospatial analytics.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point, Polygon
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
        
        risk_column = f'expected_loss_{scenario}'
        if risk_column not in risk_data.columns:
            risk_column = 'expected_loss_rcp85'  # Fallback
        
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
            
            for idx, row in risk_data.iterrows():
                if pd.isna(row[risk_column]):
                    continue
                
                radius = max(5, (row[risk_column] / max_risk) * 50)
                
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
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        risk_shares = risk_values / risk_values.sum()
        hhi = np.sum(risk_shares ** 2)
        
        return {
            'gini_coefficient': gini,
            'herfindahl_hirschman_index': hhi,
            'top_10_percent_share': risk_values.nlargest(int(0.1 * len(risk_values))).sum() / risk_values.sum()
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
                        radius=max(3, (row[risk_column] / gdf[risk_column].max()) * 20),
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
        <p><i class="fa fa-circle" style="color:orange"></i> Medium</p>
        <p><i class="fa fa-circle" style="color:red"></i> High</p>
        </div>
        '''
        
        folium_map.get_root().html.add_child(folium.Element(legend_html))
