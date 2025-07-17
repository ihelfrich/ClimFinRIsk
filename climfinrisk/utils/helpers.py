"""
Helper functions and utilities for the climate risk modeling platform.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
        
    Returns:
        Distance in kilometers
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    r = 6371
    
    return c * r


def find_nearest_grid_point(
    target_lat: float,
    target_lon: float,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray
) -> Tuple[int, int]:
    """
    Find the nearest grid point to a target location.
    
    Args:
        target_lat, target_lon: Target coordinates
        grid_lats, grid_lons: Grid coordinate arrays
        
    Returns:
        Tuple of (lat_index, lon_index)
    """
    lat_distances = np.abs(grid_lats - target_lat)
    lon_distances = np.abs(grid_lons - target_lon)
    
    lat_idx = np.argmin(lat_distances)
    lon_idx = np.argmin(lon_distances)
    
    return lat_idx, lon_idx


def interpolate_to_asset_locations(
    climate_data: xr.Dataset,
    asset_locations: pd.DataFrame,
    method: str = 'linear'
) -> xr.Dataset:
    """
    Interpolate climate data to asset locations.
    
    Args:
        climate_data: Climate dataset with spatial grid
        asset_locations: DataFrame with 'lat' and 'lon' columns
        method: Interpolation method ('linear', 'nearest', 'cubic')
        
    Returns:
        Climate data interpolated to asset locations
    """
    try:
        asset_coords = {
            'asset_id': asset_locations.index.tolist(),
            'lat': ('asset_id', asset_locations['lat'].values),
            'lon': ('asset_id', asset_locations['lon'].values)
        }
        
        interpolated = climate_data.interp(
            lat=asset_coords['lat'][1],
            lon=asset_coords['lon'][1],
            method=method
        )
        
        interpolated = interpolated.assign_coords(asset_id=asset_coords['asset_id'])
        
        return interpolated
        
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        raise


def calculate_portfolio_statistics(
    risk_estimates: pd.DataFrame,
    groupby_column: str = None
) -> Dict[str, Any]:
    """
    Calculate portfolio-level risk statistics.
    
    Args:
        risk_estimates: DataFrame with risk estimates
        groupby_column: Column to group by (e.g., 'asset_type', 'region')
        
    Returns:
        Dictionary with portfolio statistics
    """
    stats = {}
    
    risk_columns = [col for col in risk_estimates.columns if 'expected_loss' in col]
    
    for col in risk_columns:
        scenario = col.replace('expected_loss_', '')
        
        stats[f'total_risk_{scenario}'] = risk_estimates[col].sum()
        stats[f'mean_risk_{scenario}'] = risk_estimates[col].mean()
        stats[f'median_risk_{scenario}'] = risk_estimates[col].median()
        stats[f'std_risk_{scenario}'] = risk_estimates[col].std()
        stats[f'max_risk_{scenario}'] = risk_estimates[col].max()
        stats[f'min_risk_{scenario}'] = risk_estimates[col].min()
    
    if groupby_column and groupby_column in risk_estimates.columns:
        group_stats = {}
        
        for group_name, group_data in risk_estimates.groupby(groupby_column):
            group_stats[group_name] = {}
            
            for col in risk_columns:
                scenario = col.replace('expected_loss_', '')
                group_stats[group_name][f'total_risk_{scenario}'] = group_data[col].sum()
                group_stats[group_name][f'mean_risk_{scenario}'] = group_data[col].mean()
                group_stats[group_name][f'count'] = len(group_data)
        
        stats['group_statistics'] = group_stats
    
    return stats


def create_risk_summary_table(
    risk_estimates: pd.DataFrame,
    scenarios: List[str] = None
) -> pd.DataFrame:
    """
    Create a summary table of risk estimates.
    
    Args:
        risk_estimates: DataFrame with risk estimates
        scenarios: List of scenarios to include
        
    Returns:
        Summary table DataFrame
    """
    if scenarios is None:
        scenarios = ['rcp26', 'rcp45', 'rcp85']
    
    summary_data = []
    
    for scenario in scenarios:
        expected_loss_col = f'expected_loss_{scenario}'
        var_95_col = f'var_95_{scenario}'
        var_99_col = f'var_99_{scenario}'
        
        if expected_loss_col in risk_estimates.columns:
            row = {
                'Scenario': scenario.upper(),
                'Total Expected Loss': risk_estimates[expected_loss_col].sum(),
                'Mean Expected Loss': risk_estimates[expected_loss_col].mean(),
                'Assets at Risk': (risk_estimates[expected_loss_col] > 0).sum(),
                'Max Single Asset Loss': risk_estimates[expected_loss_col].max()
            }
            
            if var_95_col in risk_estimates.columns:
                row['Portfolio VaR 95%'] = risk_estimates[var_95_col].sum()
            
            if var_99_col in risk_estimates.columns:
                row['Portfolio VaR 99%'] = risk_estimates[var_99_col].sum()
            
            summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def export_results_to_excel(
    risk_estimates: pd.DataFrame,
    spatial_analysis: Dict[str, Any],
    output_path: str
):
    """
    Export analysis results to Excel file with multiple sheets.
    
    Args:
        risk_estimates: Risk estimates DataFrame
        spatial_analysis: Spatial analysis results
        output_path: Path to save Excel file
    """
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            risk_estimates.to_excel(writer, sheet_name='Risk_Estimates', index=False)
            
            summary_table = create_risk_summary_table(risk_estimates)
            summary_table.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            portfolio_stats = calculate_portfolio_statistics(risk_estimates)
            stats_df = pd.DataFrame.from_dict(portfolio_stats, orient='index', columns=['Value'])
            stats_df.to_excel(writer, sheet_name='Portfolio_Statistics')
            
            if 'spatial_statistics' in spatial_analysis:
                spatial_stats = spatial_analysis['spatial_statistics']
                if 'risk_by_region' in spatial_stats:
                    regional_df = pd.DataFrame.from_dict(
                        spatial_stats['risk_by_region'], 
                        orient='index'
                    )
                    regional_df.to_excel(writer, sheet_name='Regional_Analysis')
        
        logger.info(f"Results exported to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to export results to Excel: {e}")
        raise


def format_currency(value: float, currency: str = 'USD') -> str:
    """
    Format a numeric value as currency.
    
    Args:
        value: Numeric value to format
        currency: Currency code (default: USD)
        
    Returns:
        Formatted currency string
    """
    if pd.isna(value):
        return 'N/A'
    
    if currency == 'USD':
        if abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.2f}K"
        else:
            return f"${value:.2f}"
    else:
        return f"{value:,.2f} {currency}"


def calculate_risk_metrics(
    loss_distribution: np.ndarray,
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict[str, float]:
    """
    Calculate comprehensive risk metrics from loss distribution.
    
    Args:
        loss_distribution: Array of simulated losses
        confidence_levels: Confidence levels for VaR calculation
        
    Returns:
        Dictionary with risk metrics
    """
    metrics = {}
    
    metrics['mean_loss'] = np.mean(loss_distribution)
    metrics['median_loss'] = np.median(loss_distribution)
    metrics['std_loss'] = np.std(loss_distribution)
    metrics['max_loss'] = np.max(loss_distribution)
    metrics['min_loss'] = np.min(loss_distribution)
    
    for confidence in confidence_levels:
        percentile = confidence * 100
        var_value = np.percentile(loss_distribution, percentile)
        metrics[f'var_{int(percentile)}'] = var_value
    
    for confidence in confidence_levels:
        percentile = confidence * 100
        var_value = np.percentile(loss_distribution, percentile)
        tail_losses = loss_distribution[loss_distribution >= var_value]
        if len(tail_losses) > 0:
            cvar_value = np.mean(tail_losses)
        else:
            cvar_value = var_value
        metrics[f'cvar_{int(percentile)}'] = cvar_value
    
    metrics['probability_of_loss'] = np.mean(loss_distribution > 0)
    
    from scipy import stats
    metrics['skewness'] = stats.skew(loss_distribution)
    metrics['kurtosis'] = stats.kurtosis(loss_distribution)
    
    return metrics


def create_scenario_comparison(
    risk_estimates: pd.DataFrame,
    scenarios: List[str] = None
) -> pd.DataFrame:
    """
    Create a comparison table across climate scenarios.
    
    Args:
        risk_estimates: Risk estimates DataFrame
        scenarios: List of scenarios to compare
        
    Returns:
        Scenario comparison DataFrame
    """
    if scenarios is None:
        scenarios = ['rcp26', 'rcp45', 'rcp85']
    
    comparison_data = []
    
    for scenario in scenarios:
        expected_loss_col = f'expected_loss_{scenario}'
        
        if expected_loss_col in risk_estimates.columns:
            scenario_data = {
                'Scenario': scenario.upper(),
                'Total Portfolio Loss': risk_estimates[expected_loss_col].sum(),
                'Average Asset Loss': risk_estimates[expected_loss_col].mean(),
                'Assets at Risk (%)': (risk_estimates[expected_loss_col] > 0).mean() * 100,
                'Maximum Single Loss': risk_estimates[expected_loss_col].max(),
                'Loss Concentration (Top 10%)': (
                    risk_estimates[expected_loss_col].nlargest(
                        int(0.1 * len(risk_estimates))
                    ).sum() / risk_estimates[expected_loss_col].sum() * 100
                )
            }
            comparison_data.append(scenario_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if len(comparison_df) > 1:
        baseline_idx = comparison_df[comparison_df['Scenario'] == 'RCP26'].index
        if len(baseline_idx) > 0:
            baseline = comparison_df.loc[baseline_idx[0]]
            
            for idx, row in comparison_df.iterrows():
                if row['Scenario'] != 'RCP26':
                    for col in ['Total Portfolio Loss', 'Average Asset Loss', 'Maximum Single Loss']:
                        if baseline[col] > 0:
                            change = ((row[col] - baseline[col]) / baseline[col]) * 100
                            comparison_df.loc[idx, f'{col} Change (%)'] = change
    
    return comparison_df
