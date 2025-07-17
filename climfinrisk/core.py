"""
Core Climate Risk Modeler class that orchestrates the entire modeling pipeline.
"""

import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import xarray as xr

from .data import DataIngestion
from .utils import DataValidator
from .modeling import DimensionalityReduction, RiskEstimation
from .geospatial import SpatialAnalyzer
from .utils import ConfigManager, Logger

logger = logging.getLogger(__name__)


class ClimateRiskModeler:
    """
    Main orchestrator class for the Physical Climate Risk Modeling Platform.
    
    This class coordinates data ingestion, dimensionality reduction, risk estimation,
    and geospatial analysis to provide comprehensive climate risk assessments for
    financial asset portfolios.
    
    Attributes:
        config (ConfigManager): Configuration management
        data_ingestion (DataIngestion): Data retrieval and preprocessing
        dim_reducer (DimensionalityReduction): PCA and tensor SVD operations
        risk_estimator (RiskEstimation): Physical risk modeling
        spatial_analyzer (SpatialAnalyzer): Geospatial analysis and mapping
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Climate Risk Modeler.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigManager(config_path)
        self.logger = Logger(self.config.get('logging', {}))
        
        self.data_ingestion = DataIngestion(self.config)
        self.dim_reducer = DimensionalityReduction(self.config)
        self.risk_estimator = RiskEstimation(self.config)
        self.spatial_analyzer = SpatialAnalyzer(self.config)
        
        logger.info("Climate Risk Modeler initialized successfully")
    
    def run_full_analysis(
        self,
        assets: pd.DataFrame,
        hazard_types: List[str] = None,
        scenarios: List[str] = None,
        output_dir: str = "outputs"
    ) -> Dict[str, Any]:
        """
        Run the complete climate risk analysis workflow.
        
        Args:
            assets: DataFrame containing asset information (lat, lon, type, value, etc.)
            hazard_types: List of hazard types to analyze (e.g., ['flood', 'cyclone'])
            scenarios: List of climate scenarios (e.g., ['rcp26', 'rcp85'])
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing analysis results and outputs
        """
        logger.info("Starting full climate risk analysis")
        
        if hazard_types is None:
            hazard_types = ['flood', 'cyclone', 'drought']
        if scenarios is None:
            scenarios = ['rcp26', 'rcp45', 'rcp85']
        
        results = {}
        
        try:
            logger.info("Step 1: Ingesting climate and hazard data")
            climate_data = self.data_ingestion.load_climate_data(
                hazard_types=hazard_types,
                scenarios=scenarios,
                bbox=self._get_asset_bbox(assets)
            )
            results['climate_data'] = climate_data
            
            logger.info("Step 2: Applying dimensionality reduction")
            reduced_data = self.dim_reducer.reduce_dimensions(
                climate_data,
                method='pca',
                n_components=0.95  # Retain 95% of variance
            )
            results['reduced_data'] = reduced_data
            
            logger.info("Step 3: Estimating physical risks")
            risk_estimates = self.risk_estimator.estimate_portfolio_risk(
                reduced_data,
                assets,
                scenarios=scenarios
            )
            results['risk_estimates'] = risk_estimates
            
            logger.info("Step 4: Generating spatial analysis")
            spatial_results = self.spatial_analyzer.analyze_spatial_risk(
                risk_estimates,
                assets,
                output_dir=output_dir
            )
            results['spatial_analysis'] = spatial_results
            
            logger.info("Step 5: Generating regulatory reports")
            reports = self._generate_reports(results, output_dir)
            results['reports'] = reports
            
            logger.info("Full climate risk analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in full analysis: {str(e)}")
            raise
    
    def run_scenario_analysis(
        self,
        assets: pd.DataFrame,
        scenarios: List[str],
        hazard_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run scenario-based stress testing analysis.
        
        Args:
            assets: Asset portfolio data
            scenarios: Climate scenarios to analyze
            hazard_types: Types of hazards to consider
            
        Returns:
            Scenario analysis results
        """
        logger.info(f"Running scenario analysis for {len(scenarios)} scenarios")
        
        scenario_results = {}
        
        for scenario in scenarios:
            logger.info(f"Analyzing scenario: {scenario}")
            
            scenario_data = self.data_ingestion.load_scenario_data(
                scenario=scenario,
                hazard_types=hazard_types,
                bbox=self._get_asset_bbox(assets)
            )
            
            reduced_data = self.dim_reducer.reduce_dimensions(scenario_data)
            
            risk_estimates = self.risk_estimator.estimate_portfolio_risk(
                reduced_data,
                assets,
                scenarios=[scenario]
            )
            
            scenario_results[scenario] = {
                'data': scenario_data,
                'reduced_data': reduced_data,
                'risk_estimates': risk_estimates
            }
        
        comparison = self._compare_scenarios(scenario_results)
        
        return {
            'scenario_results': scenario_results,
            'comparison': comparison
        }
    
    def _get_asset_bbox(self, assets: pd.DataFrame) -> Dict[str, float]:
        """Extract bounding box from asset coordinates."""
        return {
            'min_lat': assets['lat'].min(),
            'max_lat': assets['lat'].max(),
            'min_lon': assets['lon'].min(),
            'max_lon': assets['lon'].max()
        }
    
    def _generate_reports(self, results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """Generate regulatory and summary reports."""
        reports = {}
        
        tcfd_report = self.risk_estimator.generate_tcfd_report(
            results['risk_estimates'],
            output_path=f"{output_dir}/tcfd_report.pdf"
        )
        reports['tcfd'] = tcfd_report
        
        ifrs_report = self.risk_estimator.generate_ifrs_s2_report(
            results['risk_estimates'],
            output_path=f"{output_dir}/ifrs_s2_report.pdf"
        )
        reports['ifrs_s2'] = ifrs_report
        
        return reports
    
    def _compare_scenarios(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results across different climate scenarios."""
        comparison = {
            'scenario_summary': {},
            'risk_differences': {},
            'key_insights': []
        }
        
        for scenario, results in scenario_results.items():
            risk_estimates = results['risk_estimates']
            
            comparison['scenario_summary'][scenario] = {
                'total_expected_loss': risk_estimates['expected_loss'].sum(),
                'max_asset_loss': risk_estimates['expected_loss'].max(),
                'assets_at_risk': (risk_estimates['expected_loss'] > 0).sum()
            }
        
        return comparison
