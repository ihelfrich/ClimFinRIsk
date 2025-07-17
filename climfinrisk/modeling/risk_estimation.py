"""
Risk estimation module for calculating physical climate risks to financial assets.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

logger = logging.getLogger(__name__)


class RiskEstimation:
    """
    Physical climate risk estimation for financial asset portfolios.
    
    Implements:
    - Bayesian loss models
    - Machine learning risk prediction
    - Vulnerability curve integration
    - Monte Carlo scenario analysis
    - Regulatory reporting (TCFD, IFRS S2)
    """
    
    def __init__(self, config=None):
        """
        Initialize risk estimation with configuration.
        
        Args:
            config: Configuration object with model parameters
        """
        self.config = config or {}
        self.fitted_models = {}
        self.vulnerability_curves = {}
        
    def estimate_portfolio_risk(
        self,
        climate_data: xr.Dataset,
        assets: pd.DataFrame,
        scenarios: List[str] = None,
        time_horizon: int = 30
    ) -> pd.DataFrame:
        """
        Estimate physical climate risks for an asset portfolio.
        
        Args:
            climate_data: Reduced climate data from dimensionality reduction
            assets: Asset portfolio data with locations and characteristics
            scenarios: Climate scenarios to analyze
            time_horizon: Time horizon for risk assessment (years)
            
        Returns:
            DataFrame with risk estimates for each asset
        """
        logger.info(f"Estimating portfolio risk for {len(assets)} assets")
        
        if scenarios is None:
            scenarios = ['rcp26', 'rcp45', 'rcp85']
        
        risk_results = assets.copy()
        
        for scenario in scenarios:
            risk_results[f'expected_loss_{scenario}'] = 0.0
            risk_results[f'var_95_{scenario}'] = 0.0
            risk_results[f'var_99_{scenario}'] = 0.0
        
        for idx, asset in assets.iterrows():
            try:
                asset_climate = self._extract_asset_climate_data(
                    climate_data, asset['lat'], asset['lon']
                )
                
                for scenario in scenarios:
                    scenario_risk = self._calculate_asset_risk(
                        asset_climate, asset, scenario, time_horizon
                    )
                    
                    risk_results.loc[idx, f'expected_loss_{scenario}'] = scenario_risk['expected_loss']
                    risk_results.loc[idx, f'var_95_{scenario}'] = scenario_risk['var_95']
                    risk_results.loc[idx, f'var_99_{scenario}'] = scenario_risk['var_99']
                
            except Exception as e:
                logger.warning(f"Failed to estimate risk for asset {asset.get('asset_id', idx)}: {e}")
                for scenario in scenarios:
                    risk_results.loc[idx, f'expected_loss_{scenario}'] = 0.0
                    risk_results.loc[idx, f'var_95_{scenario}'] = 0.0
                    risk_results.loc[idx, f'var_99_{scenario}'] = 0.0
        
        portfolio_stats = self._calculate_portfolio_statistics(risk_results, scenarios)
        
        logger.info("Portfolio risk estimation completed")
        return risk_results
    
    def train_risk_model(
        self,
        training_data: pd.DataFrame,
        target_column: str = 'historical_loss',
        model_type: str = 'bayesian'
    ) -> Dict[str, Any]:
        """
        Train a risk prediction model using historical data.
        
        Args:
            training_data: Historical data with features and losses
            target_column: Column containing historical loss data
            model_type: Type of model ('bayesian', 'random_forest', 'ensemble')
            
        Returns:
            Dictionary containing trained model and performance metrics
        """
        logger.info(f"Training {model_type} risk model")
        
        feature_columns = [col for col in training_data.columns 
                          if col not in [target_column, 'asset_id', 'lat', 'lon']]
        
        X = training_data[feature_columns]
        y = training_data[target_column]
        
        X = X.fillna(X.mean())
        y = y.fillna(0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if model_type == 'bayesian':
            model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_key = f"{model_type}_risk_model"
        self.fitted_models[model_key] = {
            'model': model,
            'feature_columns': feature_columns,
            'performance': {
                'mse': mse,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
        }
        
        logger.info(f"Model trained - R²: {r2:.3f}, RMSE: {np.sqrt(mse):.3f}")
        
        return self.fitted_models[model_key]
    
    def run_monte_carlo_simulation(
        self,
        asset: pd.Series,
        climate_scenarios: Dict[str, xr.Dataset],
        n_simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for asset risk assessment.
        
        Args:
            asset: Single asset data
            climate_scenarios: Dictionary of climate scenario datasets
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with simulation results and statistics
        """
        logger.info(f"Running Monte Carlo simulation with {n_simulations} iterations")
        
        simulation_results = {}
        
        for scenario_name, scenario_data in climate_scenarios.items():
            losses = []
            
            for i in range(n_simulations):
                climate_sample = self._sample_climate_realization(scenario_data)
                
                loss = self._calculate_single_loss(asset, climate_sample)
                losses.append(loss)
            
            losses = np.array(losses)
            simulation_results[scenario_name] = {
                'losses': losses,
                'mean_loss': np.mean(losses),
                'std_loss': np.std(losses),
                'var_95': np.percentile(losses, 95),
                'var_99': np.percentile(losses, 99),
                'max_loss': np.max(losses),
                'probability_of_loss': np.mean(losses > 0)
            }
        
        return simulation_results
    
    def calculate_value_at_risk(
        self,
        loss_distribution: np.ndarray,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) for given confidence levels.
        
        Args:
            loss_distribution: Array of simulated losses
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
            
        Returns:
            Dictionary with VaR values for each confidence level
        """
        var_results = {}
        
        for confidence in confidence_levels:
            percentile = confidence * 100
            var_value = np.percentile(loss_distribution, percentile)
            var_results[f'var_{int(percentile)}'] = var_value
        
        return var_results
    
    def generate_tcfd_report(
        self,
        risk_estimates: pd.DataFrame,
        output_path: str = None
    ) -> str:
        """
        Generate TCFD-compliant climate risk report.
        
        Args:
            risk_estimates: Portfolio risk estimates
            output_path: Path to save the report
            
        Returns:
            Path to generated report
        """
        logger.info("Generating TCFD climate risk report")
        
        report_content = self._create_tcfd_report_content(risk_estimates)
        
        if output_path is None:
            output_path = "tcfd_climate_risk_report.md"
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"TCFD report saved to {output_path}")
        return output_path
    
    def generate_ifrs_s2_report(
        self,
        risk_estimates: pd.DataFrame,
        output_path: str = None
    ) -> str:
        """
        Generate IFRS S2-compliant climate disclosure report.
        
        Args:
            risk_estimates: Portfolio risk estimates
            output_path: Path to save the report
            
        Returns:
            Path to generated report
        """
        logger.info("Generating IFRS S2 climate disclosure report")
        
        report_content = self._create_ifrs_s2_report_content(risk_estimates)
        
        if output_path is None:
            output_path = "ifrs_s2_climate_disclosure.md"
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"IFRS S2 report saved to {output_path}")
        return output_path
    
    def _extract_asset_climate_data(
        self,
        climate_data: xr.Dataset,
        lat: float,
        lon: float
    ) -> xr.Dataset:
        """Extract climate data for a specific asset location."""
        try:
            if 'lat' in climate_data.coords and 'lon' in climate_data.coords:
                asset_data = climate_data.sel(lat=lat, lon=lon, method='nearest')
            else:
                spatial_dims = [dim for dim in climate_data.dims if dim not in ['time', 'component', 'factor']]
                if len(spatial_dims) >= 2:
                    asset_data = climate_data.isel({spatial_dims[0]: 0, spatial_dims[1]: 0})
                else:
                    asset_data = climate_data
            
            return asset_data
            
        except Exception as e:
            logger.warning(f"Failed to extract climate data for location ({lat}, {lon}): {e}")
            return climate_data.isel({dim: 0 for dim in climate_data.dims if dim != 'time'})
    
    def _calculate_asset_risk(
        self,
        climate_data: xr.Dataset,
        asset: pd.Series,
        scenario: str,
        time_horizon: int
    ) -> Dict[str, float]:
        """Calculate risk metrics for a single asset."""
        asset_value = asset.get('value', 1.0)
        asset_type = asset.get('type', 'commercial')
        
        hazard_exposure = self._calculate_hazard_exposure(climate_data, scenario)
        
        vulnerability = self._get_vulnerability_curve(asset_type)
        
        expected_loss = self._calculate_expected_loss(
            hazard_exposure, vulnerability, asset_value, time_horizon
        )
        
        loss_std = expected_loss * 0.5  # Assume 50% coefficient of variation
        var_95 = expected_loss + 1.645 * loss_std  # 95% VaR
        var_99 = expected_loss + 2.326 * loss_std  # 99% VaR
        
        return {
            'expected_loss': expected_loss,
            'var_95': var_95,
            'var_99': var_99
        }
    
    def _calculate_hazard_exposure(
        self,
        climate_data: xr.Dataset,
        scenario: str
    ) -> float:
        """Calculate aggregate hazard exposure from climate data."""
        total_exposure = 0.0
        
        for var_name, var_data in climate_data.data_vars.items():
            try:
                if hasattr(var_data, 'values'):
                    values = var_data.values
                    if values.ndim > 0:
                        mean_exposure = float(np.mean(values))
                    else:
                        mean_exposure = float(values)
                else:
                    mean_exposure = float(var_data)
                
                scenario_multipliers = {
                    'rcp26': 1.0,
                    'rcp45': 1.3,
                    'rcp85': 1.8
                }
                multiplier = scenario_multipliers.get(scenario, 1.0)
                
                total_exposure += abs(mean_exposure) * multiplier
                
            except Exception as e:
                logger.warning(f"Failed to process variable {var_name}: {e}")
                continue
        
        return total_exposure
    
    def _get_vulnerability_curve(self, asset_type: str) -> Dict[str, float]:
        """Get vulnerability curve parameters for asset type."""
        vulnerability_curves = {
            'residential': {'threshold': 0.05, 'slope': 0.8, 'max_damage': 0.9},
            'commercial': {'threshold': 0.08, 'slope': 0.7, 'max_damage': 0.85},
            'industrial': {'threshold': 0.1, 'slope': 0.6, 'max_damage': 0.95},
            'infrastructure': {'threshold': 0.02, 'slope': 0.9, 'max_damage': 0.8}
        }
        
        return vulnerability_curves.get(asset_type, vulnerability_curves['commercial'])
    
    def _calculate_expected_loss(
        self,
        hazard_exposure: float,
        vulnerability: Dict[str, float],
        asset_value: float,
        time_horizon: int
    ) -> float:
        """Calculate expected loss using vulnerability curve."""
        if hazard_exposure <= vulnerability['threshold']:
            damage_ratio = 0.0
        else:
            damage_ratio = min(
                vulnerability['max_damage'],
                vulnerability['slope'] * (hazard_exposure - vulnerability['threshold'])
            )
        
        annual_loss = damage_ratio * asset_value * 0.01  # 1% annual probability
        
        expected_loss = annual_loss * time_horizon
        
        return expected_loss
    
    def _sample_climate_realization(self, scenario_data: xr.Dataset) -> xr.Dataset:
        """Sample a random climate realization from scenario data."""
        sampled_data = scenario_data.copy()
        
        for var_name in sampled_data.data_vars:
            noise_scale = 0.1  # 10% noise
            noise = np.random.normal(0, noise_scale, sampled_data[var_name].shape)
            sampled_data[var_name] = sampled_data[var_name] + noise
        
        return sampled_data
    
    def _calculate_single_loss(self, asset: pd.Series, climate_sample: xr.Dataset) -> float:
        """Calculate loss for a single climate realization."""
        hazard_exposure = self._calculate_hazard_exposure(climate_sample, 'sample')
        
        vulnerability = self._get_vulnerability_curve(asset.get('type', 'commercial'))
        loss = self._calculate_expected_loss(
            hazard_exposure, vulnerability, asset.get('value', 1.0), 1
        )
        
        return loss
    
    def _calculate_portfolio_statistics(
        self,
        risk_results: pd.DataFrame,
        scenarios: List[str]
    ) -> Dict[str, Any]:
        """Calculate portfolio-level risk statistics."""
        portfolio_stats = {}
        
        for scenario in scenarios:
            expected_loss_col = f'expected_loss_{scenario}'
            var_95_col = f'var_95_{scenario}'
            
            if expected_loss_col in risk_results.columns:
                portfolio_stats[scenario] = {
                    'total_expected_loss': risk_results[expected_loss_col].sum(),
                    'mean_asset_loss': risk_results[expected_loss_col].mean(),
                    'max_asset_loss': risk_results[expected_loss_col].max(),
                    'portfolio_var_95': risk_results[var_95_col].sum(),
                    'assets_at_risk': (risk_results[expected_loss_col] > 0).sum()
                }
        
        return portfolio_stats
    
    def _create_tcfd_report_content(self, risk_estimates: pd.DataFrame) -> str:
        """Create TCFD report content."""
        report = """# TCFD Climate Risk Assessment Report


This report presents the climate-related financial risks for the asset portfolio in accordance with the Task Force on Climate-related Financial Disclosures (TCFD) recommendations.


The organization has established climate risk governance structures to oversee the identification, assessment, and management of climate-related risks and opportunities.



"""
        
        scenarios = ['rcp26', 'rcp45', 'rcp85']
        for scenario in scenarios:
            expected_loss_col = f'expected_loss_{scenario}'
            if expected_loss_col in risk_estimates.columns:
                total_loss = risk_estimates[expected_loss_col].sum()
                assets_at_risk = (risk_estimates[expected_loss_col] > 0).sum()
                
                report += f"""
- Total Expected Loss: ${total_loss:,.0f}
- Assets at Risk: {assets_at_risk} out of {len(risk_estimates)}
- Average Loss per Asset: ${total_loss/len(risk_estimates):,.0f}
"""
        
        report += """

The organization employs a comprehensive approach to identifying, assessing, and managing climate-related risks, including:

1. Physical risk assessment using advanced climate models
2. Scenario analysis across multiple climate pathways
3. Portfolio-level risk aggregation and monitoring
4. Integration with existing risk management frameworks


Key climate risk metrics monitored include:
- Expected annual losses by climate scenario
- Value at Risk (VaR) at 95% and 99% confidence levels
- Geographic and sectoral risk concentrations
- Climate risk exposure trends over time

---
*Report generated using ClimFinRisk platform*
"""
        
        return report
    
    def _create_ifrs_s2_report_content(self, risk_estimates: pd.DataFrame) -> str:
        """Create IFRS S2 report content."""
        report = """# IFRS S2 Climate-Related Disclosures


The board of directors oversees climate-related risks and opportunities, including review of climate risk assessments and strategic responses.

Management is responsible for implementing climate risk management processes and monitoring climate-related metrics.



The organization faces physical climate risks including:
- Acute risks from extreme weather events
- Chronic risks from long-term climate changes
- Geographic concentration risks

"""
        
        scenarios = ['rcp26', 'rcp45', 'rcp85']
        for scenario in scenarios:
            expected_loss_col = f'expected_loss_{scenario}'
            if expected_loss_col in risk_estimates.columns:
                total_loss = risk_estimates[expected_loss_col].sum()
                max_loss = risk_estimates[expected_loss_col].max()
                
                report += f"""
- Portfolio Expected Loss: ${total_loss:,.0f}
- Maximum Single Asset Loss: ${max_loss:,.0f}
- Risk Distribution: Varies by geography and asset type
"""
        
        report += """

- Systematic identification of climate-related risks
- Quantitative risk modeling using climate science
- Regular updates based on latest climate projections

- Integration with enterprise risk management
- Climate risk considerations in investment decisions
- Ongoing monitoring and reporting


The organization tracks the following climate-related metrics:

1. **Expected Losses**: Projected financial losses under different climate scenarios
2. **Value at Risk**: Potential losses at specified confidence levels
3. **Risk Concentration**: Geographic and sectoral risk distributions
4. **Adaptation Measures**: Investments in climate resilience

- Reduce climate risk exposure by X% by 2030
- Implement adaptation measures for high-risk assets
- Maintain climate risk within acceptable tolerance levels

---
*Disclosure prepared in accordance with IFRS S2 Climate-related Disclosures*
"""
        
        return report
