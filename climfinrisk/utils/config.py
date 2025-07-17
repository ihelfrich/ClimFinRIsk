"""
Configuration management for the climate risk modeling platform.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration settings for the climate risk modeling platform.
    
    Supports YAML and JSON configuration files with environment-specific overrides.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = {}
        self.config_path = config_path
        
        self._load_default_config()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yml', '.yaml']:
                    user_config = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    user_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_file.suffix}")
            
            self._deep_merge(self.config, user_config)
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'data.cache_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, output_path: str):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration file
        """
        output_file = Path(output_path)
        
        try:
            with open(output_file, 'w') as f:
                if output_file.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif output_file.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported output format: {output_file.suffix}")
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _load_default_config(self):
        """Load default configuration settings."""
        self.config = {
            'data': {
                'cache_dir': './data_cache',
                'max_cache_size_gb': 10,
                'auto_download': True,
                'synthetic_fallback': True
            },
            'modeling': {
                'pca': {
                    'n_components': 0.95,
                    'standardize': True,
                    'random_state': 42
                },
                'tensor_svd': {
                    'method': 'tucker',
                    'max_rank': 10,
                    'convergence_tol': 1e-6
                },
                'risk_estimation': {
                    'time_horizon': 30,
                    'confidence_levels': [0.95, 0.99],
                    'monte_carlo_iterations': 1000
                }
            },
            'geospatial': {
                'default_crs': 'EPSG:4326',
                'buffer_distance': 0.1,
                'clustering': {
                    'n_clusters': 5,
                    'method': 'kmeans'
                }
            },
            'api': {
                'cds': {
                    'timeout': 300,
                    'retry_attempts': 3
                },
                'noaa': {
                    'timeout': 120,
                    'retry_attempts': 2
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': None
            },
            'output': {
                'default_dir': './outputs',
                'map_format': 'html',
                'report_format': 'markdown'
            }
        }
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """
        Deep merge two dictionaries.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate_config(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            required_sections = ['data', 'modeling', 'geospatial']
            for section in required_sections:
                if section not in self.config:
                    logger.error(f"Missing required configuration section: {section}")
                    return False
            
            cache_dir = self.get('data.cache_dir')
            if cache_dir:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
            n_components = self.get('modeling.pca.n_components')
            if isinstance(n_components, float) and not (0 < n_components <= 1):
                logger.error("PCA n_components must be between 0 and 1 when specified as float")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_scenario_config(self, scenario: str) -> Dict[str, Any]:
        """
        Get configuration for a specific climate scenario.
        
        Args:
            scenario: Climate scenario name (e.g., 'rcp85')
            
        Returns:
            Scenario-specific configuration
        """
        scenario_configs = {
            'rcp26': {
                'temperature_increase': 1.5,
                'precipitation_change': 0.05,
                'extreme_event_multiplier': 1.1
            },
            'rcp45': {
                'temperature_increase': 2.5,
                'precipitation_change': 0.10,
                'extreme_event_multiplier': 1.3
            },
            'rcp85': {
                'temperature_increase': 4.5,
                'precipitation_change': 0.20,
                'extreme_event_multiplier': 1.8
            }
        }
        
        return scenario_configs.get(scenario, scenario_configs['rcp45'])
