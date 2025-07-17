"""
Logging utilities for the climate risk modeling platform.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any


class Logger:
    """
    Centralized logging configuration for the climate risk platform.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize logging configuration.
        
        Args:
            config: Logging configuration dictionary
        """
        self.config = config or {}
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        level = self.config.get('level', 'INFO')
        format_str = self.config.get('format', 
                                   '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = self.config.get('file')
        
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_str,
            handlers=self._get_handlers(log_file)
        )
        
        self._configure_library_loggers()
    
    def _get_handlers(self, log_file: Optional[str]) -> list:
        """Get logging handlers."""
        handlers = []
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        handlers.append(console_handler)
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)
            handlers.append(file_handler)
        
        return handlers
    
    def _configure_library_loggers(self):
        """Configure logging levels for external libraries."""
        library_loggers = {
            'urllib3': logging.WARNING,
            'requests': logging.WARNING,
            'matplotlib': logging.WARNING,
            'PIL': logging.WARNING,
            'fiona': logging.WARNING,
            'rasterio': logging.WARNING
        }
        
        for logger_name, level in library_loggers.items():
            logging.getLogger(logger_name).setLevel(level)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)
