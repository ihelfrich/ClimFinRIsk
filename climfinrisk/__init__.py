"""
Physical Climate Risk Modeling Platform for Financial Assets

A comprehensive platform for quantifying, mapping, and analyzing physical climate risk 
for financial asset portfolios using advanced statistical and geospatial techniques.

Principal Investigator: Dr. Ian Helfrich
Institution: Georgia Institute of Technology
"""

__version__ = "0.1.0"
__author__ = "Dr. Ian Helfrich"
__email__ = "ian@gatech.edu"

from .core import ClimateRiskModeler
from .data import DataIngestion
from .modeling import DimensionalityReduction, RiskEstimation
from .geospatial import SpatialAnalyzer

__all__ = [
    "ClimateRiskModeler",
    "DataIngestion", 
    "DimensionalityReduction",
    "RiskEstimation",
    "SpatialAnalyzer"
]
