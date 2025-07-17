# Physical Climate Risk Modeling Platform

A comprehensive platform for quantifying, mapping, and analyzing physical climate risk for financial asset portfolios using advanced statistical and geospatial techniques.

## Overview

This platform implements Dr. Ian Helfrich's modeling philosophy, emphasizing:
- Advanced dimensionality reduction (PCA, tensor SVD)
- Geospatial analytics at asset-level granularity
- Robust asset vulnerability estimation at scale
- Integration with open-source climate risk and geospatial data tools

## Key Features

### 1. Data Integration
- Ingest physical hazard and climate datasets (storm surge, flood, drought, cyclone)
- Support for Copernicus (CDSAPI), NOAA (pyncei), and other open data sources
- Compatible with geospatial information (GeoPandas, rasterio)
- Asset exposure and vulnerability data integration

### 2. Core Modeling Pipeline
- **Dimensionality Reduction & Signal Extraction**
  - Principal Component Analysis (PCA) on hazard and impact variable matrices
  - Tensor SVD for multidimensional data (hazard–location–time)
  - Latent structure extraction and noise reduction
- **Physical Risk Estimation**
  - Bayesian loss models and machine learning approaches
  - Loss vulnerability curves and damage estimation functions
  - Customizable for asset types and geographies
- **Scenario & Stress Testing**
  - Monte Carlo and agent-based scenarios
  - Multiple climate futures using downscaled projections
  - Chronic and acute risk modeling
  - TCFD and IFRS S2 regulatory alignment

### 3. Geospatial Analytics
- Asset-level spatial analysis and mapping
- Fast site selection and filtering capabilities
- Spatial network modeling for exposure propagation

## Installation

```bash
# Clone the repository
git clone https://github.com/ihelfrich/ClimFinRIsk.git
cd ClimFinRIsk

# Install dependencies
pip install -r requirements.txt

# Or using poetry
poetry install
```

## Quick Start

```python
from climfinrisk import ClimateRiskModeler
from climfinrisk.data import DataIngestion
from climfinrisk.modeling import DimensionalityReduction, RiskEstimation
from climfinrisk.geospatial import SpatialAnalyzer

# Initialize the modeler
modeler = ClimateRiskModeler()

# Load and process climate data
data_ingestion = DataIngestion()
climate_data = data_ingestion.load_copernicus_data(region="global")

# Apply dimensionality reduction
dim_reducer = DimensionalityReduction()
reduced_data = dim_reducer.apply_pca(climate_data)

# Estimate physical risks
risk_estimator = RiskEstimation()
risk_estimates = risk_estimator.calculate_portfolio_risk(reduced_data, assets)

# Generate spatial analysis
spatial_analyzer = SpatialAnalyzer()
risk_map = spatial_analyzer.create_risk_map(risk_estimates)
```

## Documentation

- [Getting Started Guide](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Modeling Methodology](docs/methodology.md)
- [Data Sources](docs/data_sources.md)
- [Regulatory Reporting](docs/regulatory_reporting.md)

## Examples

See the `notebooks/` directory for comprehensive examples:
- `01_data_ingestion_demo.ipynb` - Data retrieval and preprocessing
- `02_dimensionality_reduction_demo.ipynb` - PCA and tensor SVD examples
- `03_risk_estimation_demo.ipynb` - Physical risk modeling
- `04_geospatial_analysis_demo.ipynb` - Spatial analysis and mapping
- `05_end_to_end_workflow.ipynb` - Complete workflow demonstration

## Requirements

- Python 3.8+
- CLIMADA
- OS-Climate physrisk
- xarray
- rasterio
- GeoPandas
- pyproj
- scikit-learn
- tensorly
- numpy
- pandas

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this platform in your research, please cite:

```
Helfrich, I. (2025). Physical Climate Risk Modeling Platform for Financial Assets. 
GitHub repository: https://github.com/ihelfrich/ClimFinRIsk
```

## Contact

**Principal Investigator:** Dr. Ian Helfrich  
**Email:** ian@gatech.edu  
**Institution:** Georgia Institute of Technology
