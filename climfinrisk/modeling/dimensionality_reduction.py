"""
Dimensionality reduction module implementing PCA and tensor SVD techniques.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorly as tl
from tensorly.decomposition import tucker, parafac

logger = logging.getLogger(__name__)


class DimensionalityReduction:
    """
    Advanced dimensionality reduction for climate risk data.
    
    Implements:
    - Principal Component Analysis (PCA) for 2D data matrices
    - Tensor SVD (Tucker and PARAFAC decomposition) for multidimensional data
    - Signal extraction and noise reduction techniques
    """
    
    def __init__(self, config=None):
        """
        Initialize dimensionality reduction with configuration.
        
        Args:
            config: Configuration object with reduction parameters
        """
        self.config = config or {}
        self.fitted_models = {}
        self.scalers = {}
        
    def reduce_dimensions(
        self,
        data: xr.Dataset,
        method: str = 'pca',
        n_components: Union[int, float] = 0.95,
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        """
        Apply dimensionality reduction to climate data.
        
        Args:
            data: Input xarray Dataset
            method: Reduction method ('pca', 'tucker', 'parafac')
            n_components: Number of components or variance to retain
            variables: Specific variables to reduce (if None, use all)
            
        Returns:
            Reduced xarray Dataset
        """
        logger.info(f"Applying {method} dimensionality reduction")
        
        if variables is None:
            variables = list(data.data_vars.keys())
        
        reduced_data = data.copy()
        
        for var in variables:
            logger.info(f"Reducing dimensions for variable: {var}")
            
            var_data = data[var]
            
            if method == 'pca':
                reduced_var = self._apply_pca(var_data, n_components, var)
            elif method == 'tucker':
                reduced_var = self._apply_tucker_decomposition(var_data, n_components, var)
            elif method == 'parafac':
                reduced_var = self._apply_parafac_decomposition(var_data, n_components, var)
            else:
                raise ValueError(f"Unknown reduction method: {method}")
            
            reduced_data[var] = reduced_var
        
        reduced_data.attrs.update({
            'reduction_method': method,
            'n_components': n_components,
            'reduced_variables': variables
        })
        
        logger.info(f"Dimensionality reduction completed using {method}")
        return reduced_data
    
    def apply_pca(
        self,
        data: Union[xr.DataArray, np.ndarray],
        n_components: Union[int, float] = 0.95,
        standardize: bool = True
    ) -> Tuple[np.ndarray, PCA, StandardScaler]:
        """
        Apply Principal Component Analysis to data.
        
        Args:
            data: Input data array
            n_components: Number of components or variance to retain
            standardize: Whether to standardize data before PCA
            
        Returns:
            Tuple of (transformed_data, pca_model, scaler)
        """
        logger.info(f"Applying PCA with {n_components} components")
        
        if isinstance(data, xr.DataArray):
            original_shape = data.shape
            data_array = data.values
        else:
            original_shape = data.shape
            data_array = data
        
        if len(original_shape) > 2:
            n_samples = original_shape[0]
            n_features = np.prod(original_shape[1:])
            data_2d = data_array.reshape(n_samples, n_features)
        else:
            data_2d = data_array
        
        if np.isnan(data_2d).any():
            logger.warning("Found NaN values, filling with column means")
            data_2d = pd.DataFrame(data_2d).fillna(method='ffill').fillna(method='bfill').values
        
        scaler = None
        if standardize:
            scaler = StandardScaler()
            data_2d = scaler.fit_transform(data_2d)
        
        pca = PCA(n_components=n_components, random_state=42)
        transformed_data = pca.fit_transform(data_2d)
        
        logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        logger.info(f"Number of components: {pca.n_components_}")
        
        return transformed_data, pca, scaler
    
    def apply_tensor_svd(
        self,
        tensor_data: np.ndarray,
        method: str = 'tucker',
        ranks: Optional[List[int]] = None
    ) -> Tuple[Any, List[np.ndarray]]:
        """
        Apply tensor SVD decomposition to multidimensional data.
        
        Args:
            tensor_data: Input tensor (3D or higher)
            method: Decomposition method ('tucker' or 'parafac')
            ranks: Target ranks for each mode (if None, auto-determine)
            
        Returns:
            Tuple of (core_tensor, factor_matrices) for Tucker or (weights, factors) for PARAFAC
        """
        logger.info(f"Applying {method} tensor decomposition")
        
        tl.set_backend('numpy')
        
        if ranks is None:
            ranks = [min(dim, 10) for dim in tensor_data.shape]
        
        if np.isnan(tensor_data).any():
            logger.warning("Found NaN values in tensor, filling with zeros")
            tensor_data = np.nan_to_num(tensor_data)
        
        try:
            if method == 'tucker':
                core, factors = tucker(tensor_data, rank=ranks, random_state=42)
                
                reconstructed = tl.tucker_to_tensor((core, factors))
                error = tl.norm(tensor_data - reconstructed) / tl.norm(tensor_data)
                logger.info(f"Tucker reconstruction error: {error:.4f}")
                
                return core, factors
                
            elif method == 'parafac':
                rank = ranks[0] if isinstance(ranks, list) else ranks
                weights, factors = parafac(tensor_data, rank=rank, random_state=42)
                
                reconstructed = tl.cp_to_tensor((weights, factors))
                error = tl.norm(tensor_data - reconstructed) / tl.norm(tensor_data)
                logger.info(f"PARAFAC reconstruction error: {error:.4f}")
                
                return weights, factors
                
            else:
                raise ValueError(f"Unknown tensor decomposition method: {method}")
                
        except Exception as e:
            logger.error(f"Tensor decomposition failed: {e}")
            raise
    
    def extract_principal_hazard_patterns(
        self,
        climate_data: xr.Dataset,
        n_patterns: int = 5
    ) -> Dict[str, Any]:
        """
        Extract principal hazard patterns across space and time.
        
        Args:
            climate_data: Climate dataset with hazard variables
            n_patterns: Number of principal patterns to extract
            
        Returns:
            Dictionary containing patterns and their characteristics
        """
        logger.info(f"Extracting {n_patterns} principal hazard patterns")
        
        patterns = {}
        
        for var in climate_data.data_vars:
            logger.info(f"Analyzing patterns for {var}")
            
            var_data = climate_data[var]
            
            transformed_data, pca, scaler = self.apply_pca(
                var_data, 
                n_components=n_patterns,
                standardize=True
            )
            
            patterns[var] = {
                'principal_components': transformed_data,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                'loadings': pca.components_,
                'pca_model': pca,
                'scaler': scaler
            }
            
            spatial_patterns = self._extract_spatial_patterns(var_data, pca)
            patterns[var]['spatial_patterns'] = spatial_patterns
        
        return patterns
    
    def reconstruct_from_components(
        self,
        components: np.ndarray,
        pca_model: PCA,
        scaler: Optional[StandardScaler],
        original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Reconstruct original data from PCA components.
        
        Args:
            components: Principal components
            pca_model: Fitted PCA model
            scaler: Fitted scaler (if used)
            original_shape: Shape of original data
            
        Returns:
            Reconstructed data array
        """
        reconstructed_2d = pca_model.inverse_transform(components)
        
        if scaler is not None:
            reconstructed_2d = scaler.inverse_transform(reconstructed_2d)
        
        reconstructed = reconstructed_2d.reshape(original_shape)
        
        return reconstructed
    
    def _apply_pca(
        self,
        var_data: xr.DataArray,
        n_components: Union[int, float],
        var_name: str
    ) -> xr.DataArray:
        """Apply PCA to a single variable."""
        data_array = var_data.values
        original_shape = data_array.shape
        
        transformed_data, pca, scaler = self.apply_pca(
            data_array, 
            n_components=n_components,
            standardize=True
        )
        
        self.fitted_models[f"{var_name}_pca"] = pca
        self.scalers[f"{var_name}_scaler"] = scaler
        
        n_comp = transformed_data.shape[1]
        comp_coords = [f"component_{i+1}" for i in range(n_comp)]
        
        if len(original_shape) > 2:
            new_coords = {
                var_data.dims[0]: var_data.coords[var_data.dims[0]],
                'component': comp_coords
            }
            new_dims = [var_data.dims[0], 'component']
        else:
            new_coords = {'component': comp_coords}
            new_dims = ['component']
        
        reduced_var = xr.DataArray(
            transformed_data,
            coords=new_coords,
            dims=new_dims,
            attrs={
                **var_data.attrs,
                'reduction_method': 'pca',
                'n_components': n_comp,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
            }
        )
        
        return reduced_var
    
    def _apply_tucker_decomposition(
        self,
        var_data: xr.DataArray,
        n_components: Union[int, float],
        var_name: str
    ) -> xr.DataArray:
        """Apply Tucker decomposition to a variable."""
        data_array = var_data.values
        
        if isinstance(n_components, float):
            ranks = [max(1, int(dim * n_components)) for dim in data_array.shape]
        else:
            ranks = [min(n_components, dim) for dim in data_array.shape]
        
        core, factors = self.apply_tensor_svd(data_array, method='tucker', ranks=ranks)
        
        self.fitted_models[f"{var_name}_tucker"] = {
            'core': core,
            'factors': factors,
            'ranks': ranks
        }
        
        reduced_data = core
        
        new_coords = {}
        new_dims = []
        for i, rank in enumerate(ranks):
            coord_name = f"mode_{i+1}"
            new_coords[coord_name] = list(range(rank))
            new_dims.append(coord_name)
        
        reduced_var = xr.DataArray(
            reduced_data,
            coords=new_coords,
            dims=new_dims,
            attrs={
                **var_data.attrs,
                'reduction_method': 'tucker',
                'ranks': ranks
            }
        )
        
        return reduced_var
    
    def _apply_parafac_decomposition(
        self,
        var_data: xr.DataArray,
        n_components: Union[int, float],
        var_name: str
    ) -> xr.DataArray:
        """Apply PARAFAC decomposition to a variable."""
        data_array = var_data.values
        
        if isinstance(n_components, float):
            rank = max(1, int(min(data_array.shape) * n_components))
        else:
            rank = min(n_components, min(data_array.shape))
        
        weights, factors = self.apply_tensor_svd(data_array, method='parafac', ranks=rank)
        
        self.fitted_models[f"{var_name}_parafac"] = {
            'weights': weights,
            'factors': factors,
            'rank': rank
        }
        
        reduced_data = factors[0]
        
        time_dim = var_data.dims[0] if len(var_data.dims) > 0 else 'sample'
        new_coords = {
            time_dim: var_data.coords[time_dim] if time_dim in var_data.coords else range(reduced_data.shape[0]),
            'factor': list(range(rank))
        }
        
        reduced_var = xr.DataArray(
            reduced_data,
            coords=new_coords,
            dims=[time_dim, 'factor'],
            attrs={
                **var_data.attrs,
                'reduction_method': 'parafac',
                'rank': rank
            }
        )
        
        return reduced_var
    
    def _extract_spatial_patterns(
        self,
        var_data: xr.DataArray,
        pca_model: PCA
    ) -> Dict[str, np.ndarray]:
        """Extract spatial patterns from PCA loadings."""
        spatial_patterns = {}
        
        spatial_dims = [dim for dim in var_data.dims if dim not in ['time']]
        
        if len(spatial_dims) >= 2:
            spatial_shape = [var_data.sizes[dim] for dim in spatial_dims]
            n_spatial = np.prod(spatial_shape)
            
            for i, component in enumerate(pca_model.components_):
                if len(component) >= n_spatial:
                    spatial_component = component[:n_spatial].reshape(spatial_shape)
                    spatial_patterns[f"component_{i+1}"] = spatial_component
        
        return spatial_patterns
