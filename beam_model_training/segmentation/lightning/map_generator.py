from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate
from scipy.ndimage import median_filter
from skimage.morphology import remove_small_objects, remove_small_holes

from .inference import InferenceModule

class MapGeneratorModule(InferenceModule):
    """Enhanced building segmentation inference module with geospatial capabilities."""
    
    def __init__(self, config: Dict, tta_transforms: Optional[List] = None):
        super().__init__(config, tta_transforms)
        
        # Load geospatial parameters
        self.geospatial_config = config.get('geospatial', {})
        self.crs = self.geospatial_config.get('crs')
        
        # Load polygon regularization parameters
        self.regularization_params = config.get('regularization', {
            'simplify_tolerance': 1.0,
            'buffer_distance': 0.2,
            'use_mbr': True,
            'mbr_threshold': 0.85,
            'min_building_area': 2,
            'max_building_area': 30000
        })
        
        # Load edge-guided segmentation parameters
        self.edge_params = config.get('edge_guided_segmentation', {
            'building_threshold': 0.3,
            'edge_threshold': 0.1,
            'min_building_size': 50,
            'max_hole_size': 50,
            'morph_kernel_size': 5
        })

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Enhanced prediction step with geospatial processing."""
        # Get base predictions with TTA
        predictions = super().predict_step(batch, batch_idx)
        
        # Add geospatial processing
        processed_predictions = []
        for idx in range(len(predictions['predictions'])):
            pred = predictions['predictions'][idx]
            logging.info(f'unique valuse in prediction: {torch.unique(pred)}')
            conf = predictions['confidences'][idx]
            image_path = predictions['image_path'][idx]
            
            # Load geospatial reference
            with rxr.open_rasterio(image_path) as src:
                # Create mask DataArray with geospatial information
                mask_da = xr.DataArray(
                    pred.cpu().numpy(),
                    dims=('y', 'x'),
                    coords={'y': src.y, 'x': src.x}
                )
                mask_da.rio.write_crs(src.rio.crs, inplace=True)
                mask_da.rio.write_transform(src.rio.transform(), inplace=True)
                
                # Process using edge-guided segmentation
                if self.config.get('use_edge_guided', False):
                    mask_da = self._edge_guided_segmentation(
                        mask_da,
                        conf.cpu().numpy(),
                        predictions['probabilities'][idx].cpu().numpy()
                    )
                
                # Generate and regularize polygons
                polygons = self._generate_polygons(mask_da)
                confidence_da = xr.DataArray(
                    conf.cpu().numpy(),
                    dims=mask_da.dims,
                    coords=mask_da.coords
                )
                confidence_da.rio.write_crs(src.rio.crs, inplace=True)
                confidence_da.rio.write_transform(src.rio.transform(), inplace=True)

                processed_predictions.append({
                    'mask': mask_da,
                    'polygons': polygons,
                    'confidence': confidence_da,  # Now it's a DataArray with rio attributes
                    'image_path': image_path,
                    'geospatial_reference': {
                        'crs': src.rio.crs,
                        'transform': src.rio.transform()
                    }
                })
        
        predictions['processed'] = processed_predictions
        return predictions

    def _edge_guided_segmentation(
        self, 
        mask_da: xr.DataArray,
        confidence: np.ndarray,
        probabilities: np.ndarray
    ) -> xr.DataArray:
        """Apply edge-guided segmentation to refine building predictions."""
        # Extract parameters
        building_thresh = self.edge_params['building_threshold']
        edge_thresh = self.edge_params['edge_threshold']
        min_size = self.edge_params['min_building_size']
        max_hole = self.edge_params['max_hole_size']
        
        # Get building and edge probabilities
        building_prob = probabilities[1]  # Assuming class order: [background, building, edge]
        edge_prob = probabilities[2] if probabilities.shape[0] > 2 else None
        
        # Create initial mask
        if edge_prob is not None:
            initial_mask = (1 - (edge_prob > edge_thresh).astype(np.uint8)) * \
                         (building_prob > building_thresh).astype(np.uint8)
        else:
            initial_mask = (building_prob > building_thresh).astype(np.uint8)
        
        # Apply morphological operations
        # Remove small objects and fill holes
        cleaned_mask = remove_small_objects(initial_mask.astype(bool), min_size=min_size)
        filled_mask = remove_small_holes(cleaned_mask, area_threshold=max_hole)
        
        # Apply median filter for smoothing
        smoothed_mask = median_filter(filled_mask.astype(np.uint8), size=3)
        
        # Create new DataArray with processed mask
        processed_da = xr.DataArray(
            smoothed_mask,
            dims=mask_da.dims,
            coords=mask_da.coords
        )
        processed_da.rio.write_crs(mask_da.rio.crs, inplace=True)
        processed_da.rio.write_transform(mask_da.rio.transform(), inplace=True)
        
        return processed_da

    def _generate_polygons(self, mask_da: xr.DataArray) -> gpd.GeoDataFrame:
        """Generate and regularize building polygons from mask."""
        from rasterio import features
        
        # Extract polygons from mask
        shapes = features.shapes(
            mask_da.values.astype(np.uint8),
            transform=mask_da.rio.transform()
        )

        for shape, value in shapes:
            logging.info(f"Value: {value}")

        logging.info(f"Unique values in mask_da: {np.unique(mask_da.values)}")
        
        # Convert to GeoDataFrame
        polygons = []
        for shape, value in shapes:
            if int(value) == 1:  # Building class
                poly = Polygon(shape['coordinates'][0])
                if poly.is_valid and not poly.is_empty:
                    polygons.append(self._regularize_polygon(poly))
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            geometry=polygons,
            crs=mask_da.rio.crs
        )
        
        # Area calculations and filtering
        if len(gdf) > 0:  # Only process if we have polygons
            # Get center point to determine UTM zone
            center = gdf.unary_union.centroid
            if not center.is_empty:
                lat, lon = center.y, center.x
                utm_zone = int((lon + 180) / 6) + 1
                utm_crs = f'EPSG:326{utm_zone:02d}' if lat >= 0 else f'EPSG:327{utm_zone:02d}'
                gdf = gdf.to_crs(utm_crs)
                # Calculate areas
                gdf['area'] = gdf.geometry.area
                # Project back to original CRS
                gdf = gdf.to_crs(mask_da.rio.crs)
            else:
                gdf['area'] = 0

            # Keep the size filtering
            if len(gdf) > 0:  # Still need this check as previous steps might have removed all polygons
                gdf = gdf[
                    (gdf['area'] > self.regularization_params['min_building_area']) &
                    (gdf['area'] < self.regularization_params['max_building_area'])
                ]

    def _regularize_polygon(self, polygon: Polygon) -> Polygon:
        """Regularize building polygon using specified parameters."""
        if not polygon.is_valid or polygon.is_empty:
            return polygon
            
        # Calculate dominant angle
        coords = np.array(polygon.exterior.coords)
        edges = np.diff(coords, axis=0)
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        hist, _ = np.histogram(angles, bins=18, range=(-np.pi, np.pi))
        dominant_angle = (np.argmax(hist) * np.pi / 9) - np.pi
        
        # Rotate to align with dominant angle
        rotated = rotate(polygon, -np.degrees(dominant_angle), origin='centroid')
        
        # Simplify and buffer
        params = self.regularization_params
        regularized = rotated.simplify(params['simplify_tolerance'])
        regularized = regularized.buffer(params['buffer_distance']).buffer(-params['buffer_distance'])
        
        # Apply minimum bounding rectangle if configured
        if params['use_mbr']:
            mbr = regularized.minimum_rotated_rectangle
            if mbr.area > 0:
                area_ratio = regularized.area / mbr.area
                if area_ratio > params['mbr_threshold']:
                    regularized = mbr
        
        # Rotate back
        regularized = rotate(regularized, np.degrees(dominant_angle), origin='centroid')
        
        return regularized

    def merge_predictions(self, predictions: List[Dict], boundaries_gdf: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
        """Merge predictions across tiles and optionally filter by boundary areas."""
        all_polygons = []
        
        # Collect all polygons
        for pred in predictions:
            if 'processed' in pred:
                for p in pred['processed']:
                    if p['polygons'] is not None and len(p['polygons']) > 0:  # Add this check
                        all_polygons.extend(p['polygons'].geometry.tolist())
        
        # Handle case where no valid polygons were found
        if not all_polygons:
            logging.warning("No valid polygons found in predictions")
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)
        
        # Create merged GeoDataFrame
        merged_gdf = gpd.GeoDataFrame(
            geometry=all_polygons,
            crs=self.crs
        )
        
        # Filter by boundaries if provided
        if boundaries_gdf is not None:
            if boundaries_gdf.crs != merged_gdf.crs:
                boundaries_gdf = boundaries_gdf.to_crs(merged_gdf.crs)
            merged_gdf = gpd.sjoin(merged_gdf, boundaries_gdf, how='inner', op='within')
        
        return merged_gdf