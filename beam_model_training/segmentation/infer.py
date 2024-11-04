import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import geopandas as gpd
import math
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import torch
import xarray as xr
from fastai.vision.all import *
from PIL import Image
from pyproj import CRS
from rioxarray.merge import merge_arrays
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.affinity import rotate
from shapely.validation import explain_validity
from scipy import ndimage
from scipy.ndimage import median_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, remove_small_holes
from tqdm import tqdm

from preprocess.sample import tile_in_settlement
from segmentation.train import Trainer
from utils.base_class import BaseClass
from utils.helpers import get_rgb_channels, seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MapGenerator(BaseClass):
    """
    A class used to generate maps from images using a trained model.

    ...

    Attributes
    ----------
    config : dict
        Configuration dictionary specifying paths and model details.
    generate_preds : bool
        Flag indicating whether predictions should be generated during evaluation.

    Methods
    -------
    _get_image_files(images_dir):
        Retrieve a list of image files from the specified directory.
    create_shp_from_mask(mask_da):
        Creates a GeoDataFrame from a binary mask DataArray.
    _group_tiles_by_connected_area(mask_tiles, boundaries_gdf, primary_key):
        Groups tiles by connected area for vectorization.
    single_tile_inference(image_file, AOI_gpd=None, write_shp=True):
        Processes a single image file for inference and optionally writes output as shapefile.
    _filter_by_areas(output_files, boundaries_gdf, primary_key):
        Filters and merges rasters by settlement areas.
    create_tile_inferences(images_dir=None, boundaries_gdf=None, primary_key=""):
        Performs inference on each tile in the images directory and (optionally) merges the results to match a settlement boundaries shapefile indexed by primary_key.
    generate_map_from_images(images_dir=None, settlements=None, primary_key="")
        Orchestrates the inference process across all tiles in the specified directory, optionally merging results based on settlement boundaries.
    """

    def __init__(
        self, project_dir, config_name=None, model_path=None, generate_preds=False
    ):
        """
        Constructs all the necessary attributes for the MapGenerator object.

        Parameters
        ----------
            generate_preds : bool
                Flag indicating whether predictions should be generated during evaluation.
            project_dir : str
                Path to the project directory, containing one or more models, as well as images and settlement boundaries (optional) for map generation.
            config_name : str
                The configuration file name. If missing, the constructor will look for a single file in the project directory.
        """

        super().__init__(project_dir, config_name)
        seed(self.config["seed"])
        read_dirs = ["test_images", "models"]
        prediction_dirs = ["predictions", "shapefiles"]
        self.generate_preds = generate_preds
        self.use_edge_detection = "edge" in self.config.get("codes", [])
        if self.generate_preds:
            super().load_dir_structure(read_dirs=read_dirs, write_dirs=prediction_dirs)
        else:
            super().load_dir_structure(read_dirs=(read_dirs + prediction_dirs))
        if model_path:
            self.model_version = Path(model_path).stem
            self.learner = load_learner(model_path, cpu=False if torch.cuda.is_available() else True)
        else:
            try:
                model_path = super().load_model_path(self.config.get("model_version"))
                self.model_version = model_path.parent.name
                self.learner = load_learner(model_path, cpu=False if torch.cuda.is_available() else True)
            except KeyError as e:
                raise KeyError(f"Config must have a value for {e}.")
        self.crs = self.get_crs(self.test_images_dir)
        self.erosion = self.config["tiling"]["erosion"]
        self._load_edge_guided_params()
        self._load_regularization_params()

    def _load_edge_guided_params(self):
        """
        Load edge-guided segmentation parameters.
        """
        edge_guided_keys = ["building_threshold", "edge_threshold", "min_building_size", "max_hole_size", "morph_kernel_size"]
        self.edge_guided_params = {k: self.config.get("edge_guided_segmentation", {}).get(k) for k in edge_guided_keys}
        
        # Set default values if not provided
        defaults = {
            "building_threshold": 0.3,
            "edge_threshold": 0.1,
            "min_building_size": 50,
            "max_hole_size": 50,
            "morph_kernel_size": 5
        }
        
        for k, v in defaults.items():
            if self.edge_guided_params[k] is None:
                self.edge_guided_params[k] = v
        
        # Set as attributes
        for k, v in self.edge_guided_params.items():
            setattr(self, k, v)

    def _load_regularization_params(self):
        """
        Load regularization parameters.
        """
        regularization_keys = [
            "simplify_tolerance", 
            "buffer_distance", 
            "use_mbr", 
            "mbr_threshold", 
            "min_building_area", 
            "max_building_area",
            "angle_tolerance",
            "min_edge_length",
            "orthogonal_threshold",
            "dominant_angle_count",
            "edge_snap_threshold",
            "smooth_iterations"]
        self.regularization_params = {k: self.config.get("regularization", {}).get(k) for k in regularization_keys}
        
        # Set default values if not provided
        defaults = {
            "simplify_tolerance": 1.0,
            "buffer_distance": 0.2,
            "use_mbr": True,
            "mbr_threshold": 0.85,
            "min_building_area": 2,
            "max_building_area": 30000,
            "angle_tolerance": 10.0,
            "min_edge_length": 2.0,
            "orthogonal_threshold": 15.0,
            "dominant_angle_count": 2,
            "edge_snap_threshold": 1.0,
            "smooth_iterations": 1            
        }
        
        for k, v in defaults.items():
            if self.regularization_params[k] is None:
                self.regularization_params[k] = v
        
        # Set as attributes
        for k, v in self.regularization_params.items():
            setattr(self, k, v)    

    def get_crs(self, images_dir):
        """Initializes the CRS from a directory of images, assuming the CRS is consistent across the directory."""
        images_dir_path = Path(images_dir)
        if any(images_dir_path.iterdir()):
            # Get the first image file
            img_file = next(images_dir_path.glob("*"))  # Adjust the pattern as needed
            img = rxr.open_rasterio(img_file)
            logging.info(f"CRS now initialized to {img.rio.crs}.")
            return img.rio.crs
        else:
            logging.warning("CRS could not be set at initialization.")

    def _validate_polygon(self, poly, idx):
        """
        Validate a polygon with detailed logging.
        
        Returns:
        - (bool, str): (is_valid, reason_if_invalid)
        """
        if poly is None:
            return False, "Polygon is None"
            
        try:
            if not isinstance(poly, (Polygon, MultiPolygon)):
                return False, f"Not a polygon type: {type(poly)}"
                
            if poly.is_empty:
                return False, "Polygon is empty"
                
            if not poly.is_valid:
                reason = explain_validity(poly)
                return False, f"Invalid geometry: {reason}"
                
            if len(poly.exterior.coords) <= 3:
                return False, "Too few coordinates"
                
            if poly.area == 0:
                return False, "Zero area"
                
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def create_shp_from_mask(self, mask_da):
        """
        Creates a GeoDataFrame from a binary mask DataArray.

        Parameters
        ----------
        mask_da : xr.DataArray
            Binary mask DataArray of the same size as the original image with named dimensions and coordinates.

        Returns
        -------
        gdf : gpd.GeoDataFrame
            GeoDataFrame representing the shapes extracted from the mask.
        """

        if not isinstance(mask_da, xr.DataArray) or mask_da.name is None:
            raise TypeError("mask_array must be a named DataArray.")

        if mask_da is None or mask_da.values.max() == 0:
            logging.warning("Mask is empty or all zeros.")
            return gpd.GeoDataFrame(geometry=[])

        if mask_da.rio.crs is None:
            raise ValueError("mask_array does not have a CRS attached.")

        # Extract polygons
        mask_values = mask_da.values
        filtered_mask = self.apply_median_filter_to_labels(mask_values, size=3)
        
        transform = mask_da.rio.transform()
        polygons = self._extract_precise_polygons(filtered_mask, transform)
        
        if not polygons:
            logging.warning(f"[{mask_da.name}] No valid polygons extracted")
            return gpd.GeoDataFrame(geometry=[])

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=mask_da.rio.crs)
        
        # Project to local UTM zone
        if gdf.crs.is_geographic:
            try:
                # Get UTM zone from center coordinates
                center_lon = gdf.geometry.centroid.x.mean()
                center_lat = gdf.geometry.centroid.y.mean()
                utm_zone = int((center_lon + 180) / 6) + 1
                utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84"
                if center_lat < 0:
                    utm_crs += " +south"
                
                gdf_projected = gdf.to_crs(utm_crs)
                
                # Recalculate areas after projection
                gdf_projected['area'] = gdf_projected.geometry.area
                
                logging.info(f"[{mask_da.name}] Area statistics in UTM zone {utm_zone}:")
                logging.info(f"  - Mean area: {gdf_projected['area'].mean():.2f} sq meters")
                logging.info(f"  - Area range: [{gdf_projected['area'].min():.2f}, {gdf_projected['area'].max():.2f}] sq meters")
                
                # Filter by area thresholds (in square meters)
                min_area = 2  # 2 square meters
                max_area = 3000  # 3000 square meters
                gdf_projected = gdf_projected[
                    (gdf_projected['area'] >= min_area) & 
                    (gdf_projected['area'] <= max_area)
                ]
                
                if not gdf_projected.empty:
                    # Project back to original CRS
                    gdf_final = gdf_projected.to_crs(mask_da.rio.crs)
                    return gdf_final
                else:
                    logging.warning(f"[{mask_da.name}] No polygons within area thresholds")
                    return gpd.GeoDataFrame(geometry=[])
                    
            except Exception as e:
                logging.error(f"[{mask_da.name}] Projection/area calculation failed: {str(e)}")
                return gpd.GeoDataFrame(geometry=[])
        
        return gdf

    def _extract_precise_polygons(self, mask, transform):
        """Enhanced polygon extraction optimized for tiles."""
        from skimage import measure
        
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        
        # Log mask properties
        logging.info(f"Mask properties: shape={mask.shape}, dtype={mask.dtype}, range=[{mask.min()}, {mask.max()}]")
        
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            logging.error(f"Mask shape {mask.shape} is too small for contour extraction")
            return []
        
        # Extract contours
        contours = measure.find_contours(
            mask, 
            level=0.5, 
            fully_connected='high'
        )
        
        logging.info(f"Found {len(contours)} contours")
        
        polygons = []
        contour_stats = defaultdict(int)
        
        # Calculate pixel area thresholds
        min_pixels = 10  # Minimum 10 pixels
        max_pixels = (mask.shape[0] * mask.shape[1]) // 4  # Maximum 1/4 of tile
        
        for i, contour in enumerate(contours):
            try:
                if len(contour) < 4:
                    contour_stats['too_few_points'] += 1
                    continue
                    
                # Convert to pixel coordinates first
                pixel_coords = []
                for y, x in contour:
                    pixel_coords.append((x, y))
                    
                # Close the polygon in pixel space
                if pixel_coords[0] != pixel_coords[-1]:
                    pixel_coords.append(pixel_coords[0])
                
                # Create polygon in pixel space to check area
                pixel_poly = Polygon(pixel_coords)
                pixel_count = pixel_poly.area  # Area in pixel space
                
                if min_pixels <= pixel_count <= max_pixels:
                    # Convert to world coordinates
                    world_coords = []
                    for x, y in pixel_coords:
                        real_x, real_y = transform * (x, y)
                        world_coords.append((real_x, real_y))
                    
                    try:
                        poly = Polygon(world_coords)
                        if not poly.is_valid:
                            cleaned = poly.buffer(0)
                            if cleaned.is_valid and not cleaned.is_empty:
                                poly = cleaned
                                contour_stats['cleaned_invalid'] += 1
                        
                        if poly.is_valid and not poly.is_empty:
                            # Try regularization with logging
                            try:
                                regularized = self._regularize_polygon(poly, force_rectangle=False)
                                if regularized and regularized.is_valid:
                                    original_area = poly.area
                                    regularized_area = regularized.area
                                    area_change = (regularized_area - original_area) / original_area * 100
                                    
                                    original_points = len(poly.exterior.coords)
                                    regularized_points = len(regularized.exterior.coords)
                                    
                                    logging.debug(f"Regularization metrics for polygon {i}:")
                                    logging.debug(f"  - Points: {original_points} -> {regularized_points}")
                                    logging.debug(f"  - Area change: {area_change:.1f}%")
                                    
                                    if abs(area_change) > 20:  # More than 20% area change
                                        logging.debug("  - Using original (area change too large)")
                                        polygons.append(poly)
                                        contour_stats['kept_original_area'] += 1
                                    else:
                                        polygons.append(regularized)
                                        contour_stats['successfully_regularized'] += 1
                                else:
                                    polygons.append(poly)
                                    contour_stats['regularization_failed'] += 1
                            except Exception as e:
                                logging.debug(f"Regularization failed for polygon {i}: {str(e)}")
                                polygons.append(poly)
                                contour_stats['regularization_error'] += 1
                        else:
                            contour_stats['invalid_polygon'] += 1
                    except Exception as e:
                        contour_stats[f'polygon_creation_error_{type(e).__name__}'] += 1
                else:
                    contour_stats['invalid_pixel_area'] += 1
                    
            except Exception as e:
                contour_stats[f'contour_processing_error_{type(e).__name__}'] += 1
                
        logging.info("Contour processing statistics:")
        for stat, count in contour_stats.items():
            logging.info(f"  - {stat}: {count}")
                        
        return polygons

    def _regularize_polygon(self, polygon, force_rectangle=False):
        """
        Regularize a polygon using multiple techniques.
        
        Args:
            polygon: Input shapely polygon
            force_rectangle: Whether to force rectangular shape (default: False)
        
        Returns:
            Regularized polygon or None if regularization fails
        """
        try:
            if not polygon.is_valid or polygon.is_empty:
                return None
                
            # 1. Initial simplification to remove noise
            simplified = polygon.simplify(tolerance=0.5, preserve_topology=True)
            if not simplified.is_valid:
                return polygon
                
            # 2. Calculate dominant angles
            coords = np.array(simplified.exterior.coords)
            edges = np.diff(coords, axis=0)
            angles = np.arctan2(edges[:, 1], edges[:, 0])
            angles_deg = np.degrees(angles) % 180
            
            # Find dominant angles (allowing for multiple)
            hist, bins = np.histogram(angles_deg, bins=36, range=(0, 180))
            peaks = []
            angle_threshold = 15  # degrees
            
            for i, count in enumerate(hist):
                if count > len(angles) * 0.1:  # At least 10% of edges
                    angle = (bins[i] + bins[i+1]) / 2
                    peaks.append(angle)
            
            # 3. Align edges to dominant angles
            aligned_coords = []
            prev_point = None
            
            for i in range(len(coords) - 1):
                current_point = coords[i]
                next_point = coords[i + 1]
                
                if prev_point is not None:
                    # Calculate edge angle
                    edge_angle = np.degrees(np.arctan2(
                        next_point[1] - current_point[1],
                        next_point[0] - current_point[0]
                    )) % 180
                    
                    # Find closest dominant angle
                    closest_peak = None
                    min_diff = angle_threshold
                    
                    for peak in peaks:
                        diff = abs((edge_angle - peak + 90) % 180 - 90)
                        if diff < min_diff:
                            min_diff = diff
                            closest_peak = peak
                    
                    if closest_peak is not None:
                        # Align edge to dominant angle
                        angle_rad = np.radians(closest_peak)
                        edge_length = np.sqrt(
                            (next_point[0] - current_point[0])**2 +
                            (next_point[1] - current_point[1])**2
                        )
                        
                        new_point = (
                            current_point[0] + edge_length * np.cos(angle_rad),
                            current_point[1] + edge_length * np.sin(angle_rad)
                        )
                        aligned_coords.append(new_point)
                    else:
                        aligned_coords.append(tuple(current_point))
                else:
                    aligned_coords.append(tuple(current_point))
                    
                prev_point = current_point
                
            # Close the polygon
            aligned_coords.append(aligned_coords[0])
            
            # 4. Create aligned polygon
            try:
                aligned = Polygon(aligned_coords)
                if aligned.is_valid and not aligned.is_empty:
                    # 5. Optional rectangle conversion
                    if force_rectangle:
                        rect = aligned.minimum_rotated_rectangle
                        if rect.is_valid and not rect.is_empty:
                            # Only use rectangle if it's similar enough to original
                            if rect.area / aligned.area < 1.2:  # Within 20% area difference
                                return rect
                    return aligned
            except Exception as e:
                logging.debug(f"Failed to create aligned polygon: {str(e)}")
            
            # Fallback to simplified version if alignment fails
            return simplified if simplified.is_valid else polygon
            
        except Exception as e:
            logging.debug(f"Regularization failed: {str(e)}")
            return polygon  # Return original polygon if regularization fails



    def _advanced_regularization(self, polygon):
        """Enhanced regularization with better error handling"""
        try:
            if isinstance(polygon, MultiPolygon):
                return MultiPolygon([
                    self._advanced_regularization(p) for p in polygon.geoms
                    if p.is_valid and not p.is_empty
                ])
                
            if not polygon.is_valid or polygon.is_empty:
                return None
                
            # Calculate dominant angles
            dominant_angles = self._calculate_dominant_angles(polygon)
            
            # Log original geometry properties
            logging.debug(f"Original geometry - Area: {polygon.area:.2f}, Points: {len(polygon.exterior.coords)}")
            
            # Rotate to align with dominant angle
            main_angle = dominant_angles[0] if dominant_angles else 0
            try:
                rotated = rotate(polygon, -main_angle, origin='centroid')
                if not rotated.is_valid:
                    logging.debug("Rotation produced invalid geometry")
                    return None
            except Exception as e:
                logging.debug(f"Rotation failed: {str(e)}")
                return None
                
            # Enhanced regularization
            try:
                regularized = self._regularize_orthogonal(rotated)
                if not regularized.is_valid:
                    logging.debug("Orthogonal regularization produced invalid geometry")
                    return None
            except Exception as e:
                logging.debug(f"Orthogonal regularization failed: {str(e)}")
                return None
                
            # Rotate back
            try:
                final = rotate(regularized, main_angle, origin='centroid')
                if not final.is_valid:
                    logging.debug("Final rotation produced invalid geometry")
                    return None
            except Exception as e:
                logging.debug(f"Final rotation failed: {str(e)}")
                return None
                
            # Verify final geometry
            if final.is_valid and final.area > 0:
                return final
            return None
            
        except Exception as e:
            logging.debug(f"Advanced regularization failed: {str(e)}")
            return None

    def _calculate_dominant_angles(self, polygon):
        """Calculate multiple dominant angles in the polygon."""
        coords = np.array(polygon.exterior.coords)
        edges = np.diff(coords, axis=0)
        
        # Calculate angles of all edges longer than min_edge_length
        edge_lengths = np.sqrt(np.sum(edges**2, axis=1))
        valid_edges = edges[edge_lengths > self.min_edge_length]
        
        if len(valid_edges) == 0:
            return [0]  # Return default angle if no valid edges
            
        angles = np.arctan2(valid_edges[:, 1], valid_edges[:, 0])
        angles = np.degrees(angles) % 180
        
        # Create histogram of angles
        hist, bins = np.histogram(angles, bins=36, range=(0, 180))
        
        # Find peaks in histogram
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=max(hist)/4)
        
        dominant_angles = bins[peaks]
        
        # Sort by prominence and take top N angles
        angle_counts = [(angle, hist[peak]) for angle, peak in zip(dominant_angles, peaks)]
        angle_counts.sort(key=lambda x: x[1], reverse=True)
        
        return [angle for angle, _ in angle_counts[:self.dominant_angle_count]]

    def _regularize_orthogonal(self, polygon):
        """Regularize assuming roughly orthogonal angles."""
        coords = np.array(polygon.exterior.coords)
        new_coords = []
        
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            
            # Calculate edge angle
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            angle_deg = np.degrees(angle) % 90
            
            # Snap to nearest 90 degree increment if within threshold
            if angle_deg < self.orthogonal_threshold or \
            angle_deg > (90 - self.orthogonal_threshold):
                if angle_deg < 45:
                    p2 = np.array([p2[0], p1[1]])
                else:
                    p2 = np.array([p1[0], p2[1]])
            
            new_coords.append(p1)
            
            # Snap nearby vertices
            if len(new_coords) > 1:
                prev = np.array(new_coords[-2])
                curr = np.array(new_coords[-1])
                if np.linalg.norm(prev - curr) < self.edge_snap_threshold:
                    new_coords[-1] = tuple(prev)  # Snap to previous point
        
        new_coords.append(new_coords[0])  # Close the polygon
        
        try:
            return Polygon(new_coords)
        except:
            return polygon

    def apply_median_filter_to_labels(self, labeled_mask, size=3):
        unique_labels = np.unique(labeled_mask)
        filtered_mask = np.zeros_like(labeled_mask)
        for label in unique_labels:
            if label == 0:  # background
                continue
            mask = labeled_mask == label
            filtered = median_filter(mask.astype(np.uint8), size=size)
            filtered_mask[filtered > 0] = label
        return filtered_mask    

    def _group_tiles_by_connected_area(self, mask_tiles, boundaries_gdf, primary_key):
        """
        Groups tiles by area covered by a connected group of tiles, from which building polygons will be extracted at once.

        Parameters
        ----------
        mask_tiles : list
            A list of rasterio dataset objects representing tiles.
        boundaries_gdf : gpd.GeoDataFrame
            A GeoDataFrame containing settlement data. Each row represents a settlement with a geometry column for its polygon boundaries.
        primary_key : str
            The name of the column in the boundaries GeoDataFrame to use as primary key.

        Returns
        -------
        grouped_tiles : pd.Series
            A pandas Series where each entry is a list of tile names associated with a merged settlement group. The index of the Series is a MultiIndex with levels [f'{primary_key}_group', 'geometry'].
        """

        if primary_key not in boundaries_gdf.columns:
            raise KeyError(
                f"The primary_key '{primary_key}' is not present in the boundaries_gdf columns. "
                f"Please update the dataframe, or choose a key from the following columns as primary key argument: {list(boundaries_gdf.columns)}"
            )

        # Create a GeoDataFrame from the list of tiles
        data = [
            {"name": da.name, "geometry": box(*da.rio.bounds())} for da in mask_tiles
        ]
        tiles_gdf = gpd.GeoDataFrame(data, crs=self.crs)

        # Find settlement boundaries in tiles and create a graph from polygons for connected component search
        joined = gpd.sjoin(boundaries_gdf, tiles_gdf, how="inner", op="intersects")

        G = nx.Graph()

        # Add nodes for each settlement
        for index, row in joined.iterrows():
            G.add_node(
                index, **{primary_key: row[primary_key], "geometry": row["geometry"]}
            )

        # Add edges between settlement polygons that share the same tile (indicating potential overlap)
        for _, group in joined.groupby("name"):
            for i, _ in group.iterrows():
                for j, _ in group.iterrows():
                    if i != j:
                        G.add_edge(i, j)

        # Find connected components (groups of polygons covered by same group of tiles)
        components = list(nx.connected_components(G))

        merged_settlements = []
        for component in components:
            # Extract the polygons in this component
            component_settlements = joined.loc[list(component)]

            # Merge their geometries into a single geometry (unary_union)
            merged_geometry = component_settlements["geometry"].unary_union

            # Create a new entry for the merged settlement
            merged_settlement = {
                f"{primary_key}_group": "_".join(
                    map(str, component_settlements[primary_key].unique())
                ),
                "geometry": merged_geometry,
            }
            merged_settlements.append(merged_settlement)

        # Create a new GeoDataFrame for the merged polygon groups
        merged_settlements_gdf = gpd.GeoDataFrame(merged_settlements, crs=self.crs)

        # Step 4: Repeat spatial join with the merged settlements to update tile associations
        final_joined = gpd.sjoin(
            merged_settlements_gdf, tiles_gdf, how="inner", op="intersects"
        )
        grouped_tiles = final_joined.groupby([f"{primary_key}_group", "geometry"])[
            "name"
        ].apply(list)

        return grouped_tiles

    def _filter_by_areas(self, output_files, boundaries_gdf, primary_key):
        """
        Filters and merges geospatial data arrays by settlement areas.

        Parameters
        ----------
        output_files : list of xarray.DataArray
            Geospatial data arrays with a 'name' attribute.
        boundaries_gdf : gpd.GeoDataFrame
            Settlement areas with geometries and attributes.
        primary_key : str
            Column in `boundaries_gdf` for unique settlement identification.

        Returns
        -------
        filtered_buildings : gpd.GeoDataFrame
            Merged and masked geospatial data for each settlement, enriched with settlement attributes.
        """

        grouped_tiles = self._group_tiles_by_connected_area(
            output_files, boundaries_gdf, primary_key
        )
        gdfs = []
        components_list = list(grouped_tiles.items())

        for (pkey, _), names in tqdm(components_list, desc="Progressing"):
            # Filter the data arrays by name in output_files
            settlement_tiles = [da for da in output_files if da.name in names]

            combined = merge_arrays(settlement_tiles)
            combined.name = pkey

            gdf = self.create_shp_from_mask(combined)
            gdfs.append(gdf)

        all_buildings = gpd.GeoDataFrame(
            pd.concat(gdfs), geometry="geometry", crs=self.crs
        )
        filtered_buildings = gpd.sjoin(
            all_buildings, boundaries_gdf, how="inner", op="intersects"
        )
        filtered_buildings = filtered_buildings.drop_duplicates(
            subset=["geometry"]
        ).reset_index()

        return filtered_buildings
    
    def _calculate_area(self, gdf):
        """
        Private method to calculate areas for geometries in a GeoDataFrame, reprojecting if necessary.
        
        Args:
            gdf (GeoDataFrame): Input GeoDataFrame with geometries.
        
        Returns:
            GeoDataFrame: Input GeoDataFrame with added 'bldg_area' column.
        """
        original_crs = gdf.crs
        
        if original_crs.is_geographic:
            # Project to an equal area projection
            gdf_projected = gdf.to_crs({'proj':'cea'})
            gdf['bldg_area'] = gdf_projected.area
        else:
            gdf['bldg_area'] = gdf.area
        
        return gdf    
    
    def regularize_polygon(self, polygon, simplify_tolerance=1.0, buffer_distance=0.5, use_mbr=False, mbr_threshold=0.85):
        """
        Regularize a polygon using simplification, buffering, and optionally minimum bounding rectangle.

        Parameters:
        - polygon: Shapely Polygon or MultiPolygon
        - simplify_tolerance: Tolerance for polygon simplification
        - buffer_distance: Distance for buffer operations
        - use_mbr: Whether to consider using minimum bounding rectangle
        - mbr_threshold: Threshold for deciding to use MBR (based on area ratio)

        Returns:
        - Regularized Shapely Polygon or MultiPolygon
        """
        if polygon.is_empty or polygon.area == 0:
            return polygon

        # Handle MultiPolygons
        if isinstance(polygon, MultiPolygon):
            return MultiPolygon([self.regularize_polygon(p, simplify_tolerance, buffer_distance, use_mbr, mbr_threshold) for p in polygon.geoms])

        # Simplify
        simplified = polygon.simplify(simplify_tolerance, preserve_topology=True)
        
        # Smooth using buffer
        smoothed = simplified.buffer(-buffer_distance).buffer(buffer_distance)
        
        # Fill holes
        filled = smoothed.buffer(0)

        if use_mbr and filled.area > 0:
            # Calculate the minimum bounding rectangle
            mbr = filled.minimum_rotated_rectangle
            
            # Calculate area ratio, avoiding division by zero
            if mbr.area > 0:
                area_ratio = filled.area / mbr.area
                
                # If the area ratio is above the threshold, use the MBR
                if area_ratio > mbr_threshold:
                    return mbr

        return filled

    def _calculate_dominant_angle(self, polygon):
        """Calculate the dominant angle of a polygon."""
        coords = np.array(polygon.exterior.coords)
        edges = np.diff(coords, axis=0)
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        hist, _ = np.histogram(angles, bins=18, range=(-np.pi, np.pi))
        dominant_angle = (np.argmax(hist) * np.pi / 9) - np.pi
        return np.degrees(dominant_angle)




    def single_tile_inference(self, image_file, boundaries_gdf=None, write_shp=True):
        """
        Processes a single image file for inference and optionally writes a shapefile.

        Parameters
        ----------
        image_file : str or Path
            Path to the image file.
        boundaries_gdf : gpd.GeoDataFrame, optional
            Area of Interest as a GeoPandas DataFrame (default: None).
        write_shp : bool, optional
            Whether to write the output to a shapefile (default: True).

        Returns
        -------
        output_da : xarray.DataArray
            Geospatial data array with inference results.
        """
        tile = get_rgb_channels(image_file)
        if tile.rio.crs != self.crs:
            tile = tile.rio.reproject(self.crs)

        assert (
            self.learner
        ), "The model version must be specified in the configuration settings."

        spatial_check = boundaries_gdf is not None and not boundaries_gdf.empty

        if spatial_check and not tile_in_settlement(tile, boundaries_gdf):
            return
        # Run inference and save as grayscale image
        image = Image.fromarray(tile.data.transpose(1, 2, 0))
        prediction = self.learner.predict(image)
        
        # Unpack prediction
        fully_decoded, decoded, raw_pred = prediction
        
        # raw_pred contains the probabilities
        class_probs = raw_pred.cpu().numpy()
        
        # Extract building and edge probabilities
        background_prob = class_probs[0, :, :]
        building_prob = class_probs[1, :, :]

        if self.use_edge_detection:
            edge_prob = class_probs[2, :, :]
            initial_mask = (1 - (edge_prob > self.edge_threshold).astype(np.uint8)) * (building_prob > self.building_threshold).astype(np.uint8)
            
            building_mask = self.edge_guided_segmentation(
                building_prob, 
                edge_prob,
                building_threshold=self.building_threshold,
                edge_threshold=self.edge_threshold,
                min_building_size=self.min_building_size,
                max_hole_size=self.max_hole_size,
                morph_kernel_size=self.morph_kernel_size
            )
        else:
            edge_prob = None
            initial_mask = (building_prob > self.building_threshold).astype(np.uint8)
            building_mask = initial_mask.copy()

        inference_path = self.predictions_dir / f"{image_file.stem}_INFERENCE.TIF"

        # Create a DataArray from the output and assign the coordinate reference system and affine transform from original tile
        output_da = xr.DataArray(
            name=str(image_file.stem),
            data=building_mask.astype(float),
            dims=["y", "x"],
            coords={"y": tile.coords["y"], "x": tile.coords["x"]},
        )
        output_da = output_da.rio.write_crs(self.crs)
        output_da = output_da.rio.write_transform(tile.rio.transform())
        output_da.rio.to_raster(inference_path)

        viz_path = self.predictions_dir / f"{image_file.stem}_visualizations.png"
        rgb_image = tile.data.transpose(1, 2, 0)  # Convert to HWC format for visualization
        self.save_threshold_visualizations(rgb_image, building_prob, edge_prob, 
                                   building_mask, 
                                   viz_path)


        # Generate shapefile
        if write_shp:
            shp_path = self.shapefiles_dir / f"{image_file.stem}_predicted.shp"
            vector_df = self.create_shp_from_mask(output_da)
            if not vector_df.empty:
                vector_df.to_file(
                    shp_path,
                    driver="ESRI Shapefile",
                )
        return output_da


    def edge_guided_segmentation(self, building_prob, edge_prob=None, 
                                building_threshold=0.3, edge_threshold=0.3, 
                                min_building_size=15, max_hole_size=10, 
                                morph_kernel_size=3):
        
        if edge_prob is None:
            combined_prob = building_prob
        else:
            # Combine probabilities 
            combined_prob = building_prob * (1 - 0.5 * edge_prob)
        
        # Apply morphological operations on the probability map
     #   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        
        # Opening (erosion followed by dilation)
      #  opened_prob = cv2.morphologyEx(combined_prob, cv2.MORPH_OPEN, kernel)
        
        # Closing (dilation followed by erosion)
      #  closed_prob = cv2.morphologyEx(opened_prob, cv2.MORPH_CLOSE, kernel)
        
        # Apply median filter
        smoothed_prob = median_filter(combined_prob, size=3)
        
        # Final thresholding
        binary_mask = (smoothed_prob > building_threshold).astype(np.uint8)
        
        # Remove small objects and fill small holes
        cleaned_mask = remove_small_objects(binary_mask.astype(bool), min_size=min_building_size)
        filled_mask = remove_small_holes(cleaned_mask, area_threshold=max_hole_size)
        
        return filled_mask.astype(np.uint8)

    def save_threshold_visualizations(self, rgb_image, building_prob, edge_prob, final_mask, output_path):
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        
        # 1. RGB image
        axs[0, 0].imshow(rgb_image)
        axs[0, 0].set_title('RGB Image')
        axs[0, 0].axis('off')
        
        # 2. Building probability
        axs[0, 1].imshow(building_prob, cmap='viridis')
        axs[0, 1].set_title('Building Probability')
        axs[0, 1].axis('off')
        
        # 3. Edge probability
        if edge_prob is not None:
            axs[1, 0].imshow(edge_prob, cmap='viridis')
            axs[1, 0].set_title('Edge Probability')
        else:
            axs[1, 0].set_title('Edge Probability (Not Used)')
        axs[1, 0].axis('off')
        
        # 4. Final cleaned mask
        axs[1, 1].imshow(final_mask, cmap='binary')
        axs[1, 1].set_title('Final Cleaned Mask')
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def create_tile_inferences(
        self, image_files=None, boundaries_gdf=None, write_shp=True, parallel=True
    ):
        """
        Performs inference on a collection of image files and optionally writes the results to shapefiles.

        Parameters
        ----------
        image_files : list of Path or str
            A list containing the file paths of the images to be processed.
        boundaries_gdf : geopandas.GeoDataFrame or None
            A GeoDataFrame representing the Areas of Interest (AOI) for filtering the inference results. If None, no filtering is applied.
        write_shp : bool
            A flag indicating whether to write individual output predictions as shapefiles. If True, shapefiles are generated and saved.
        parallel : bool, optional
            A flag indicating whether to process the image files in parallel. If True (default), parallel processing is used to speed up the inference.

        Returns
        -------
        output_files : list of xarray.DataArray
            A list of DataArrays containing the inference results for each processed image file. Each DataArray in the list corresponds to the predictions for a single image file.
        """

        image_files = image_files or get_files(
            self.test_images_dir, extensions=[".tif", ".tiff"]
        )
        assert (
            image_files
        ), "No valid image files found. Make sure the directory has files and that the format is correct."

        assert (
            self.learner
        ), "The model version must be specified in the configuration settings."

        output_files = []
        if parallel:
            with ThreadPoolExecutor() as executor:
                self.learner.model.cpu()
                futures = [
                    executor.submit(
                        self.single_tile_inference,
                        image_file,
                        boundaries_gdf,
                        write_shp,
                    )
                    for image_file in image_files
                ]

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Progressing"
                ):
                    try:
                        result = future.result()
                    except Exception as e:
                        logging.error(f"An exception occurred: {e}")
                        continue
                    if result is not None:
                        output_files.append(result)
        else:
            for image in tqdm(image_files):
                output_da = self.single_tile_inference(image, boundaries_gdf, write_shp)
                if output_da is not None:
                    output_files.append(output_da)

        logging.info(f"Inference completed for {len(output_files)} tiles.")
        return output_files

    def generate_map_from_images(
        self, images_dir=None, boundaries_gdf=None, primary_key=""
    ):
        """
        Performs inference on each tile in the images directory and optionally merges the results.

        Parameters
        ----------
        images_dir : str or Path, optional
            The directory containing the image tiles. If not provided, the default test_images_dir will be used (default: None).
        boundaries_gdf : geopandas.GeoDataFrame, optional
            A GeoDataFrame representing the Area of Interest (AOI). If not provided, all tiles in the images_dir will be processed (default: None).
        primary_key : str, optional
            The primary key to use when merging outputs (default: "").

        Notes
        -----
        This function saves the inference results to disk. If boundaries_gdf is provided, the results are filtered within those boundaries before being saved.
        """

        images_dir = Path(images_dir or self.test_images_dir)
        self.crs = self.crs or self.get_crs(images_dir)
        if self.crs is None:
            raise rxr.exceptions.MissingCRS(
                "CRS could not be set. Please provide a valid images directory for CRS assignment."
            )

        if not self.generate_preds and any(self.predictions_dir.iterdir()):
            output_files = []
            preds = list(self.predictions_dir.iterdir())
            if not preds:
                raise FileNotFoundError(
                    f"No prediction files found in directory {self.predictions_dir}. Set generate_preds=True if you'd like to generate the predictions to use for the map."
                )
            logging.info(
                f"Found {len(preds)} predictions in directory {self.predictions_dir}. Loading.. "
            )
            for file in preds:
                img = rxr.open_rasterio(file)
                if img.name is None:
                    img.name = img.long_name
                output_files.append(img)

        else:

            image_files = get_files(images_dir, extensions=[".tif", ".tiff"])
            assert (
                image_files
            ), f"No valid image files found in directory {images_dir}. Make sure the directory has files and that the format is correct."
            logging.info(
                f"Found {len(image_files)} image files in directory {images_dir}. "
            )
            write_shp = boundaries_gdf is None

            output_files = self.create_tile_inferences(
                image_files, boundaries_gdf, write_shp
            )

        if len(output_files) == 0:
            raise FileNotFoundError(
                f"Couldn't find any tiles overlapping the target area. Update the set of tiles or the boundaries shapefile accordingly. Images directory: {images_dir}."
            )
            return
        if boundaries_gdf is not None:
            if boundaries_gdf.crs != self.crs:
                boundaries_gdf = boundaries_gdf.to_crs(self.crs)
            buildings_gdf = self._filter_by_areas(
                output_files, boundaries_gdf, primary_key
            )

            shapefile_path = self.shapefiles_dir / "settlement_buildings.shp"
            buildings_gdf.to_file(shapefile_path, driver="ESRI Shapefile")
            logging.info(
                "Output merged, joined with boundaries geoDataframe, and saved to disk."
            )

        logging.info("Map generation process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a map for all tiles in the test directory."
    )
    parser.add_argument(
        "-d", "--project_dir", type=str, help="The project directory.", required=True
    )  # required
    parser.add_argument(
        "-c",
        "--config_name",
        type=str,
        help="The configuration file name. If missing, the constructor will look for a single file in the project directory.",
    )  # optional
    parser.add_argument(
        "--generate_preds",
        default=False,
        action="store_true",
        help="Flag indicating whether predictions should be generated for mapping, or pulled from the predictions directory. Defaults to False.",
    )
    args = parser.parse_args()
    map_gen = MapGenerator(args.project_dir, args.config_name, args.generate_preds)
    map_gen.generate_map_from_images()
