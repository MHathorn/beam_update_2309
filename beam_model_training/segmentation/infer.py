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
        regularization_keys = ["simplify_tolerance", "buffer_distance", "use_mbr", "mbr_threshold", "min_building_area", "max_building_area"]
        self.regularization_params = {k: self.config.get("regularization", {}).get(k) for k in regularization_keys}
        
        # Set default values if not provided
        defaults = {
            "simplify_tolerance": 1.0,
            "buffer_distance": 0.2,
            "use_mbr": True,
            "mbr_threshold": 0.85,
            "min_building_area": 2,
            "max_building_area": 30000
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

        # Dilate the mask with a 3x3 square kernel. This is the inverse of the erosion applied in preprocessing
        # if self.erosion:
        #     kernel = np.ones((3, 3), np.uint8)
        #     if len(mask_da.values.shape) > 2:
        #         mask_da = mask_da.squeeze(dim=None)
        #     mask_da.values = cv2.dilate(mask_da.values, kernel, iterations=1)

        # Extract all connected component and turn to polygons
        
        mask_values = mask_da.values
        filtered_mask = self.apply_median_filter_to_labels(mask_values, size=3)
        shapes = rasterio.features.shapes(
            filtered_mask.astype(np.uint8), transform=mask_da.rio.transform()
        )
        polygons = [Polygon(shape[0]["coordinates"][0]) for shape in shapes]
        if not polygons:
            logging.warning("No polygons found in the mask.")
        gdf = gpd.GeoDataFrame(crs=self.crs, geometry=polygons)

        # Project to an equal area projection for regularization
        if gdf.crs.is_geographic:
            gdf_projected = gdf.to_crs({'proj':'cea'})
        else:
            gdf_projected = gdf

        regularized_polygons = []
        for idx, poly in tqdm(enumerate(gdf_projected.geometry), desc="Regularizing polygons", total=len(gdf_projected)):
            if poly.is_valid and not poly.is_empty and len(poly.exterior.coords) > 3:
                dominant_angle = self._calculate_dominant_angle(poly)
                rotated = rotate(poly, -dominant_angle, origin='centroid')
                regularized = self.regularize_polygon(rotated, 
                                    simplify_tolerance=self.simplify_tolerance,
                                    buffer_distance=self.buffer_distance, 
                                    use_mbr=self.use_mbr,
                                    mbr_threshold=self.mbr_threshold)
                regularized_rotated = rotate(regularized, dominant_angle, origin='centroid')
                regularized_polygons.append(regularized_rotated)
            else:
                regularized_polygons.append(poly)

        gdf_projected['geometry'] = regularized_polygons

        # Project back to original CRS if necessary
        if gdf.crs.is_geographic:
            gdf = gdf_projected.to_crs(self.crs)
        else:
            gdf = gdf_projected



        # Calculate areas
        gdf = self._calculate_area(gdf)

        if len(gdf) > 1:
            max_area_idx = gdf["bldg_area"].idxmax()
            gdf = gdf.drop(max_area_idx)        

        # Filter by area
        gdf = gdf[(gdf["bldg_area"] > self.min_building_area) & (gdf["bldg_area"] < self.max_building_area)]

        # in case the geo-dataframe is empty which means no settlement buildings are detected
        if gdf.empty:
            logging.warning("Filtered GeoDataFrame is empty after dropping small/large areas.")
            return gpd.GeoDataFrame(geometry=[])

        return gdf

   

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
        Performs regularization after merging for better context awareness.
        """
        grouped_tiles = self._group_tiles_by_connected_area(
            output_files, boundaries_gdf, primary_key
        )
        gdfs = []
        components_list = list(grouped_tiles.items())

        for (pkey, _), names in tqdm(components_list, desc="Processing settlement areas"):
            try:
                # Merge tiles first
                settlement_tiles = [da for da in output_files if da.name in names]
                combined = merge_arrays(settlement_tiles)
                combined.name = pkey

                # Extract polygons without regularization
                mask_values = combined.values
                filtered_mask = self.apply_median_filter_to_labels(mask_values, size=3)
                shapes = rasterio.features.shapes(
                    filtered_mask.astype(np.uint8), 
                    transform=combined.rio.transform()
                )
                polygons = [Polygon(shape[0]["coordinates"][0]) for shape in shapes]
                
                if not polygons:
                    continue
                    
                # Create initial GeoDataFrame
                gdf = gpd.GeoDataFrame(crs=self.crs, geometry=polygons)
                
                # Project to equal area for regularization
                if gdf.crs.is_geographic:
                    gdf_projected = gdf.to_crs({'proj':'cea'})
                else:
                    gdf_projected = gdf

                # Find dominant angles across all buildings
                all_angles = []
                for poly in gdf_projected.geometry:
                    if isinstance(poly, (Polygon, MultiPolygon)) and poly.is_valid and not poly.is_empty:
                        try:
                            angle = self._calculate_dominant_angle(poly)
                            if angle is not None:
                                all_angles.append(angle)
                        except Exception as e:
                            logging.warning(f"Error calculating angle: {e}")
                            continue

                # Get dominant angle if we have enough buildings
                dominant_angle = None
                if len(all_angles) > 3:
                    angles_rad = np.deg2rad(all_angles)
                    hist, bins = np.histogram(angles_rad, bins=36, range=(-np.pi, np.pi))
                    dominant_angle = np.rad2deg(bins[np.argmax(hist)])

                # Regularize using settlement context
                regularized_polygons = []
                for poly in gdf_projected.geometry:
                    try:
                        if isinstance(poly, (Polygon, MultiPolygon)) and poly.is_valid and not poly.is_empty:
                            # Use settlement-wide dominant angle if available
                            angle_to_use = dominant_angle if dominant_angle is not None else self._calculate_dominant_angle(poly)
                            
                            if angle_to_use is not None:
                                # Rotate, regularize, and rotate back
                                rotated = rotate(poly, -angle_to_use, origin='centroid')
                                regularized = self.regularize_polygon(rotated)
                                regularized_rotated = rotate(regularized, angle_to_use, origin='centroid')
                                regularized_polygons.append(regularized_rotated)
                            else:
                                # If no clear angle, just regularize
                                regularized = self.regularize_polygon(poly)
                                regularized_polygons.append(regularized)
                        else:
                            regularized_polygons.append(poly)
                    except Exception as e:
                        logging.warning(f"Error regularizing polygon: {e}")
                        regularized_polygons.append(poly)

                gdf_projected['geometry'] = regularized_polygons

                # Project back if necessary
                if gdf.crs.is_geographic:
                    gdf = gdf_projected.to_crs(self.crs)
                else:
                    gdf = gdf_projected

                # Calculate areas and filter
                gdf = self._calculate_area(gdf)
                gdf = gdf[
                    (gdf["bldg_area"] > self.min_building_area) & 
                    (gdf["bldg_area"] < self.max_building_area)
                ]
                
                if not gdf.empty:
                    gdfs.append(gdf)
            except Exception as e:
                logging.error(f"Error processing settlement {pkey}: {e}")
                continue

        if not gdfs:
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)

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
    
    def regularize_polygon(self, polygon, simplify_tolerance=None, buffer_distance=None, 
                        use_mbr=None, mbr_threshold=None):
        """
        Regularize polygon using simplification, smoothing and MBR where appropriate.
        """
        # Use class attributes as defaults
        simplify_tolerance = simplify_tolerance or self.simplify_tolerance
        buffer_distance = buffer_distance or self.buffer_distance
        use_mbr = use_mbr if use_mbr is not None else self.use_mbr
        mbr_threshold = mbr_threshold or self.mbr_threshold

        if polygon.is_empty or polygon.area == 0:
            return polygon

        # Handle MultiPolygons
        if isinstance(polygon, MultiPolygon):
            regularized_parts = [self.regularize_polygon(p, 
                                                    simplify_tolerance, 
                                                    buffer_distance,
                                                    use_mbr,
                                                    mbr_threshold) for p in polygon.geoms]
            return MultiPolygon(regularized_parts)

        try:
            # Initial simplification and smoothing
            simplified = polygon.simplify(simplify_tolerance, preserve_topology=True)
            smoothed = simplified.buffer(-buffer_distance).buffer(buffer_distance)
            filled = smoothed.buffer(0)

            # Handle case where buffer operations create a MultiPolygon
            if isinstance(filled, MultiPolygon):
                filled = max(filled.geoms, key=lambda x: x.area)

            if not filled.is_valid or filled.is_empty:
                return polygon

            # Check if MBR would be appropriate
            if use_mbr and filled.area > 0:
                mbr = filled.minimum_rotated_rectangle
                if mbr.area > 0:
                    area_ratio = filled.area / mbr.area
                    if area_ratio > mbr_threshold:
                        return mbr

            return filled
            
        except Exception as e:
            logging.warning(f"Error during polygon regularization: {e}. Returning original polygon.")
            return polygon
        
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
            
            building_mask = self.edge_guided_segmentation(
                building_prob, 
                edge_prob
            )


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
                                building_threshold=None, edge_threshold=None, 
                                min_building_size=None, max_hole_size=None, 
                                morph_kernel_size=None):
        """
        Apply edge-guided segmentation to building probability map.
        Uses class attributes as defaults if parameters not provided.
        """
        # Use class attributes as defaults
        building_threshold = building_threshold or self.building_threshold
        edge_threshold = edge_threshold or self.edge_threshold
        min_building_size = min_building_size or self.min_building_size
        max_hole_size = max_hole_size or self.max_hole_size
        morph_kernel_size = morph_kernel_size or self.morph_kernel_size
        
        if edge_prob is None:
            combined_prob = building_prob
        else:
            # Create edge mask using threshold
            edge_mask = edge_prob > edge_threshold
            # Suppress building probabilities where edges are detected
            combined_prob = building_prob * (1 - edge_mask)
        
        # Apply median filter for smoothing
        smoothed_prob = median_filter(combined_prob, size=3)
        
        # Final thresholding
        binary_mask = (smoothed_prob > building_threshold).astype(np.uint8)
        
        # Remove small objects and fill small holes
        cleaned_mask = remove_small_objects(binary_mask.astype(bool), min_size=min_building_size)
        filled_mask = remove_small_holes(cleaned_mask, area_threshold=max_hole_size).astype(np.uint8)
        
        # Create morphological kernels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Apply morphological opening to remove small protrusions
        opened = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply morphological closing to fill small gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Final dilation to slightly expand building footprints
        dilated = cv2.dilate(closed, small_kernel, iterations=1)
        
        return dilated

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
    
    def _validate_boundaries(self, boundaries_gdf, primary_key):
        """
        Validate boundaries GeoDataFrame and primary key before processing.
        
        Parameters
        ----------
        boundaries_gdf : gpd.GeoDataFrame
            GeoDataFrame containing settlement boundaries
        primary_key : str
            Column name to use as primary key
            
        Raises
        ------
        ValueError
            If primary key is not found in boundaries_gdf columns
        """
        if primary_key not in boundaries_gdf.columns:
            available_columns = ', '.join(boundaries_gdf.columns)
            raise ValueError(
                f"The primary_key '{primary_key}' is not present in the boundaries_gdf columns. "
                f"Available columns are: {available_columns}"
            )

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
        
        if boundaries_gdf is not None:
            if boundaries_gdf.empty:
                raise ValueError("The boundaries GeoDataFrame is empty.")
            self._validate_boundaries(boundaries_gdf, primary_key)

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
