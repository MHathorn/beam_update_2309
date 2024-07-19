import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import torch
import xarray as xr
from fastai.vision.all import *
from matplotlib.colors import ListedColormap
from PIL import Image
from pyproj import CRS
from rioxarray.merge import merge_arrays
from shapely.geometry import Polygon, box, LineString
from shapely.ops import orient
from scipy.stats import mode
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
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
        self.min_building_area = 2
        self.max_building_area = 30000

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
        shapes = rasterio.features.shapes(
            mask_da.values.astype(np.uint8), transform=mask_da.rio.transform()
        )
        polygons = [Polygon(shape[0]["coordinates"][0]) for shape in shapes]
        if not polygons:
            logging.warning("No polygons found in the mask.")
        gdf = gpd.GeoDataFrame(crs=self.crs, geometry=polygons)


        # simplified_count = 0
        # regularized_count = 0
        # unchanged_count = 0
        # invalid_count = 0

        # regularized_polygons = []
        # for idx, poly in tqdm(enumerate(gdf.geometry), desc="Regularizing polygons", total=len(gdf)):
        #     if poly.is_valid and not poly.is_empty and len(poly.exterior.coords) > 3:
        #         original_points = len(poly.exterior.coords)
        #         original_area = poly.area
        #         regularized_poly = self._regularize_polygon(poly)
        #         regularized_points = len(regularized_poly.exterior.coords)
        #         regularized_area = regularized_poly.area
                
        #         if regularized_poly.equals(poly):
        #             unchanged_count += 1
        #             logging.debug(f"Polygon {idx}: Unchanged ({original_points} points)")
        #         elif not regularized_poly.is_valid:
        #             invalid_count += 1
        #             logging.warning(f"Polygon {idx}: Regularization resulted in an invalid polygon. Keeping original.")
        #             regularized_poly = poly
        #         elif regularized_points < original_points:
        #             if abs(regularized_area - original_area) / original_area < 0.05:  # Less than 5% area change
        #                 simplified_count += 1
        #                 logging.debug(f"Polygon {idx}: Simplified from {original_points} to {regularized_points} points")
        #             else:
        #                 regularized_count += 1
        #                 logging.debug(f"Polygon {idx}: Regularized from {original_points} to {regularized_points} points")
        #         else:
        #             regularized_count += 1
        #             logging.debug(f"Polygon {idx}: Regularized from {original_points} to {regularized_points} points")
                
        #         regularized_polygons.append(regularized_poly)
        #     else:
        #         logging.warning(f"Polygon {idx} is invalid, empty, or has less than 4 points. Keeping original polygon.")
        #         regularized_polygons.append(poly)
        #         unchanged_count += 1

        # gdf['geometry'] = regularized_polygons

        # logging.info(f"Regularization results:")
        # logging.info(f"  Simplified: {simplified_count}")
        # logging.info(f"  Regularized: {regularized_count}")
        # logging.info(f"  Unchanged: {unchanged_count}")
        # logging.info(f"  Invalid (kept original): {invalid_count}")



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
        Private method to calculate areas for geometries in a GeoDataFrame.
        
        Args:
            gdf (GeoDataFrame): Input GeoDataFrame with geometries.
        
        Returns:
            GeoDataFrame: Input GeoDataFrame with added 'bldg_area' column.
        """
        crs = CRS(gdf.crs)
        
        if crs.is_geographic:
            gdf['bldg_area'] = gdf.geometry.to_crs({'proj':'cea'}).area
        else:
            gdf['bldg_area'] = gdf.geometry.area
        
        return gdf    
    
    def _simplify_polygon(self, polygon, tolerance=0.5):
        """Simplify a polygon using Douglas-Peucker algorithm while ensuring validity."""
        simplified = polygon.simplify(tolerance, preserve_topology=True)
        if simplified.is_valid:
            return simplified
        return polygon

    def _remove_small_angles(self, coords, min_angle):
        """Remove points that create small angles."""
        if len(coords) < 4:
            return coords
        
        new_coords = [coords[0]]
        for i in range(1, len(coords) - 1):
            v1 = coords[i] - coords[i-1]
            v2 = coords[i+1] - coords[i]
            angle = np.abs(np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2))))
            if angle > min_angle:
                new_coords.append(coords[i])
        new_coords.append(coords[-1])
        return np.array(new_coords)
    
    def _snap_to_right_angles(self, coords, snap_threshold):
        """Snap angles close to 90 degrees."""
        new_coords = [coords[0]]
        for i in range(1, len(coords) - 1):
            v1 = coords[i] - coords[i-1]
            v2 = coords[i+1] - coords[i]
            angle = np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2))) % 360
            if abs(angle - 90) < snap_threshold or abs(angle - 270) < snap_threshold:
                # Snap to right angle
                v2_snapped = np.array([-v1[1], v1[0]])  # Rotate v1 by 90 degrees
                new_point = coords[i] + v2_snapped * (np.dot(v2, v2_snapped) / np.dot(v2_snapped, v2_snapped))
                new_coords.append(new_point)
            else:
                new_coords.append(coords[i])
        new_coords.append(coords[-1])
        return np.array(new_coords)    

    def _regularize_polygon(self, polygon):
        """Regularize a polygon while ensuring it remains valid."""
        try:
            # Initial simplification
            simplified = self._simplify_polygon(polygon, tolerance=0.5)
            
            if simplified.geom_type != 'Polygon':
                logging.warning(f"Simplification resulted in a {simplified.geom_type}. Keeping original polygon.")
                return polygon

            # Get coordinates
            coords = np.array(simplified.exterior.coords)

            # Ensure we have at least 4 points
            if len(coords) < 4:
                logging.warning(f"Polygon has less than 4 points after simplification. Keeping original polygon.")
                return polygon

            # Remove small angles
            new_coords = self._remove_small_angles(coords[:-1], min_angle=10)  # Exclude last point (same as first)


            # Create new polygon
            new_polygon = Polygon(new_coords)

            if not new_polygon.is_valid:
                logging.warning("Regularization resulted in an invalid polygon. Keeping original polygon.")
                return polygon

            # Ensure the polygon has the same orientation as the original
            new_polygon = orient(new_polygon, sign=1.0)

            return new_polygon

        except Exception as e:
            logging.error(f"Error in _regularize_polygon: {str(e)}")
            return polygon



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
        edge_prob = class_probs[2, :, :]
        
        building_threshold = 0.3
        edge_threshold = 0.6

        # Apply watershed segmentation
        building_mask = self.watershed_segmentation(
            building_prob, 
            edge_prob, 
            background_prob,
            building_threshold=building_threshold,  # Increase this if you're getting too many false positives
            edge_threshold=edge_threshold,       
            min_distance=5,          # Increase this if buildings are being over-segmented
            min_size=20              # Increase this to remove smaller segmented areas
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
        self.save_threshold_visualizations(rgb_image, building_prob, edge_prob, building_mask, 
                                    building_threshold, edge_threshold, viz_path)


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


    def watershed_segmentation(self, building_prob, edge_prob, background_prob=None, 
                            building_threshold=0.3, edge_threshold=0.3, 
                            min_distance=4, min_size=15):
        # Use building probability as the base for segmentation
        building_mask = (building_prob > building_threshold).astype(np.uint8)
        
        # Apply threshold to edge probability
        strong_edges = (edge_prob > edge_threshold).astype(np.uint8)
        
        # If we have background probability, use it to refine the mask
        if background_prob is not None:
            building_mask[background_prob > building_prob] = 0
        
        # Compute distance transform on the building mask
        distance = ndimage.distance_transform_edt(building_mask)
        
        # Find local maxima in the distance transform
        coordinates = peak_local_max(distance, min_distance=min_distance, labels=building_mask)
        
        # Create a marker array for watershed
        markers = np.zeros(distance.shape, dtype=np.uint8)
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
        
        # Prepare the image for watershed
        # We invert the building probability and add the thresholded edge probability
        watershed_image = 1 - building_prob + strong_edges
        
        # Apply watershed
        labels = watershed(watershed_image, markers, mask=building_mask)
        
        # Remove small objects
        labels = remove_small_objects(labels, min_size=min_size)
        
        return labels
    
    def save_threshold_visualizations(self, rgb_image, building_prob, edge_prob, final_mask, 
                                  building_threshold, edge_threshold, output_path):
        fig, axs = plt.subplots(3, 2, figsize=(12, 18))
        
        # Original RGB image
        axs[0, 0].imshow(rgb_image)
        axs[0, 0].set_title('Original RGB Image')
        axs[0, 0].axis('off')
        
        # Final mask
        # Create a colormap for the segmentation mask
        n_labels = final_mask.max()
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_labels))
        colors = np.vstack(([0, 0, 0, 1], colors))  # Add black color for background
        custom_cmap = ListedColormap(colors)
        
        axs[0, 1].imshow(final_mask, cmap=custom_cmap)
        axs[0, 1].set_title('Final Segmentation Mask')
        axs[0, 1].axis('off')
        
        # Building probability
        axs[1, 0].imshow(building_prob, cmap='viridis')
        axs[1, 0].set_title('Building Probability')
        axs[1, 0].axis('off')
        
        # Thresholded building mask
        axs[1, 1].imshow(building_prob > building_threshold, cmap='binary')
        axs[1, 1].set_title(f'Building Mask (threshold={building_threshold})')
        axs[1, 1].axis('off')
        
        # Edge probability
        axs[2, 0].imshow(edge_prob, cmap='viridis')
        axs[2, 0].set_title('Edge Probability')
        axs[2, 0].axis('off')
        
        # Thresholded edge mask
        axs[2, 1].imshow(edge_prob > edge_threshold, cmap='binary')
        axs[2, 1].set_title(f'Strong Edges (threshold={edge_threshold})')
        axs[2, 1].axis('off')
        
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
