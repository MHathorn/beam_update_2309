import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from fastai.vision.all import load_learner
from PIL import Image
from rioxarray.merge import merge_arrays
from shapely.geometry import Polygon, box
from tqdm import tqdm

from preprocess.sample import tile_in_settlement
from segmentation.train import Trainer
from utils.base_class import BaseClass
from utils.helpers import get_rgb_channels, seed

logging.basicConfig(level=logging.INFO)


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
    _create_shp_from_mask(mask_da):
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
        self, project_dir, config_name="project_config.yaml", generate_preds=False
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
                Name of the config file. Defaults to project_config.yaml.
        """

        self.root_dir = super()._set_project_dir(project_dir)
        config = super().load_config(self.root_dir / config_name)
        seed(config["seed"])
        read_dirs = ["test_images", "models"]
        prediction_dirs = ["predictions", "shapefiles"]
        self.generate_preds = generate_preds
        if self.generate_preds:
            super().__init__(
                self.root_dir, read_dirs=read_dirs, write_dirs=prediction_dirs
            )
        else:
            super().__init__(self.root_dir, read_dirs=(read_dirs + prediction_dirs))
        model_path = super().load_model_path(self.root_dir, config["model_version"])
        self.model = load_learner(model_path)
        self.crs = self.get_crs(self.test_images_dir)
        self.erosion = config["tiling"]["erosion"]

    def get_crs(self, images_dir):
        images_dir_path = Path(images_dir)
        if any(images_dir_path.iterdir()):
            # Get the first image file
            img_file = next(images_dir_path.glob("*"))  # Adjust the pattern as needed
            img = rxr.open_rasterio(img_file)
            print(f"CRS now initialized to {img.rio.crs}.")
            return img.rio.crs
        else:
            print("Warning: CRS could not be set at initialization.")

    def _get_image_files(self, images_dir):
        """
        Retrieve a list of image files from the specified directory.

        This method searches for files with .TIF or .TIFF extensions within the given directory.
        If no such files are found, it raises a FileNotFoundError.

        Args:
            images_dir (Path): A Path object representing the directory to search for image files.

        Returns:
            list: A list of Path objects representing the image files found in the directory.
        """
        image_files = list(images_dir.glob("*.TIF")) + list(images_dir.glob("*.TIFF"))
        if not image_files:
            raise FileNotFoundError(
                f"No valid image files found in directory {images_dir}. Make sure the directory has files and that the format is correct."
            )
        logging.info(
            f"Found {len(image_files)} image files in directory {images_dir}. "
        )
        return image_files

    def _create_shp_from_mask(self, mask_da):
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
            return gpd.GeoDataFrame(geometry=[])

        if mask_da.rio.crs is None:
            raise ValueError("mask_array does not have a CRS attached.")

        # Dilate the mask with a 3x3 square kernel. This is the inverse of the erosion applied in preprocessing
        if self.erosion:
            kernel = np.ones((3, 3), np.uint8)
            if len(mask_da.values.shape) > 2:
                mask_da = mask_da.squeeze(dim=None)
            mask_da.values = cv2.dilate(mask_da.values, kernel, iterations=1)
        shapes = rasterio.features.shapes(
            mask_da.values, transform=mask_da.rio.transform()
        )
        polygons = [Polygon(shape[0]["coordinates"][0]) for shape in shapes]

        gdf = gpd.GeoDataFrame(crs=self.crs, geometry=polygons)
        gdf["bldg_area"] = gdf["geometry"].area
        max_area_idx = gdf["bldg_area"].idxmax()
        gdf = gdf.drop([max_area_idx])
        # Drop shapes that are too small or too large to be an informal settlement building
        gdf = gdf[(gdf["bldg_area"] > 2) & (gdf["bldg_area"] < 30000)]
        # in case the geo-dataframe is empty which means no settlement buildings are detected
        if gdf.empty:
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

            gdf = self._create_shp_from_mask(combined)
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

        if boundaries_gdf is not None and not tile_in_settlement(tile, boundaries_gdf):
            return
        else:
            # Run inference and save as grayscale image
            image = Image.fromarray(tile.data.transpose(1, 2, 0))
            pred, _, _ = self.model.predict(image)
            output = torch.exp(pred[:, :]).detach().cpu().numpy()

            output_min = output.min()
            output_max = output.max()

            output = (
                np.zeros_like(output)
                if output_min == output_max
                else (output - output_min) / (output_max - output_min)
            )

        inference_path = self.predictions_dir / f"{image_file.stem}_inference.tif"

        # Create a DataArray from the output and assign the coordinate reference system and affine transform from original tile
        output_da = xr.DataArray(
            name=str(image_file.stem),
            data=output,
            dims=["y", "x"],
            coords={"y": tile.coords["y"], "x": tile.coords["x"]},
        )
        output_da = output_da.rio.write_crs(self.crs)
        output_da = output_da.rio.write_transform(tile.rio.transform())
        output_da.rio.to_raster(inference_path)

        # Generate shapefile
        if write_shp:
            shp_path = self.shapefiles_dir / f"{image_file.stem}_predicted.shp"
            vector_df = self._create_shp_from_mask(output_da)
            vector_df.to_file(
                shp_path,
                driver="ESRI Shapefile",
            )
        return output_da

    def create_tile_inferences(
        self, image_files=None, boundaries_gdf=None, write_shp=False, parallel=True
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

        image_files = image_files or self._get_image_files(self.test_images_dir)
        output_files = []
        if parallel:
            with ProcessPoolExecutor() as executor:
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
                        print(f"An exception occurred: {e}")
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

        images_dir = Path(images_dir) or self.test_images_dir
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

            image_files = self._get_image_files(images_dir)
            write_shp = boundaries_gdf is None

            output_files = self.create_tile_inferences(
                image_files, boundaries_gdf, write_shp
            )

        if len(output_files) == 0:
            raise FileNotFoundError(
                "Couldn't find any tiles overlapping the target area. Update the set of tiles or the boundaries shapefile accordingly."
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

        logging.info("Tile inference process completed.")


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
        default="project_config.yaml",
        help="The configuration file name. Defaults to 'project_config.yaml'.",
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
