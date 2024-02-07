
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import torch
import xarray as xr
from fastai.vision.all import load_learner
from PIL import Image
from shapely import vectorized
from shapely.geometry import Polygon, box
from tqdm import tqdm

from preprocess.sample import tile_in_settlement
from segmentation.train import Trainer
from utils.base_class import BaseClass
from utils.helpers import get_rgb_channels, load_config

logging.basicConfig(level=logging.INFO)


class MapGenerator(BaseClass):
    """
    A class used to generate maps from images using a trained model.

    ...

    Parameters
    ----------
    config: dict
        - root_dir : str
        The root directory containing all training files. 
            - images : str
            The directory where the image tiles are stored.
            - shapefiles : str
            The directory where the shapefiles will be saved.
            - predictions : str
            The directory where the prediction files will be saved.
            - models : str
            The directory where model checkpoints are saved.
        - erosion : bool
        Whether erosion has been applied to building labels in preprocessing.
        - test: dict
            - model_name : PyTorch model
            The trained model used for inference.

    Methods
    -------
    _create_shp_from_mask(file, mask_array):
        Creates a shapefile from a binary mask array and saves it to disk.
    create_tile_inferences():
        Performs inference on each tile in the images directory and saves the results.
    """
    def __init__(self, config, overwrite=False):
        """
        Constructs all the necessary attributes for the MapGenerator object.

        Parameters
        ----------
            config : dict
                Configuration dictionary containing paths and inference arguments.
        """
        read_dirs = ["test_images", "models"]
        prediction_dirs = ["predictions", "shapefiles"]
        if overwrite:
            super().__init__(config, read_dirs=read_dirs, write_dirs=prediction_dirs)
        else:
            super().__init__(config, read_dirs=(read_dirs + prediction_dirs))
        model_path = super().load_model_path(config)
        self.model = load_learner(model_path)
        self.crs = self.get_crs(self.test_images_dir)
        self.erosion = config["tiling"]["erosion"]
    

    def get_crs(self, images_dir):
        images_dir_path = Path(images_dir)
        if any(images_dir_path.iterdir()):
            # Get the first image file
            img_file = next(images_dir_path.glob('*'))  # Adjust the pattern as needed
            img = rxr.open_rasterio(img_file)
            return img.rio.crs

    def _create_shp_from_mask(self, mask_da, primary_key='shape_id'):
        """
        Creates a GeoDataFrame from a binary mask array.
        The function first dilates the mask with a 3x3 square kernel, which
        is the inverse of the erosion applied in preprocessing. Then, it uses the `rasterio.features.shapes` function to
        extract the shapes of the connected regions of the mask.

        Parameters
        ----------
            mask_array : numpy.ndarray
                Binary mask array of the same size as the original image.
        Returns
        -------
            gdf: geopandas.GeoDataFrame
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
        shapes = rasterio.features.shapes(mask_da.values, transform=mask_da.rio.transform())
        polygons = [Polygon(shape[0]["coordinates"][0]) for shape in shapes]

        gdf = gpd.GeoDataFrame(crs=self.crs, geometry=polygons)
        gdf[primary_key] = mask_da.name
        gdf["bldg_area"] = gdf["geometry"].area
        max_area_idx = gdf["bldg_area"].idxmax()
        gdf = gdf.drop([max_area_idx])
        # Drop shapes that are too small or too large to be a building
        gdf = gdf[(gdf["bldg_area"] > 2) & (gdf["bldg_area"] < 500000)]
        # in case the geo-dataframe is empty which means no settlements are detected
        if gdf.empty:
            return gpd.GeoDataFrame(geometry=[])

        return gdf

    
    def group_tiles_by_settlement(self, mask_tiles, settlements, primary_key):
        """
        Groups tiles by their corresponding settlement based on geographic coordinates.

    
        Parameters:
        - mask_tiles (list): List of Data Arrays representing the tiles to be potentially merged.
        - settlements (GeoDataFrame): A GeoDataFrame representing the settlements.
    
        Returns
        -------
        grouped_tiles : pandas.Series
            A series where each entry is a list of tile names that cover a unique settlement. 
            The index of the series are tuples containing the OBJECTID and geometry of the settlement.

        """
        data = [{'name': da.name, 'geometry': box(*da.rio.bounds())} for da in mask_tiles]

        # Create a GeoDataFrame from the list of dictionaries
        gdf = gpd.GeoDataFrame(data, crs=self.crs)
        joined = gpd.sjoin(settlements, gdf, how="inner", op='intersects')
        grouped_tiles = joined.groupby([primary_key, 'geometry'])['name'].apply(list)
    
        return grouped_tiles
    
        

    def single_tile_inference(self, image_file, AOI_gpd=None, write_shp=True):
        """
        Process a single image file for inference.
        Parameters:
        - image_file (str): Path to the image file.
        - AOI_gpd (GeoDataFrame): Area of Interest as a GeoPandas DataFrame.

        This function opens the image file, checks if it is within the Area of Interest if exists,
        runs inference on the image, normalizes the output, saves the output as a new 
        GeoTIFF file, and generates a shapefile from the output mask.
        """
        tile = get_rgb_channels(image_file)

        if AOI_gpd is not None and not tile_in_settlement(tile, AOI_gpd):
            output = np.zeros_like(tile.data.transpose(1, 2, 0)[:,:,0])
        else:
            # Run inference and save as grayscale image
            image = Image.fromarray(tile.data.transpose(1, 2, 0))
            pred, _, _ = self.model.predict(image)
            output = torch.exp(pred[:, :]).detach().cpu().numpy() 

            output_min = output.min()
            output_max = output.max()

            output = np.zeros_like(output) if output_min == output_max else (output - output_min) / (output_max - output_min)

        inference_path = self.predictions_dir / f"{image_file.stem}_inference.tif"

        # Create a DataArray from the output and assign the coordinate reference system and affine transform from original tile
        output_da = xr.DataArray(
            name=str(image_file.stem),
            data=output,
            dims=["y", "x"],
            coords={
                "y": tile.coords["y"],
                "x": tile.coords["x"]
            }
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
    
    def filter_by_areas(self, output_files, settlements, primary_key):
        """
        Filters geospatial data arrays by settlement areas and merges them into a single GeoDataFrame.

        Groups and merges tiles based on settlement boundaries, applies a mask to retain data 
        within these boundaries, and enriches the resulting GeoDataFrame with attributes from `settlements`.

        Parameters:
        - output_files (list of xarray.DataArray): Geospatial data arrays with a 'name' attribute.
        - settlements (geopandas.GeoDataFrame): Settlement areas with geometries and attributes.
        - primary_key (str): Column in `settlements` for unique settlement identification.

        Returns:
        - geopandas.GeoDataFrame: Merged and masked geospatial data for each settlement, enriched with 
        settlement attributes.
        """
    
        if settlements.crs != self.crs:
            settlements = settlements.to_crs(self.crs)
        grouped_tiles = self.group_tiles_by_settlement(output_files, settlements, primary_key)
        gdfs = []
        for (pkey, geometry), names in grouped_tiles.items():
            # Filter the data arrays by name in output_files
            settlement_tiles = [da for da in output_files if da.name in names]

            combined = merge_arrays(settlement_tiles)
            combined.name = pkey


            # Create a mask from the settlement's geometry
            minx, miny, maxx, maxy = combined.rio.bounds()
            # Correcting for flipped bounds in certain geometries
            if maxy < miny:
                miny, maxy = maxy, miny
            if maxx < minx:
                minx, maxx = maxx, minx
            height, width = combined.shape[-2:]
            y, x = np.mgrid[miny:maxy:height*1j, minx:maxx:width*1j]
            mask = vectorized.contains(geometry, x, y)

            # Apply the mask to the data array
            combined = combined.where(mask, other=0)
            gdf = self._create_shp_from_mask(combined, primary_key)
            gdfs.append(gdf)
        all_buildings = gpd.GeoDataFrame(pd.concat(gdfs), geometry='geometry', crs=self.crs)
        return all_buildings.merge(settlements.drop('geometry', axis=1), on=primary_key)


    def create_tile_inferences(self, images_dir=None, settlements=None, primary_key=''):
        """
        Performs inference on each tile in the images directory and optionally merges the results.

        Parameters
        ----------
        images_dir : str or Path, optional
            The directory containing the image tiles. If not provided, the default test_images_dir will be used.
        AOI_gpd : geopandas.GeoDataFrame, optional
            A GeoDataFrame representing the Area of Interest (AOI). If not provided, all tiles in the images_dir will be processed.
        merge_outputs : bool, optional
            If True, the outputs from each tile will be merged into a single output. If False, each tile's output will be kept separate.

        Returns
        -------
        None

        Notes
        -----
        This function saves the inference results to disk. If merge_outputs is True, the results are grouped by settlement before being saved.
        """

        

        
        
        output_files = []
        if any(self.predictions_dir.iterdir()):
            preds = list(self.predictions_dir.iterdir())
            logging.info(f'Found {len(preds)} predictions in directory {self.predictions_dir}. Loading.. ')
            output_files = [rxr.open_rasterio(file) for file in preds]
        else:
            
            logging.info('Starting tile inferences...')
            images_dir = self.test_images_dir if images_dir is None else Path(images_dir)
            image_files = list(images_dir.glob('*.tif')) + list(images_dir.glob('*.tiff'))
            logging.info(f'Found {len(image_files)} image files in directory {images_dir}. ')
            write_shp = True if settlements is None else False
            for image in tqdm(image_files):
                output_da = self.single_tile_inference(image, settlements, write_shp)
                output_files.append(output_da)
        # with ProcessPoolExecutor() as executor:
        #     futures = [executor.submit(self.single_tile_inference, image_file, settlements, (not merge_outputs)) for image_file in image_files]
        #     for future in futures:
        #         result = future.result()
        #         if result is not None:
        #             output_files.append(result)

            logging.info(f'Inference completed for {len(output_files)} tiles.')

        if settlements is not None:
            buildings_gdf = self.filter_by_areas(output_files, settlements, primary_key)
            
            shapefile_path = self.shapefiles_dir / "settlement_buildings.shp"
            buildings_gdf.to_file(
                shapefile_path,
                driver="ESRI Shapefile"
            )
            logging.info('Output merged, joined with settlements data, and saved to disk.')

        logging.info('Tile inference process completed.')


if __name__ == "__main__":
    config = load_config("test_config.yaml")
    map_gen = MapGenerator(config)
    map_gen.create_tile_inferences()

