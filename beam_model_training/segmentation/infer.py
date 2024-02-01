
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import logging

import cv2
import geopandas as gpd
import numpy as np
from PIL import Image
import rasterio
from rasterio.features import geometry_mask
import rioxarray as rxr
import torch
import xarray as xr
from fastai.vision.all import load_learner
from shapely.geometry import Polygon, box

from preprocess.sample import tile_in_settlement
from segmentation.train import Trainer
from utils.base_class import BaseClass
from utils.helpers import get_rgb_channels, load_config


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
    def __init__(self, config):
        """
        Constructs all the necessary attributes for the MapGenerator object.

        Parameters
        ----------
            config : dict
                Configuration dictionary containing paths and inference arguments.
        """
        read_dirs = ["test_images", "models"]
        write_dirs = ["predictions", "shapefiles"]
        super().__init__(config, read_dirs=read_dirs, write_dirs=write_dirs)
        model_path = super().load_model_path(config)
        self.model = load_learner(model_path)
        self.erosion = config["tiling"]["erosion"]
    
    def _create_shp_from_mask(self, mask_da):
        """
        Creates a shapefile from a binary mask array and saves it to disk.         
        The function first dilates the mask with a 3x3 square kernel, which
        is the inverse of the erosion applied in preprocessing. Then, it uses the `rasterio.features.shapes` function to
        extract the shapes of the connected regions of the mask. Finally, it saves the shapes as polygons in a shapefile
        with the same name as the original image file, suffixed with "_predicted.shp".

        Parameters
        ----------
            mask_array : numpy.ndarray
                Binary mask array of the same size as the original image.
        """

        if not isinstance(mask_da, xr.DataArray) or mask_da.name is None:
            raise TypeError("mask_array must be a named DataArray.")
        
        pred_name = mask_da.name
        shp_path = self.shapefiles_dir / f"{pred_name}_predicted.shp"
        
        if mask_da is None or len(mask_da.values) == 1:
            
            empty_schema = {"geometry": "Polygon", "properties": {"id": "int"}}
            no_crs = None
            gdf = gpd.GeoDataFrame(geometry=[])
            
            gdf.to_file(
                shp_path,
                driver="ESRI Shapefile",
                schema=empty_schema,
                crs=no_crs,
            )
        

        if not hasattr(mask_da.rio, 'crs'):
            raise ValueError("mask_array does not have a CRS attached.")

        # Dilate the mask with a 3x3 square kernel. This is the inverse of the erosion applied in preprocessing
        if self.erosion:
            kernel = np.ones((3, 3), np.uint8)
            if len(mask_da.values.shape) > 2:
                mask_da.values = mask_da.values[0, :, :]
            mask_da.values = cv2.dilate(mask_da.values, kernel, iterations=1)
        shapes = rasterio.features.shapes(mask_da.values, transform=mask_da.rio.transform())
        polygons = [
            Polygon(shape[0]["coordinates"][0]) for shape in shapes
        ]
        gdf = gpd.GeoDataFrame(crs=mask_da.rio.crs, geometry=polygons)
        gdf["area"] = gdf["geometry"].area
        max_area_idx = gdf["area"].idxmax()
        gdf = gdf.drop([max_area_idx])
        # Drop shapes that are too small or too large to be a building
        gdf = gdf[(gdf["area"] > 2) & (gdf["area"] < 500000)]
        # in case the geo-dataframe is empty which means no settlements are detected
        if gdf.empty:
            empty_schema = {"geometry": "Polygon", "properties": {"id": "int"}}
            no_crs = None
            gdf = gpd.GeoDataFrame(geometry=[])
            gdf.to_file(
                shp_path,
                driver="ESRI Shapefile",
                schema=empty_schema,
                crs=no_crs,
            )
        else:
            gdf.to_file(
                shp_path, driver="ESRI Shapefile"
            )
    
    def group_tiles_by_settlement(self, mask_tiles, settlements):
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
    
        crs = mask_tiles[0].rio.crs
        data = [{'name': da.name, 'geometry': box(*da.rio.bounds())} for da in mask_tiles]

        # Create a GeoDataFrame from the list of dictionaries
        gdf = gpd.GeoDataFrame(data, crs=crs)
        if settlements.crs != crs:
            settlements = settlements.to_crs(crs)
        joined = gpd.sjoin(settlements, gdf, how="inner", op='intersects')
        grouped_tiles = joined.groupby(['OBJECTID', 'geometry'])['name'].apply(list)
    
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

        inference_path = self.predictions_dir / (image_file.stem +'_inference.tif')

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
        output_da = output_da.rio.write_crs(tile.rio.crs)
        output_da = output_da.rio.write_transform(tile.rio.transform())
        output_da.rio.to_raster(inference_path)

        # Generate shapefile
        if write_shp:
            self._create_shp_from_mask(output_da)
        return output_da


    def create_tile_inferences(self, images_dir=None, AOI_gpd=None, merge_outputs=False):
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

        images_dir = self.test_images_dir if images_dir is None else Path(images_dir)
        image_files = list(images_dir.glob('*.tif')) + list(images_dir.glob('*.tiff'))

        logging.info(f'Found {len(image_files)} image files in directory {images_dir}. ')
        logging.info('Starting tile inferences...')
        
        # output_files = []
        # for image in image_files:
        #     output_da = self.single_tile_inference(image, AOI_gpd, not merge_outputs)
        #     output_files.append(output_da)
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.single_tile_inference, image_file, AOI_gpd, (not merge_outputs)) for image_file in image_files]
            output_files = []
            for future in futures:
                result = future.result()
                if result is not None:
                    output_files.append(result)

        logging.info(f'Inference completed for {len(output_files)} tiles.')

        if merge_outputs:
            grouped_tiles = self.group_tiles_by_settlement(output_files, AOI_gpd)
            for (objectid, geometry), names in grouped_tiles.items():
                # Filter the data arrays by name in output_files
                settlement_tiles = [da for da in output_files if da.name in names]
                combined=xr.DataArray(name=objectid)
                for da in settlement_tiles:
                    combined = combined.combine_first(da)
                # Create a mask from the settlement's geometry
                mask = geometry_mask([geometry], transform=combined.rio.transform(), out_shape=combined.shape[-2:], invert=True)

                # Apply the mask to the data array
                combined = combined.where(mask, other=0)
                self._create_shp_from_mask(combined)
            logging.info('Output merged and saved to disk.')

        logging.info('Tile inference process completed.')


if __name__ == "__main__":
    config = load_config("test_config.yaml")
    map_gen = MapGenerator(config)
    map_gen.create_tile_inferences()

