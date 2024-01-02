
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from fastai.vision.all import get_image_files, load_learner
from natsort import os_sorted
from shapely.geometry import Polygon
from segmentation.train import Trainer
from utils.helpers import create_if_not_exists, load_config


class MapGenerator:
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
        try:
            path = Path(config["root_dir"])
            self.images_dir = path / config["dirs"]["test"] / "images"
            self.shp_dir = create_if_not_exists(path / config["dirs"]["shapefiles"], overwrite=True)
            self.predict_dir = create_if_not_exists(path / config["dirs"]["predictions"], overwrite=True)
            self.erosion = config["erosion"]

            infer_args = config["test"]
            model_path = path / config["dirs"]["models"] / infer_args["model_name"]
            if not model_path.exists():
                raise ValueError(f"Couldn't find model under {model_path}.")
            self.model = load_learner(model_path)
        except KeyError as e:
            raise KeyError(f"Config must have a value for {e}.")
    
    def _create_shp_from_mask(self, file, mask_array):
        """
        Creates a shapefile from a binary mask array and saves it to disk.         
        The function first dilates the mask with a 3x3 square kernel, which
        is the inverse of the erosion applied in preprocessing. Then, it uses the `rasterio.features.shapes` function to
        extract the shapes of the connected regions of the mask. Finally, it saves the shapes as polygons in a shapefile
        with the same name as the original image file, suffixed with "_predicted.shp".

        Parameters
        ----------
            file : str
                Path to the original image file.
            mask_array : numpy.ndarray
                Binary mask array of the same size as the original image.
        """
        
        with rasterio.open(file, "r") as src:
            raster_meta = src.meta

        pred_name = file.stem
        shp_path = self.shp_dir / f"{pred_name}_predicted.shp"
        
        if mask_array is None or len(mask_array) == 1:
            
            empty_schema = {"geometry": "Polygon", "properties": {"id": "int"}}
            no_crs = None
            gdf = gpd.GeoDataFrame(geometry=[])
            
            gdf.to_file(
                shp_path,
                driver="ESRI Shapefile",
                schema=empty_schema,
                crs=no_crs,
            )
            return
        elif not isinstance(mask_array, np.ndarray):
            mask_array = np.array(mask_array)

        # Dilate the mask with a 3x3 square kernel. This is the inverse of the erosion applied in preprocessing
        if self.erosion:
            kernel = np.ones((3, 3), np.uint8)
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        shapes = rasterio.features.shapes(mask_array, transform=raster_meta["transform"])
        polygons = [
            Polygon(shape[0]["coordinates"][0]) for shape in shapes
        ]
        my_list = raster_meta["crs"]
        gdf = gpd.GeoDataFrame(crs=my_list, geometry=polygons)
        gdf["area"] = gdf["geometry"].area
        gdf = gdf.drop([gdf["area"].idxmax()])
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

    
    def merge_tiles(self, arr, h, w):
        """
        Takes as input a NumPy array of shape (n, nrows, ncols, c) or (n, nrows, ncols) 
        and the original height and width of the image. 
        If the array has a color channel, will reshape into a 3d array of shape HWC.
        If no color channel, will reshape into array of shape: h, w.
        NB: Currently only works without color channel (NHWC wrong shape for PyTorch tensors) 
        """

        try:  # with color channel
            _, nrows, ncols, c = arr.shape # get the shape of the array
            return (
                arr.reshape(h // nrows, -1, nrows, ncols, c).swapaxes(1, 2).reshape(h, w, c) # reshape the array
            )
        except ValueError:  # without color channel
            _, nrows, ncols = arr.shape # get the shape of the array
            return arr.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w) # reshape the array


    def create_tile_inferences(self, images_dir=None):
        """
        Performs inference on each tile in the images directory and saves the results.
        """
        images_dir = self.images_dir if images_dir is None else Path(images_dir)

        for image_file in images_dir.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in ['.tif', '.tiff']:

                # Run inference and save as grayscale image
                pred, _, _ = self.model.predict(image_file) 
                output = torch.exp(pred[:, :]).detach().cpu().numpy() 

                if output.min() != output.max():
                    output = (output - output.min()) / (output.max() - output.min())

                inference_path = self.predict_dir / (image_file.stem +'_inference.tif')
                with rasterio.open(image_file) as src:
                    profile = src.profile
                    
                profile.update(dtype=rasterio.float32, count=1)
                with rasterio.open(inference_path, 'w', **profile) as dst:
                    dst.write(output.astype(rasterio.float32), 1)

                # Generate shapefile
                self._create_shp_from_mask(image_file, output)

if __name__ == "__main__":
    config = load_config("base_config.yaml")
    map_gen = MapGenerator(config)
    map_gen.create_tile_inferences()

