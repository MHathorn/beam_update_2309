import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytz
import rioxarray as rxr
import xarray as xr
import yaml
from fastai.vision.all import set_seed, torch, PILImage


def seed(seed_value=0):
    """Seed randomization functions for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    set_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_name):
    """
    This function loads a configuration file and updates it with the values from a base configuration file.

    Parameters:
    config_name (str): The name of the configuration file to load.

    Returns:
    dict: A dictionary containing the loaded configuration.
    """
    default_config = Path(__file__).parents[1] / "configs" / "test_config.yaml"
    config_path = default_config.parent / config_name

    if not default_config.exists():
         raise ImportError("Configs default file not found. Make sure the configs/ directory location is correct and test_config exists.")
    if not config_path.exists():
        raise IOError(f"Config file {config_path} not found. Make sure the name is correct.")
    
    with open(default_config) as default_file, open(config_path) as config_file:
        config = yaml.safe_load(default_file)
        config.update(yaml.safe_load(config_file))

    return config

def timestamp():
    """Timestamp for conducting experiments"""
    tz = pytz.timezone('Europe/Berlin')
    now = datetime.now(tz)
    date_time = now.strftime("%Y%m%d-%H%M")
    return date_time


def crs_to_pixel_coords(x, y, transform):
    """
    Convert CRS coordinates to pixel coordinates.
    
    Parameters:
    x, y: Coordinates in CRS.
    transform: A rasterio.Affine object that defines a coordinate transformation.
               This should be the .transform attribute of your rasterio/xarray dataset.
               
    Returns:
    px, py: Coordinates in pixel space.
    """
    px, py = ~transform * (x, y)
    return int(px), int(py)



def get_rgb_channels(input_data):
    """
    Get RGB channels from varying format TIFF files or directly from a DataArray.

    Reads a TIFF file using rioxarray and converts it to an RGB image.
    It supports RGBA images and Worldview-3 multi-band satellite images. For RGBA images, 
    the first three bands are used for RGB. For multi-band satellite images, 
    bands for red (4), green (2), and blue (1) are selected.

    Parameters:
    input_data (str, pathlib.Path, or xarray.DataArray): The the TIFF file (or path thereof) to be converted
        or a pre-loaded DataArray.

    Returns:
    xarray.DataArray: The resulting RGB image.

    Raises:
    IOError: If the provided file is not in TIFF format.
    ValueError: If the number of bands in the image is unexpected.
    """

    if isinstance(input_data, (str, Path)):
        tiff_file_path = Path(input_data)
        if not tiff_file_path.suffix.lower() in ('.tif', '.tiff'):
            raise IOError(f"Expecting tiff file format for conversion ({tiff_file_path.name}).")
        
        riox_img = rxr.open_rasterio(tiff_file_path)
    elif isinstance(input_data, xr.DataArray):
        riox_img = input_data
    else:
        raise ValueError("Input data must be a string, pathlib.Path, or xarray.DataArray.")
    
    num_bands = riox_img.shape[0]

    if num_bands == 3: # already RGB
        rgb_image = riox_img
    elif num_bands == 4: # RGBA image
        rgb_image = riox_img.sel(band=[1, 2, 3])
    elif num_bands == 8:  # Worlview-3 satellite image
        # Select bands for red (4), green (2), and blue (1)
        rgb_image = riox_img.sel(band=[4, 2, 1])
    elif num_bands == 1:
        rgb_image = xr.concat([riox_img for _ in range(3)], dim='band')
    else:
        raise ValueError("Unexpected number of bands.")

    return rgb_image


def multiband_to_png(file_path, output_dir):
    tiff_file = Path(file_path)
    png_file = output_dir / tiff_file.with_suffix('.png').name
    

    # Open the TIFF file and convert it to PNG
    try:
        img = get_rgb_channels(tiff_file)
        plt.imsave(png_file, np.transpose(img.values, (1,2,0)))   
        print(f"Converted {tiff_file} to {png_file}")
    except Exception as e:
        print(f"An error occurred while converting {tiff_file}: {e}")


def get_tile_size(image_path):
    img = PILImage.create(image_path)

    if img.shape[0] != img.shape[1]:
        raise ValueError(f"Tiles should be of shape (tile_size, tile_size): {img.shape}")

    return img.shape[0]
