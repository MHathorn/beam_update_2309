import logging
import random
from datetime import datetime
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytz
import rioxarray as rxr
from tqdm import tqdm
import xarray as xr
import random
import torch
from PIL import Image
#from fastai.vision.all import set_seed, torch, PILImage


def seed(seed_value=0):
    """Seed randomization functions for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)  # Replaces fastai's set_seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def timestamp():
    """Timestamp for tracking experiments."""
    tz = pytz.timezone("Europe/Berlin")
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
        if not tiff_file_path.suffix.lower() in (".tif", ".tiff"):
            raise IOError(
                f"Expecting tiff file format for conversion ({tiff_file_path.name})."
            )

        riox_img = rxr.open_rasterio(tiff_file_path)
        riox_img.name = tiff_file_path.stem
    elif isinstance(input_data, xr.DataArray):
        riox_img = input_data
    else:
        raise ValueError(
            "Input data must be a string, pathlib.Path, or xarray.DataArray."
        )

    num_bands = riox_img.shape[0]

    if num_bands == 3:  # already RGB
        rgb_image = riox_img
    elif num_bands == 4:  # RGBA image
        rgb_image = riox_img.sel(band=[1, 2, 3])
    elif num_bands == 8:  # Worlview-3 satellite image
        # Select bands for red (4), green (2), and blue (1)
        rgb_image = riox_img.sel(band=[4, 2, 1])
    elif num_bands == 1:
        rgb_image = xr.concat([riox_img for _ in range(3)], dim="band")
    else:
        raise ValueError("Unexpected number of bands.")

    return rgb_image


def multiband_to_png(file_path, output_dir):
    """
    Converts a multiband TIFF file to a PNG file and saves it to the specified output directory.

    Parameters:
        file_path (str|PosixPath): The path to the input TIFF file.
        output_dir (PosixPath): The directory where the output PNG file will be saved.
    """
    tiff_file = Path(file_path)
    png_file = output_dir / tiff_file.with_suffix(".png").name

    # Open the TIFF file and convert it to PNG
    try:
        img = get_rgb_channels(tiff_file)
        # Normalize the image values to the 0..1 range if they are floating-point
        if img.dtype.kind == 'f':
            img = (img - img.min()) / (img.max() - img.min())
        elif img.dtype.kind in 'ui':
            img = img / np.iinfo(img.dtype).max
        plt.imsave(png_file, np.transpose(img.values, (1, 2, 0)))
    except Exception as e:
        logging.error(f"An error occurred while converting {tiff_file}: {e}")


def get_tile_size(image_path):
    """Get the size of a tile from an image file."""
    img = Image.open(image_path)  
    width, height = img.size  
    if width != height:
        raise ValueError(
            f"Tiles should be square, got dimensions: ({width}, {height})"
        )
    return width  # or height, they're the same


def copy_leaf_files(src_dir, dest_dir):
    """
    Copy all leaf files from src_dir to dest_dir without overwriting directories.
    """

    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    for src_file_path in tqdm(src_dir.rglob("*"), total=len(src_dir.rglob("*"))):
        if src_file_path.is_file():
            # Determine the relative path from the source directory
            rel_path = src_file_path.relative_to(src_dir)
            dest_path = dest_dir / rel_path

            # Ensure the destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy each file in the current directory to the destination directory
            if not dest_path.exists():
                shutil.copy2(src_file_path, dest_path)
            else:
                logging.warning(f"File already exists, not overwriting: {dest_path}")
