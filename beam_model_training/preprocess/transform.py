import shutil
from fastai.vision.all import get_image_files

from sklearn.model_selection import train_test_split
from utils.my_paths import SEED
from utils.helpers import create_if_not_exists


import rioxarray

def get_rgb_channels(tiff_file_path):
    """
    Get RGB channels from varying format TIFF files.

    Reads a TIFF file using rioxarray and converts it to an RGB image.
    It supports RGBA images and Worldview-3 multi-band satellite images. For RGBA images, 
    the first three bands are used for RGB. For multi-band satellite images, 
    bands for red (4), green (2), and blue (1) are selected.

    Parameters:
    tiff_file_path (str): The path of the TIFF file to be converted.

    Returns:
    xarray.DataArray: The resulting RGB image.

    Raises:
    IOError: If the provided file is not in TIFF format.
    ValueError: If the number of bands in the image is unexpected.
    """

    if not tiff_file_path.suffix.lower() in ('.tif', '.tiff'):
        raise IOError(f"Expecting tiff file format for conversion ({tiff_file_path.name}).")
    
    riox_img = rioxarray.open_rasterio(tiff_file_path)
    num_bands = riox_img.shape[0]

    if num_bands == 4:  # RGBA image
        rgb_image = riox_img.sel(band=[1, 2, 3])
    elif num_bands == 8:  # Worlview-3 satellite image
        # Select bands for red (4), green (2), and blue (1)
        rgb_image = riox_img.sel(band=[4, 2, 1])
    else:
        raise ValueError("Unexpected number of bands.")

    return rgb_image


def gen_train_test(tiles_dir, test_size=0.2, seed=2022):
    """
    Splits image and mask files into training and testing sets and moves the testing files to specified directories.
        
    Parameters:
    tiles_dir (str): The directory containing the image tiles and mask tiles. Expected structure matches the output of the DataTiler:
    ├── tiles_dir
        │ ├── images
        │ │ ├── image.tiff
        │ ├── masks
        │ │ ├── mask.tiff
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.

    Returns:
    None

    Note:
    This function expects mask_files match the name of an image f ile.
    """
    
    images_dir = tiles_dir / "images"
    masks_dir = tiles_dir / "masks"

    # Ensure the source directories exist and no files are missing.
    if not images_dir.exists() or not masks_dir.exists():
        raise IOError("Source directories do not exist.")
    elif len(get_image_files(images_dir)) != len(get_image_files(masks_dir)):
        raise IOError(f"Mismatch in image and mask count (Images: {len()})")


    # Get all file names from the image directory
    image_files = [f for f in images_dir.iterdir() if f.is_file()]

    # Split the files into training and testing 
    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=seed)

    for dir_name, files in [("test", test_files), ("train", train_files)]:
        target_images_dir = create_if_not_exists(tiles_dir / dir_name / "images", overwrite=True)
        target_masks_dir = create_if_not_exists(tiles_dir / dir_name / "masks", overwrite=True)
        
        for file_path in files:
            rgb_image = get_rgb_channels(file_path)
            rgb_image.rio.to_raster(target_images_dir / file_path.name)

            mask_path = masks_dir / (file_path.name)
            if mask_path.exists():
                shutil.copy(mask_path, target_masks_dir / file_path.name)
            else:
                raise IOError(f"Couldn't find corresponding mask file for {file_path.name}.")
        
        print(f"Moved {len(files)} files to {dir_name} directories.")