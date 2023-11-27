


import os
import shutil

from sklearn.model_selection import train_test_split
from utils.my_paths import SEED
from utils.helpers import create_if_not_exists, reset_directories


def train_test_split_files(tiles_dir, test_size=0.2):
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
    This function expects mask_files match the name of an image file.
    """
    
    images_dir = tiles_dir / "images"
    masks_dir = tiles_dir / "masks"
    test_images_dir = tiles_dir / "test_images"
    test_masks_dir = tiles_dir / "test_masks"

    # Ensure the source directories exist and no files are missing.
    if not images_dir.exists() or not masks_dir.exists():
        raise IOError("Source directories do not exist.")
    elif len(images_dir.iterdir()) != len(masks_dir.iterdir()):
        raise IOError(f"Mismatch in image and mask count (Images: {len()})")

    # Ensure the test directories exist, if not, create them.
    create_if_not_exists(test_masks_dir)
    create_if_not_exists(test_images_dir)

    
    reset_directories(test_images_dir, images_dir)
    reset_directories(test_masks_dir, masks_dir)

    # Get all file names from the image directory
    image_files = [f for f in images_dir.iterdir() if f.isfile()]

    # Split the files into training and testing sets
    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=SEED)

    # Move the selected image and mask files to the test directories
    for file_path in test_files:
        # Move image file
        shutil.move(file_path, test_images_dir / file_path.name)
        mask_path = masks_dir / (file_path.name)
        if mask_path.exists():
            shutil.move(mask_path, test_masks_dir / mask_path.name)
        else:
            raise IOError(f"Couldn't find corresponding mask file for {file_path.name}.")
    print(f"Moved {len(test_files)} files to test directories.")