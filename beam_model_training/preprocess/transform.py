import logging
#from fastai.vision.all import *
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from utils.helpers import get_rgb_channels
from utils.base_class import BaseClass


def count_buildings(buildings, tile_geom):
    """
    Calculate the number of buildings for a given tile.

    Args:
    - buildings (GeoDataframe): Dataset of building labels.
    - tile_geom (Polygon): The geometry of the tile.

    Returns:
    - number of buildings
    """
    # Count buildings in the tile
    buildings_in_tile = buildings[buildings.intersects(tile_geom)]
    num_buildings = len(buildings_in_tile)

    return num_buildings


def calculate_average_confidence(buildings, tile_geom):
    """
    Calculate the average confidence score for buildings within a given tile.

    Args:
    - buildings (GeoDataframe): Dataset of building labels.
    - tile_geom (Polygon): The geometry of the tile.

    Returns:
    - float: Average confidence score for the tile.
    """
    buildings_in_tile = buildings[buildings.intersects(tile_geom)]
    return buildings_in_tile["confidence"].mean()


def gen_train_test(project_dir, test_size=0.2, val_size=0, seed=2022, distance_weighting=False):
    """
    Splits image and mask files into training and testing sets with optional validation set based on the specified test size and seed,
    and moves the testing files to separate directories.

    Args:
        project_dir (str or Path): The directory containing all project tiles.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        val_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.1.
        seed (int, optional): The random seed used for splitting the data. Defaults to 2022.
        distance_weighting (bool, optional): Whether to include distance weighting files in the split. Defaults to False.

    Raises:
        IOError: If the images directory does not exist or a corresponding mask/weight file is missing.

    Returns:
        None: This function performs file operations and does not return any value.

    Note:
        This function expects mask and weight files to match the name of an image file.
    """
    project_dir = Path(project_dir)
    images_dir = project_dir / BaseClass.DIR_STRUCTURE["image_tiles"]

    # Ensure the source directories exist and no files are missing.
    if not images_dir.exists():
        raise IOError(f"Images directory does not exist.")

    # Get all file names from the image directory
    image_files = list(images_dir.glob("*"))

    # Split the files into training and testing
    train_files, test_files = train_test_split(
        image_files, test_size=test_size, random_state=seed
    )

    # Optional: split training data into train/val if val_size > 0
    if val_size > 0:
        train_files, val_files = train_test_split(
            train_files, test_size=val_size, random_state=seed
        )
        splits = [("train", train_files), ("val", val_files), ("test", test_files)]
    else:
        splits = [("train", train_files), ("test", test_files)]

    tile_types = ["image", "mask"]
    if distance_weighting:
        tile_types.append("weight")

    for split_name, files in splits:
        for tile in tile_types:
            source_dir = project_dir / BaseClass.DIR_STRUCTURE[f"{tile}_tiles"]
            target_dir = BaseClass.create_if_not_exists(
                project_dir / BaseClass.DIR_STRUCTURE[f"{split_name}_{tile}s"],
                overwrite=True,
            )

            for file_path in files:
                if tile == "image":
                    rgb_image = get_rgb_channels(file_path)
                    rgb_image.rio.to_raster(target_dir / file_path.name)
                else:
                    source_path = source_dir / (file_path.name)
                    if source_path.exists():
                        shutil.copy(source_path, target_dir / file_path.name)
                    else:
                        raise IOError(
                            f"Couldn't find corresponding {tile} file for {file_path.name}."
                        )

            logging.info(f"Moved {len(files)} files to '{split_name}_{tile}s' directory.")
