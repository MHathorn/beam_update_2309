from fastai.vision.all import *
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from utils.helpers import get_rgb_channels
from utils.base_class import BaseClass


class AddWeightsToTargets(Transform):
    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)

    def encodes(self, x: PILMask):
        # Load the corresponding weights file
        weights_file = self.weights_dir / x.name
        weights = PILMask.create(weights_file)
        stacked = torch.stack([tensor(x), tensor(weights).squeeze(0)], dim=0)
        print("Shape after stacking in transform:", stacked.shape)
        return stacked


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


def gen_train_test(root_dir, test_size=0.2, seed=2022, distance_weighting=False):
    """
    Splits image and mask files into training and testing sets and moves the testing files to specified directories.

    Parameters:
    root_dir (str): The directory containing all project tiles.
    dir_structure (dict): Dictionary representing the relative file structure. It should include the following entries:
      - image_tiles: Directory containing all image tiles.
      - mask_tiles: Directory containing all mask_tiles.
      - train: Train files directory.
      - test: Test files directory.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.

    Returns:
    None

    Note:
    This function expects mask_files match the name of an image f ile.
    """
    root_dir = Path(root_dir)
    images_dir = root_dir / BaseClass.DIR_STRUCTURE["image_tiles"]

    # Ensure the source directories exist and no files are missing.
    if not images_dir.exists():
        raise IOError(f"Images directory does not exist.")

    # Get all file names from the image directory
    image_files = list(images_dir.glob("*"))

    # Split the files into training and testing
    train_files, test_files = train_test_split(
        image_files, test_size=test_size, random_state=seed
    )

    tile_types = ["image", "mask"]
    if distance_weighting:
        tile_types.append("weight")

    for dir_name, files in [("test", test_files), ("train", train_files)]:
        for tile in tile_types:
            source_dir = root_dir / BaseClass.DIR_STRUCTURE[f"{tile}_tiles"]
            target_dir = BaseClass.create_if_not_exists(
                root_dir / BaseClass.DIR_STRUCTURE[f"{dir_name}_{tile}s"],
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

            print(f"Moved {len(files)} files to '{dir_name}_{tile}s' directory.")
