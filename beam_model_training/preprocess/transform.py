import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from utils.helpers import get_rgb_channels
from utils.base_class import BaseClass




def gen_train_test(root_dir, test_size=0.2, seed=2022):
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
    masks_dir = root_dir / BaseClass.DIR_STRUCTURE["mask_tiles"]

    # Ensure the source directories exist and no files are missing.
    if not images_dir.exists() or not masks_dir.exists():
        raise IOError("Source directories do not exist.")
    
    # Get all file names from the image directory
    image_files = list(images_dir.glob('*'))

    if len(image_files) != len(list(masks_dir.glob('*'))):
        raise ValueError(f"Mismatch in image and mask count (Images: {len()})")

    # Split the files into training and testing 
    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=seed)

    for dir_name, files in [("test", test_files), ("train", train_files)]:
        target_images_dir = BaseClass.create_if_not_exists(root_dir / BaseClass.DIR_STRUCTURE[f"{dir_name}_images"], overwrite=True)
        target_masks_dir = BaseClass.create_if_not_exists(root_dir / BaseClass.DIR_STRUCTURE[f"{dir_name}_masks"], overwrite=True)
        
        for file_path in files:
            rgb_image = get_rgb_channels(file_path)
            rgb_image.rio.to_raster(target_images_dir / file_path.name)

            mask_path = masks_dir / (file_path.name)
            if mask_path.exists():
                shutil.copy(mask_path, target_masks_dir / file_path.name)
            else:
                raise IOError(f"Couldn't find corresponding mask file for {file_path.name}.")
        
        print(f"Moved {len(files)} files to {dir_name} directories.")