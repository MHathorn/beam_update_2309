from pathlib import Path
import shutil

class BaseClass:

    DIR_STRUCTURE = {
        "images": "images",
        "labels": "labels",
        "image_tiles": "tiles/images",
        "mask_tiles": "tiles/masks",
        "label_tiles": "tiles/labels",
        "models": "models",
        "train_images": "tiles/train/images",
        "train_masks": "tiles/train/masks",
        "eval": "eval",
        "predictions": "predict/masks",
        "shapefiles": "predict/shapefiles",
        "test_images": "tiles/test/images",
        "test_masks": "tiles/test/masks"
    }

    def __init__(self, config, read_dirs=[], write_dirs=[]):
        # self.load_params(config)
        self.load_dir_structure(config, read_dirs, write_dirs)

    def load_params(self, config):
        pass

    def load_dir_structure(self, config, read_dirs, write_dirs):
        path = Path(config["root_dir"])
        all_dirs = set(read_dirs + write_dirs)
        for dir_name in all_dirs:
            try:
                dir_path = path / self.DIR_STRUCTURE[dir_name]
            except KeyError as e:
                raise KeyError(f"The directory key {e} is not registered in BaseClass.")
            overwrite = (dir_name in write_dirs)
            setattr(self, f"{dir_name}_dir", self.create_if_not_exists(dir_path, overwrite=overwrite))

    def load_model_path(self, config):
        model_version = config["test"]["model_version"]
        model_version_dir = self.models_dir / model_version
        if not model_version_dir.exists():
            raise ValueError(f"Couldn't find model under {model_version_dir}.")
        # Find all pickle files in the directory
        pickle_files = list(model_version_dir.glob('*.pkl'))

        # Check if there is exactly one pickle file
        if len(pickle_files) != 1:
            raise ValueError(f"Expected exactly one pickle file in {model_version_dir}, but found {len(pickle_files)}.")

        return pickle_files[0]
        

    @staticmethod
    def create_if_not_exists(dir_path, overwrite=False):
        """
        Create a directory if it does not exist. Optionally, f the directory exists and is not empty,
        files will get overwritten.

        Parameters:
        dir (PosixPath|str): The path of the directory to create.

        Returns:
        dir_path (PosixPath): The path of the created directory.

        """
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        elif overwrite and any(dir_path.iterdir()):
            print(f"Warning: {dir_path.name} directory is not empty. Overwriting files.")
            shutil.rmtree(dir_path)  # Delete the directory and its contents
            dir_path.mkdir(parents=True)
        return dir_path

