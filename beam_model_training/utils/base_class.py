import logging
from pathlib import Path
import shutil
import yaml


class BaseClass:
    """
    BaseClass is a utility class for managing directory structures throughout the codebase.

    It provides methods to set up directories based on the configuration file, and ensures that
    the required directories exist or are created as needed. It also supports loading model paths
    from specified directories.

    Attributes:
        DIR_STRUCTURE (dict): A class attribute that defines the standard directory structure.
    """

    DIR_STRUCTURE = {
        "images": "images",
        "labels": "labels",
        "pretrained": "pretrained",
        "image_tiles": "tiles/images",
        "mask_tiles": "tiles/masks",
        "weight_tiles": "tiles/weights",
        "models": "models",
        "train_images": "tiles/train/images",
        "train_masks": "tiles/train/masks",
        "train_weights": "tiles/train/weights",
        "eval": "eval",
        "predictions": "predict/masks",
        "shapefiles": "predict/shapefiles",
        "test_images": "tiles/test/images",
        "test_masks": "tiles/test/masks",
        "test_weights": "tiles/test/weights",
    }

    def __init__(self, project_dir, config_name, read_dirs=[], write_dirs=[]):
        """
        Initializes the BaseClass with the given configuration.

        Parameters:
            project_dir (PosixPath): Path to the project directory.
            read_dirs (list): List of directory keys to be read from.
            write_dirs (list): List of directory keys to write in. Those directories will be overwritten if files already exist in them.
        """
        self.load_dir_structure(project_dir, read_dirs, write_dirs)
        self.project_dir = self.set_project_dir(project_dir)
        self.config_name = self.resolve_config_name(config_name)
        self.config = self.load_config(project_dir / config_name)

    def set_project_dir(self, project_dir):
        project_dir_path = Path(project_dir)
        if project_dir_path.exists():
            return project_dir_path
        raise ImportError(f"The project directory {project_dir} could not be found.")

    def resolve_config_name(self, config_name):
        if config_name:
            return config_name
        config_files = list(self.project_dir.glob("*.yaml")) + list(
            self.project_dir.glob("*.yml")
        )
        if len(config_files) == 0:
            raise FileNotFoundError(
                "Couldn't find any configuration file in the project. Make sure you've include a .yaml file in the top directory."
            )
        if len(config_files) > 1:
            raise ValueError(
                "Couldn't resolve the configuration file to be used. Specify a value in the `config_name` argument."
            )
        return config_files[0]

    def load_config(self, config_path):
        """
        This function loads a configuration file and returns it as a dictionary.

        Parameters:
        config_path (str): The path to the configuration file to load.

        Returns:
        dict: A dictionary containing the loaded configuration.
        """

        if not config_path.exists():
            raise ValueError(
                f"The configuration file not found. Make sure {config_path.name} exists, or provide a different file name."
            )

        with open(config_path) as config_file:
            config = yaml.safe_load(config_file)

        return config

    def load_dir_structure(self, read_dirs, write_dirs):
        """
        Loads the directory structure based on the provided configuration, read_dirs, and write_dirs.

        It sets attributes for each directory path after ensuring they exist or creating them if necessary.

        Parameters:
            project_dir (PosixPath): Path to the project directory.
            read_dirs (list): List of directory keys to be read from.
            write_dirs (list): List of directory keys to write in. Those directories will be overwritten if files already exist in them.

        Raises:
            KeyError: If a directory key is not registered in DIR_STRUCTURE.
        """
        all_dirs = set(read_dirs + write_dirs)
        for dir_name in all_dirs:
            try:
                dir_path = self.project_dir / self.DIR_STRUCTURE[dir_name]
            except KeyError as e:
                raise KeyError(f"The directory key {e} is not registered in BaseClass.")
            overwrite = dir_name in write_dirs
            setattr(
                self,
                f"{dir_name}_dir",
                self.create_if_not_exists(dir_path, overwrite=overwrite),
            )

    def load_model_path(self, model_version, pretrained=False):
        """
        Loads the path to the model file, either as pretrained model, or as a model to be evaluated.

        Parameters:
            project_dir (PosixPath): Path to the project directory.
            model_version (str): Identifier of the model version.
            pretrained (bool): Flag indicating whether to load a pretrained model.

        Returns:
            PosixPath: The path to the model pickle file.

        Raises:
            ValueError: If the model directory does not exist or does not contain exactly one pickle file.
        """
        if pretrained:
            model_version_dir = (
                self.project_dir / self.DIR_STRUCTURE["pretrained"] / model_version
            )
        else:
            model_version_dir = self.models_dir / model_version
        if not model_version_dir.exists():
            raise ValueError(
                f"Couldn't find model directory {model_version_dir} for {'pretraining' if pretrained else 'evaluation'}."
            )
        # Find all pickle files in the directory
        pickle_files = list(model_version_dir.glob("*.pkl"))

        # Check if there is exactly one pickle file
        if len(pickle_files) != 1:
            raise ValueError(
                f"Expected exactly one pickle file in {model_version_dir}, but found {len(pickle_files)}."
            )

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
            logging.warning(
                f"Warning: {dir_path.name} directory is not empty. Overwriting files."
            )
            shutil.rmtree(dir_path)  # Delete the directory and its contents
            dir_path.mkdir(parents=True)
        return dir_path
