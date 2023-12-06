import random
import shutil
from datetime import datetime
from pathlib import Path

import pytz
import yaml
from fastai.vision.all import set_seed, torch


def seed(seed_value=0):
    """Seed randomization functions for reproducibility."""
    random.seed(seed_value)
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
    default_config = Path(__file__).parents[1] / "configs" / "base_config.yaml"
    config_path = default_config.parent / config_name

    if not default_config.exists():
         raise ImportError("Configs file not found. Make sure the configs/ directory location is correct.")
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

def create_if_not_exists(dir, overwrite=False):
        """
        Create a directory if it does not exist. Optionally, f the directory exists and is not empty,
        files will get overwritten.

        Parameters:
        dir (PosixPath|str): The path of the directory to create.

        Returns:
        dir_path (PosixPath): The path of the created directory.

        """
        dir_path = Path(dir)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        elif overwrite and any(dir_path.iterdir()):
            print(f"Warning: {dir_path.name} directory is not empty. Overwriting files.")
            shutil.rmtree(dir_path)  # Delete the directory and its contents
            dir_path.mkdir(parents=True)
        return dir_path
