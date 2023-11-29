from datetime import datetime
from pathlib import Path
import shutil
import random
from fastai.vision.all import set_seed, torch
import pytz


def seed(SEED):
    """Seed randomization functions for reproducibility."""
    random.seed(SEED)
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def timestamp():
    """Timestamp for conducting experiments"""
    tz = pytz.timezone('Europe/Berlin')
    date = str(datetime.now(tz)).split(" ")
    date_time = f"{date[0]}_{date[1].split('.')[0][:5]}"
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
