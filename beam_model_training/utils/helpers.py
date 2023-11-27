import os
from pathlib import Path
import shutil

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
            print(f"Warning: Output directory is not empty. Overwriting files.")
            shutil.rmtree(dir_path)  # Delete the directory and its contents
            dir_path.mkdir(parents=True)
        return dir_path

def reset_directories(source_dir, target_dir):
    """Move all files from source to target directory."""
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join(target_dir, filename))