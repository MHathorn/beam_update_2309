"""
This prepares a sample of tiles in PNG format to be tasked for labeling.

The script reads configuration settings from the provided config file in the project directory (default: project_config.yaml).
It then samples a subset of these tiles based on Areas of Interest (AOIs) and copies them to a designated
sample directory. Finally, it converts the sampled multiband tiles to PNG format and saves them in a separate
PNG directory.

"""

import argparse
import shutil
from pathlib import Path

from utils.helpers import multiband_to_png

from preprocess.data_tiler import DataTiler
from preprocess.sample import sample_tiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare tiles in PNG format for labeling. This script expects to find the settlement boundaries file in the AOIs/ sub-directory."
    )
    parser.add_argument(
        "-d", "--project_dir", type=str, help="The project directory.", required=True
    )  # required
    parser.add_argument(
        "-c",
        "--config_name",
        type=str,
        default="project_config.yaml",
        help="The configuration file name. Defaults to 'project_config.yaml'.",
    )  # optional
    parser.add_argument(
        "--sample_size",
        type=int,
        default=80,
        help="The size of the sample to generate for labeling.",
    )  # optional
    args = parser.parse_args()
    root_path = Path(args.project_dir)
    input_dir = root_path / DataTiler.DIR_STRUCTURE["image_tiles"]

    if not input_dir.exists():
        img_tiler = DataTiler(args.project_dir, args.config_name)
        img_tiler.generate_tiles()

    output_dir = DataTiler.create_if_not_exists(
        input_dir.parent / "sample/images", overwrite=True
    )

    png_output_dir = DataTiler.create_if_not_exists(
        input_dir.parent / "sample/png", overwrite=True
    )

    sampled_tile_paths = sample_tiles(input_dir, root_path / "AOIs", args.sample_size)

    for file_path in sampled_tile_paths:
        try:
            shutil.copy2(file_path, output_dir / file_path.name)
            multiband_to_png(file_path, png_output_dir)
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
