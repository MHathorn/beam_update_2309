"""
This prepares a sample of tiles in PNG format to be tasked for labeling.

The script finds the tiles overlapping with informal settlements and samples from that population. 
It then converts the sampled multiband tiles to PNG format and saves them in a separate
PNG directory.

"""

import argparse
import logging
from pathlib import Path
from preprocess.data_tiler import DataTiler

from preprocess.sample import create_sample_dir, sample_tiles

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare tiles in PNG format for labeling. This script expects to find the settlement boundaries file in the AOIs/ sub-directory."
    )
    parser.add_argument(
        "-d", "--project_dir", type=str, help="The project directory.", required=True
    )  # required
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
        raise FileNotFoundError(
            "Tiles are missing from project directory {args.project_dir}. \
            Go through the tiling process before running this script."
        )

    logging.info("Creating sample population..")
    sampled_tile_paths = sample_tiles(input_dir, root_path / "AOIs", args.sample_size)
    logging.info("Generating sample directory..")
    create_sample_dir(input_dir, sampled_tile_paths)
