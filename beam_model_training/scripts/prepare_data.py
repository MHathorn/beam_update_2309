import argparse
import logging

from preprocess.data_tiler import DataTiler
from preprocess.transform import gen_train_test

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(args):

    img_tiler = DataTiler(args.project_dir, args.config_name)
    img_tiler.generate_tiles(args.tile_size)

    if img_tiler.config["training"]:
        test_size = img_tiler.config.get("test_size", 0.2)
        if args.test_size is not None:
            if 0 <= args.test_size <= 1:
                test_size = args.test_size
            else:
                raise ValueError("The --test_size argument must be between 0 and 1.")

        gen_train_test(
            args.project_dir,
            test_size=test_size,
            distance_weighting=img_tiler.tiling_params["distance_weighting"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script runs tiling and, in the case of training, train-test split for a given set of images and (if training) labels."
    )

    # Required argument: Project directory
    parser.add_argument(
        "-d", "--project_dir", type=str, help="The project directory.", required=True
    )

    # Optional argument: Configuration file name
    parser.add_argument(
        "-c",
        "--config_name",
        type=str,
        default="project_config.yaml",
        help="The configuration file name. Defaults to 'project_config.yaml'.",
    )

    # Optional argument: Tile size
    parser.add_argument(
        "--tile_size",
        type=int,
        default=0,
        help="The size of the tiles to be generated and saved. Defaults to the value in the configuration file.",
    )

    # Optional argument: Test size
    parser.add_argument(
        "--test_size",
        type=float,
        help="The size of the tiles to be generated and saved. Should be a float between 0 and 1. Defaults to the value in the configuration file, or 0.2 if missing.",
    )

    args = parser.parse_args()
    main(args)
