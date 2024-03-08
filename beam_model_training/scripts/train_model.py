import argparse
import os
import ssl

from segmentation.train import Trainer
from segmentation.eval import Evaluator


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    ssl._create_default_https_context = ssl._create_unverified_context

    trainer = Trainer(args.project_dir, args.config_name)
    trainer.run()

    evaluator = Evaluator(args.project_dir, args.config_name)
    evaluator.evaluate(n_images=args.n_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a model and evaluates it based on a specified configuration file."
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

    # Optional argument: Number of images for evaluation analysis
    parser.add_argument(
        "--n_images",
        type=int,
        default=0,
        help="The number of images of tiles with detections to generate during evaluation.",
    )

    args = parser.parse_args()
    main(args)
