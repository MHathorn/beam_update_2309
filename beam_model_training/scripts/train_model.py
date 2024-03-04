import argparse
import os
import ssl

from segmentation.train import Trainer
from segmentation.eval import Evaluator
from segmentation.losses import CombinedLoss, DualFocalLoss, CrossCombinedLoss

from utils.helpers import load_config


def main(config_file):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    ssl._create_default_https_context = ssl._create_unverified_context

    config = load_config(config_file)

    trainer = Trainer(config)
    trainer.run()

    evaluator = Evaluator(config)
    evaluator.evaluate(n_images=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with a specified configuration file."
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="The name of the configuration file.",
        default="test_config.yaml",
    )

    args = parser.parse_args()
    main(args.config_file)
