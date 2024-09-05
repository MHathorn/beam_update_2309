import argparse
import logging
import os
import ssl

from segmentation.train import Trainer
from segmentation.eval import Evaluator
from fastai.vision.all import *
from fastai.distributed import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    ssl._create_default_https_context = ssl._create_unverified_context

    setup_distrib()

    try:
        logging.info(f"Rank {rank_distrib()}: Starting main function")
        trainer = Trainer(args.project_dir, args.config_name)
        model_path = trainer.run()

        if rank_distrib() == 0:
            logging.info("Rank 0: Starting evaluation")
            evaluator = Evaluator(
                args.project_dir, args.config_name, model_path=model_path, generate_preds=True
            )
            evaluator.evaluate(n_images=args.n_images)
    except Exception as e:
        logging.error(f"Rank {rank_distrib()}: {e}")        
    finally:
        teardown_distrib()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        help="The configuration file name. If missing, the constructor will look for a single file in the project directory.",
    )

    # Optional argument: Number of images for evaluation analysis
    parser.add_argument(
        "--n_images",
        type=int,
        default=10,
        help="The number of images of tiles with detections to generate during evaluation.",
    )

    args = parser.parse_args()
    main(args)
