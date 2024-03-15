import argparse
import logging
import os

from preprocess.data_tiler import DataTiler
from preprocess.transform import gen_train_test
from segmentation.train import Trainer
from segmentation.eval import Evaluator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def prepare_data(project_dir, config_name):

    img_tiler = DataTiler(project_dir, config_name)
    img_tiler.generate_tiles()

    if img_tiler.config["training"]:
        gen_train_test(project_dir, test_size=img_tiler.config["test_size"])


def train(project_dir, config_name):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"

    trainer = Trainer(project_dir, config_name)
    model_path = trainer.run()
    return model_path


def eval(project_dir, config_name, model_path):
    evaluator = Evaluator(project_dir, config_name, model_path=model_path)
    evaluator.evaluate()


def main(project_dir, config_name):
    prepare_data(project_dir, config_name)
    model_path = train(project_dir, config_name)
    eval(project_dir, config_name, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model end-to-end including tiling, training and evaluation based on a specified configuration file."
    )
    parser.add_argument(
        "-d",
        "--project_dir",
        type=str,
        help="The project directory.",
    )
    parser.add_argument(
        "-c",
        "--config_name",
        type=str,
        help="The configuration file name. If missing, the constructor will look for a single file in the project directory.",
    )

    args = parser.parse_args()
    main(args.project_dir, args.config_file)
