import argparse
import os

from preprocess.data_tiler import DataTiler
from preprocess.transform import gen_train_test
from segmentation.train import Trainer
from segmentation.eval import Evaluator
from utils.helpers import load_config, seed

def prepare_data(config):
    seed(config["seed"])
    
    img_tiler = DataTiler(config)
    img_tiler.generate_tiles(config["tiling"]["tile_size"])

    if config["training"]:
        gen_train_test(config["root_dir"], test_size=config["test_size"])

def train(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    trainer = Trainer(config)
    model_path = trainer.run()
    return model_path

def eval(config):
    evaluator = Evaluator(config)
    evaluator.evaluate()

def main(config_file):
    config = load_config("test_config.yaml")
    prepare_data(config)
    model_path = train(config)
    config["test"]["model_name"] = model_path.name
    eval(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with a specified configuration file.')
    parser.add_argument('-c', '--config_file', type=str, help='The name of the configuration file.', default="test_config.yaml")

    args = parser.parse_args()
    main(args.config_file)