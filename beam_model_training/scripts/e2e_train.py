
import os
from pathlib import Path

from preprocess.data_tiler import DataTiler
from preprocess.transform import gen_train_test
from segmentation.train import Trainer
from segmentation.eval import Evaluator
from utils.helpers import load_config, seed

def prepare_data(config):
    seed(config["seed"])
    
    img_tiler = DataTiler(config)
    img_tiler.generate_tiles(config["tile_size"])

    if config["training"]:
        gen_train_test(config["root_dir"], config["dirs"], test_size=config["test_size"])

def train(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    trainer = Trainer(config)
    model_path = trainer.run()
    return model_path

def eval(config):
    evaluator = Evaluator(config)
    evaluator.evaluate()

def main():
    config = load_config("base_config.yaml")
    prepare_data(config)
    model_path = train(config)
    config["test"]["model_name"] = model_path.name
    eval(config)

if __name__ == '__main__':
    main()