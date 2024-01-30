
import argparse
from preprocess.data_tiler import DataTiler
from preprocess.transform import gen_train_test
from utils.helpers import seed, load_config

def main(config_file):
    config = load_config(config_file)
    seed(config["seed"])
    
    img_tiler = DataTiler(config)
    img_tiler.generate_tiles(config["tiling"]["tile_size"])

    if config["training"]:
        gen_train_test(config["root_dir"], test_size=config["test_size"], distance_weighting=config["tiling"]["distance_weighting"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with a specified configuration file.')
    parser.add_argument('-c', '--config_file', type=str, help='The name of the configuration file.', default="test_config.yaml")

    args = parser.parse_args()
    main(args.config_file)