
from pathlib import Path

from preprocess.data_tiler import DataTiler
from preprocess.transform import gen_train_test
from utils.helpers import seed, load_config

def main():
    config = load_config("base_config.yaml")
    seed(config["seed"])
    root_dir = Path(config["root_dir"])
    
    img_tiler = DataTiler(root_dir)
    img_tiler.generate_tiles(config["tile_size"])

    if config["training"]:
        gen_train_test(root_dir / config["dirs"]["tiles"], test_size=config["test_size"])


if __name__ == '__main__':
    main()