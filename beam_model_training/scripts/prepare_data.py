
from pathlib import Path

from preprocess.data_tiler import DataTiler
from preprocess.transform import gen_train_test
from utils.my_paths import ROOT_PATH, SEED
from utils.helpers import seed

def main():
    path = Path(ROOT_PATH)
    seed(SEED)
    tile_size = 512
    
    img_tiler = DataTiler(path)
    img_tiler.generate_tiles(tile_size)

    # Consider this is a training run if masks have been detected by the tiler. Otherwise, inference.
    training = (path / "tiles/masks").exists()
    if training:
        gen_train_test(path / "tiles", test_size=0.2)


if __name__ == '__main__':
    main()