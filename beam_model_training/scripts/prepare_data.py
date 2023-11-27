from preprocess.data_tiler import DataTiler
from preprocess.augmentation import train_test_split_files
from pathlib import Path
from utils.my_paths import ROOT_PATH

def main():
    path = Path(ROOT_PATH)
    tile_size = 512
    
    img_tiler = DataTiler(path)
    img_tiler.generate_tiles(tile_size)

    # Consider this is a training run if masks have been detected by the tiler. Otherwise, inference.
    training = (path / "tiles/masks").exists()
    if training:
        train_test_split_files(path / "tiles", split=0.2)


if __name__ == '__main__':
    main()