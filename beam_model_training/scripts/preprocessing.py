from tiling.data_tiler import DataTiler
from pathlib import Path
from utils.my_paths import ROOT_PATH

def main():
    path = Path(ROOT_PATH)
    tile_size = 512
    
    img_tiler = DataTiler(path)
    img_tiler.generate_tiles(tile_size)

if __name__ == '__main__':
    main()