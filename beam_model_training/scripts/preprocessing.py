from tiling.data_tiler import DataTiler
from pathlib import Path

def main():
    path = Path("C:/Users/adm.nkiner/Documents/BEAM/training_files")
    image_dir = path / "images"
    output_dir = path / "output_dir"
    img_tiler = DataTiler(image_dir, output_dir)
    img_tiler.generate_tiles(512)

if __name__ == '__main__':
    main()