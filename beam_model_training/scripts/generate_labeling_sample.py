# Example usage
import shutil
from pathlib import Path

from utils.helpers import load_config, seed, multiband_to_png

from preprocess.data_tiler import DataTiler
from preprocess.sample import sample_tiles



if __name__ == '__main__':
    config = load_config("labeling_config.yaml")
    seed(config["seed"])
    # Assuming the tiles have already been prepared
    root_path = Path(config["root_dir"])
    input_dir =  root_path / DataTiler.DIR_STRUCTURE["image_tiles"]
    label_dir = root_path / DataTiler.DIR_STRUCTURE["label_tiles"]

    if not input_dir.exists():
        img_tiler = DataTiler(config)
        img_tiler.generate_tiles(512)

    output_dir = DataTiler.create_if_not_exists(input_dir.parent / "sample/images", overwrite=True)
    
    output_label_dir = DataTiler.create_if_not_exists(input_dir.parent / "sample/labels", overwrite=True)
    output_json_dir = DataTiler.create_if_not_exists(input_dir.parent / "sample/json", overwrite=True)  

    
    sampled_tile_paths = sample_tiles(input_dir, root_path / "AOIs", 80)

    print(sampled_tile_paths)

    for file_path in sampled_tile_paths:
        shutil.copy2(file_path, output_dir / file_path.name)
        

        # Get the base name of the image file (without extension)
        base_name = file_path.stem
        
        # Find all files in the label directory that start with the base name
        label_files = list(label_dir.glob(f"{base_name}.*"))
        
        # Copy each matching label file to the output label directory
        for label_file in label_files:
            shutil.copy2(label_file, output_label_dir / label_file.name)

    # Tile sample in 128*128
    config["root_dir"] = input_dir.parent / 'sample'
    img_tiler = DataTiler(config)
    img_tiler.generate_tiles(128)

    sample_images_dir = img_tiler.image_tiles_dir
    png_output_dir = DataTiler.create_if_not_exists(sample_images_dir.parent / "png", overwrite=True)

    for img_path in sample_images_dir.iterdir():
        multiband_to_png(img_path, png_output_dir)

