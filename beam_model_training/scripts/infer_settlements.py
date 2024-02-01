# Example usage
from pathlib import Path
import pandas as pd
from segmentation.infer import MapGenerator
import geopandas as gpd

from utils.helpers import load_config, seed



if __name__ == '__main__':
    config = load_config("UNet_config.yaml")
    seed(config["seed"])
    # Assuming the tiles have already been prepared
    root_path = Path(config["root_dir"])
    map_gen = MapGenerator(config)
    shp_dir = root_path / "AOIs"
    img_dir = root_path / "tiles/images"

    settlements = pd.DataFrame()
    for file_path in shp_dir.iterdir():
        if file_path.suffix == '.shp':
            df = gpd.read_file(file_path)   
            settlements = pd.concat([settlements, df])
    
    map_gen.create_tile_inferences(images_dir=img_dir, AOI_gpd=settlements, merge_outputs=True)
    # map_gen.create_tile_inferences(images_dir=img_dir, AOI_gpd=settlements)
