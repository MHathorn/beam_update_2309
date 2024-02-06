# Example usage
from pathlib import Path
import pandas as pd
from segmentation.infer import MapGenerator
import geopandas as gpd

from utils.helpers import load_config, seed


if __name__ == "__main__":
    config = load_config("UNet_config.yaml")
    seed(config["seed"])
    # Assuming the tiles have already been prepared
    root_path = Path(config["root_dir"])
    shp_dir = root_path / "AOIs"
    img_dir = root_path / "tiles/images"

    settlements = pd.DataFrame()
    for file_path in shp_dir.iterdir():
        if file_path.suffix == ".shp" and file_path.stem.endswith("ElSavaldor"):
            df = gpd.read_file(file_path)
            settlements = pd.concat([settlements, df])

    map_gen = MapGenerator(config)
    map_gen.create_tile_inferences(
        images_dir=img_dir, settlements=settlements, primary_key="OBJECTID"
    )
