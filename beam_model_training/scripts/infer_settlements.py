# Example usage
from pathlib import Path
import pandas as pd
from segmentation.infer import MapGenerator
import geopandas as gpd

from utils.helpers import load_config, seed


if __name__ == "__main__":
    config = load_config("settlements_gen_config.yaml")
    seed(config["seed"])
    # Assuming the tiles have already been prepared
    root_path = Path(config["root_dir"])
    shp_dir = root_path / "AOIs"
    img_dir = root_path / "tiles/images"

    settlements = pd.DataFrame()
    for file_path in shp_dir.iterdir():
        if file_path.suffix == ".shp":
            df = gpd.read_file(file_path)
            settlements = pd.concat([settlements, df])

    settlements = settlements[settlements.OBJECTID.isin([267,414,493,533,196,290,300,518,276])]
    map_gen = MapGenerator(config, generate_preds=True)
    map_gen.create_tile_inferences(
        images_dir=img_dir,
        settlements=settlements, primary_key="OBJECTID"
    )
