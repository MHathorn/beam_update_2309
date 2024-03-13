# Example usage
import argparse
from pathlib import Path
import pandas as pd
from segmentation.infer import MapGenerator
import geopandas as gpd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the map for a given set of settlements."
    )
    parser.add_argument(
        "-d", "--project_dir", type=str, help="The project directory.", required=True
    )  # required
    parser.add_argument(
        "-c",
        "--config_name",
        type=str,
        default="project_config.yaml",
        help="The configuration file name. Defaults to 'project_config.yaml'.",
    )  # optional

    args = parser.parse_args()

    root_path = Path(args.project_dir)
    shp_dir = root_path / "AOIs"
    img_dir = (
        root_path / "tiles/images"
    )  # assuming the tiles have already been prepared

    settlements = pd.DataFrame()
    for file_path in shp_dir.iterdir():
        if file_path.suffix == ".shp":
            df = gpd.read_file(file_path)
            settlements = pd.concat([settlements, df])

    map_gen = MapGenerator(
        project_dir=args.project_dir, config_name=args.config_name, generate_preds=True
    )
    map_gen.generate_map_from_images(
        images_dir=img_dir, boundaries_gdf=settlements, primary_key="OBJECTID"
    )
