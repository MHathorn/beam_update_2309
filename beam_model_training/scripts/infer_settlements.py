import argparse
import geopandas as gpd
import logging
from pathlib import Path
import pandas as pd

from segmentation.infer import MapGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def validate_primary_key(gdf, primary_key):
    """Validate that the primary key exists in the GeoDataFrame."""
    if primary_key not in gdf.columns:
        available_columns = ', '.join(gdf.columns)
        raise ValueError(
            f"Primary key '{primary_key}' not found in settlement data. "
            f"Available columns are: {available_columns}"
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
        help="The configuration file name. If missing, the constructor will look for a single file in the project directory.",
    )  # optional
    parser.add_argument(
        "-p",
        "--primary_key",
        type=str,
        default="OBJECTID",
        help="The primary key identifying each individual settlement in the settlements file. \
            This defaults to OBJECTID for informal settlement boundaries in the SICA region.",
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

    if settlements.empty:
        raise FileNotFoundError(f"No shapefiles found in {shp_dir}")

    validate_primary_key(settlements, args.primary_key)        

    map_gen = MapGenerator(
        project_dir=args.project_dir, config_name=args.config_name, generate_preds=True
    )
    map_gen.generate_map_from_images(
        images_dir=img_dir, boundaries_gdf=settlements, primary_key=args.primary_key
    )
