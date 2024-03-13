from functools import partial
import json
import logging
from pathlib import Path
from random import sample
import shutil

import geopandas as gpd
import pandas as pd
import rioxarray as rxr

from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import box
from tqdm import tqdm
from utils.base_class import BaseClass
from utils.helpers import crs_to_pixel_coords, multiband_to_png, seed


def tile_in_settlement(tile, settlements):
    """
    Check if the given raster tile overlaps with any polygons in a GeoDataFrame representing settlements.

    Args:
        tile (DataArray): The raster tile to check.
        settlements (GeoDataFrame): A GeoDataFrame containing polygons of settlements.

    Returns:
        bool: True if the tile overlaps with any settlement polygons, False otherwise.
    """

    # Ensure both geometries are in the same CRS
    if tile.rio.crs != settlements.crs:
        settlements = settlements.to_crs(tile.rio.crs)

    tile_geom = box(*tile.rio.bounds())
    return any(settlements.intersects(tile_geom))


def include_tile(tile_path, settlements):
    """Function to map each tile to an inclusion decision in distributed sample population creation below."""
    try:
        tile = rxr.open_rasterio(tile_path)
        if tile_in_settlement(tile, settlements):
            return tile_path
        else:
            return None
    except Exception as e:
        logging.error(f"Error verifying {tile_path}: {e}")
        return None


def sample_tiles(tile_directory, shp_dir, sample_size, seed_id=2022):
    """
    Sample a set of raster tiles from informal settlements.

    Args:
        tile_directory (str or Path): The directory containing raster tiles.
        shp_dir (str or Path): The directory containing shapefiles of settlements.
        sample_size (int): The number of tiles to sample.

    Returns:
        list: A list of paths to the sampled raster tiles.
    """
    tile_dir = Path(tile_directory)
    seed(seed_id)

    settlements = pd.DataFrame()
    for file_path in shp_dir.iterdir():
        if file_path.suffix == ".shp":
            df = gpd.read_file(file_path)
            settlements = pd.concat([settlements, df])

    include_tile_with_settlements = partial(include_tile, settlements=settlements)

    # Load tiles and calculate scores
    with ProcessPoolExecutor() as executor:
        settlement_tiles = list(
            tqdm(
                executor.map(include_tile_with_settlements, tile_dir.iterdir()),
                total=len(list(tile_dir.iterdir())),
            )
        )

    settlement_tiles = [tile for tile in settlement_tiles if tile is not None]

    if len(settlement_tiles) < sample_size:
        logging.warning(
            f"The total population of settlement tiles is lower than sample size. Returning all {len(settlement_tiles)} tiles."
        )
        return settlement_tiles

    # Generate sample of tiles
    sampled_tile_paths = sample(settlement_tiles, sample_size)

    return sampled_tile_paths


def create_sample_dir(image_tiles_dir, sampled_tile_paths):
    """
    Copies sampled tiles to a 'sample/images' directory and converts them to PNG format in a 'sample/png' directory.

    Args:
        image_tiles_dir (str or Path): The directory containing the original image tiles.
        sampled_tile_paths (list of Path): A list of paths to the image tiles that have been sampled.

    """
    output_dir = BaseClass.create_if_not_exists(
        image_tiles_dir.parent / "sample/images", overwrite=False
    )

    png_output_dir = BaseClass.create_if_not_exists(
        image_tiles_dir.parent / "sample/png", overwrite=False
    )

    for file_path in tqdm(sampled_tile_paths):
        try:
            shutil.copy2(file_path, output_dir / file_path.name)
            multiband_to_png(file_path, png_output_dir)
        except Exception as e:
            logging.error(f"An error occurred while processing {file_path}: {e}")


def generate_label_json(label_file, tiff_file, output_file, tile_size=512):
    """
    Generate a JSON file containing label data  in LabelStudio's expected format for a given image tile.
    See: https://labelstud.io/guide/predictions

    Note: This is unused and likely not usable as is - leaving here for possible
    future development with LabelStudio Enterprise.

    Args:
        label_file (str or Path): The file path to the vector data (shapefile) containing labels.
        tiff_file (str or Path): The file path to the raster tile (TIFF).
        output_file (str or Path): The file path where the output JSON will be saved.
        tile_size (int, optional): The size of the tile in pixels. Defaults to 512.

    """
    # Load your GeoDataFrame
    gdf = gpd.read_file(label_file)

    # Load your image tile as a DataArray
    image_tile = rxr.open_rasterio(tiff_file)

    # Iterate over GeoDataFrame and build JSON
    json_data = []
    for _, row in gdf.iterrows():
        geom = row["geometry"]

        if geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            polygons = [geom]

        for index, polygon in enumerate(polygons):
            pixel_coords = []

            # Convert each point in the geometry to pixel coordinates
            for x, y in polygon.exterior.coords:
                px, py = crs_to_pixel_coords(x, y, image_tile.rio.transform())
                pixel_coords.append([100.0 * px / tile_size, 100.0 * py / tile_size])

            polygon_json = {
                "type": "polygon",
                "from_name": "polygon",
                "to_name": "image",
                "original_width": tile_size,
                "original_height": tile_size,
                "id": index,
                "value": {"points": pixel_coords},
            }

            json_data.append(polygon_json)

    # Wrap in the overall JSON structure
    final_json = [
        {
            "data": {"image": "/data/upload/2/" + tiff_file.name},
            "predictions": [{"model_version": "google", "result": json_data}],
        }
    ]

    # Export to JSON file
    with open(output_file, "w") as f:
        json.dump(final_json, f, indent=4)
