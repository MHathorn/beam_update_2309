import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
from shapely.geometry import box
from utils.helpers import crs_to_pixel_coords


def load_tile_info(tile_path):
    """
    Load a raster tile using rioxarray and extract its metadata.

    Args:
        tile_path (str or Path): The file path to the raster tile.

    Returns:
        tuple: A tuple containing the loaded tile as an xarray DataArray,
               the number of buildings as an integer, and the probability score as a float.
    """
    tile = rxr.open_rasterio(tile_path)
    num_buildings = tile.attrs.get("num_buildings", 0)
    probability_score = tile.attrs.get(
        "probability_score", 1
    )  # Default to 1 (high confidence)
    return tile, num_buildings, probability_score


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


def calculate_score(num_buildings, probability_score, in_settlement):
    """
    Calculate a score for a tile based on the number of buildings, probability score, and whether it is in a settlement.

    Args:
        num_buildings (int): The number of buildings detected in the tile.
        probability_score (float): The probability score associated with the tile.
        in_settlement (bool): Whether the tile is within an informal settlement.

    Returns:
        float: The calculated sampling score for the tile.
    """
    sampling_score = (1 - probability_score) if num_buildings > 10 else 0.5

    if in_settlement:
        sampling_score *= 6  # Boost for informal settlements

    return sampling_score


def sample_tiles(tile_directory, shp_dir, sample_size):
    """
    Sample a set of raster tiles based on scores calculated from metadata and overlap with settlements.

    Args:
        tile_directory (str or Path): The directory containing raster tiles.
        shp_dir (str or Path): The directory containing shapefiles of settlements.
        sample_size (int): The number of tiles to sample.

    Returns:
        list: A list of paths to the sampled raster tiles.
    """
    tile_dir = Path(tile_directory)

    settlements = pd.DataFrame()
    for file_path in shp_dir.iterdir():
        if file_path.suffix == ".shp":
            df = gpd.read_file(file_path)
            settlements = pd.concat([settlements, df])

    # Load tiles and calculate scores
    tile_scores = []
    for tile_path in tile_dir.glob("*.tif"):
        tile, num_buildings, probability_score = load_tile_info(tile_path)
        in_settlement = tile_in_settlement(tile, settlements)
        score = calculate_score(num_buildings, probability_score, in_settlement)
        tile_scores.append((tile_path, score))

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(tile_scores, columns=["tile_path", "score"])

    # Normalize and sample
    df["normalized_score"] = df["score"] / df["score"].sum()
    sampled_tiles = df.sample(n=sample_size, weights="normalized_score")

    return sampled_tiles["tile_path"].tolist()


def generate_label_json(label_file, tiff_file, output_file, tile_size=512):
    """
    Generate a JSON file containing label data  in LabelStudio's expected format for a given image tile.
    See: https://labelstud.io/guide/predictions

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
