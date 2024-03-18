"""
This script processes JSON files exported from LabelStudio and converts the labeling information into shapefiles. 
It is designed to work with raster tiles in TIFF formats, that have been used a input for labeling in LabelStudio.

Usage:
Run the script from the command line, providing paths to directories containing JSON files, TIFF images, and where the shapefiles will be saved.

Example command:
python script.py --json_dir /path/to/jsons --img_dir /path/to/tiff/images --shp_dir /path/to/save/shapefiles

The script assumes that the input JSON files are exports from LabelStudio tasks, each containing labels for an image. The labeled points are converted to polygons using the coordinate reference system of the input TIFF images, which are then reprojected to the specified CRS before being saved as shapefiles.

Note:
- The script expects the TIFF images to be named consistently with the 'file_upload' field in the LabelStudio JSON export, after removal of the prefix used by LabelStudio.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from shapely.geometry import Polygon, box
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_all_exports(json_dir, img_dir, shp_dir):
    """
    Process JSON files exported from LabelStudio and save them as shapefiles.

    Parameters:
    - json_dir (Path): Directory containing JSON files.
    - img_dir (Path): Directory where TIFF images used for labeling will be found.
    - shp_dir (Path): Directory where shapefiles will be saved.
    """
    file_gdfs = []
    crs = "EPSG:32616"
    json_files = [f for f in json_dir.iterdir() if f.suffix == ".json"]
    with tqdm(json_files, desc="Processing JSON files", unit="file") as pbar_json:
        for file in pbar_json:
            pbar_json.set_postfix(file=file.name)
            with open(file) as f:
                data = json.load(f)

                # Inner progress bar for tasks within a JSON file
                with tqdm(
                    data,
                    desc=f"Processing tasks in {file.name}",
                    leave=False,
                    unit="task",
                ) as pbar_task:
                    gdf_list = []
                    for task in pbar_task:
                        try:
                            gdf = convert_task_to_gdf(task, crs, img_dir)
                            if gdf is not None:
                                gdf_list.append(gdf)
                        except FileNotFoundError:
                            logging.error(
                                f"TIFF image not found for task {task['id']}."
                            )
                        except Exception as e:
                            logging.error(f"Error processing task {task['id']}: {e}")
                        finally:
                            pbar_task.update(1)  # Update the inner progress bar

                    if gdf_list:
                        file_gdf = pd.concat(gdf_list, ignore_index=True)
                        file_gdf["json_name"] = file.name
                        file_gdfs.append(file_gdf)

            pbar_json.update(1)  # Update the outer progress bar

    if file_gdfs:
        gdf = pd.concat(file_gdfs, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        shapefile_name = f"labeled_buildings_{timestamp}.shp"
        try:
            gdf.drop(columns=["image_bounds"]).to_file(shp_dir / shapefile_name)
        except Exception as e:
            logging.error(f"Failed to save shapefile: {e}")
    else:
        logging.error(f"No labels to save in {json_dir}.")


def convert_task_to_gdf(task, crs, img_dir):
    """
    Convert a labeling task to a GeoDataFrame.

    Parameters:
    - task (dict): A dictionary containing information about a single labeling task.
    - crs (str): The coordinate reference system to use for the GeoDataFrame.
    - img_dir (Path): The directory where TIFF images used for labeling are located.

    Returns:
    - pd.DataFrame or None: A DataFrame containing all the results from the task converted.
    """
    img_name = extract_name_from_task(task)

    img_path = (img_dir / img_name).with_suffix(".TIF")
    img = rxr.open_rasterio(img_path, default_name=img_path.name)
    if img.rio.crs != crs:
        img = img.rio.reproject(crs)

    gdf_list = [
        get_gdf_from_result(result, img, crs)
        for annotation in task["annotations"]
        for result in annotation["result"]
        if result
    ]
    if gdf_list:
        return pd.concat(gdf_list, ignore_index=True)
    else:
        logging.warning(f"No labels could be found on image {img_name}.")


def extract_name_from_task(task):
    """
    Extract the image name from a single task's information.

    Parameters:
    - task (dict): A dictionary containing information about a single task (as defined in LabelStudio).

    Returns:
    - str: The extracted image name.
    """
    img_id = task["file_upload"]
    parts = img_id.split("-", 1)  # Split the string at the first occurrence of '-'
    return parts[1] if len(parts) > 1 else parts[0]


def get_gdf_from_result(result, img, crs):
    """
    Create a GeoDataFrame from a labeling result.

    Parameters:
    - result (dict): A dictionary containing the labeling result for one image.
    - img (xarray.DataArray): An xarray DataArray representing the geospatial raster data.
    - crs (str): The coordinate reference system to use for the GeoDataFrame.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing the label and geometry information.
    """
    tile_size = int(result["original_width"])  # make sure tiles are square
    points = result["value"]["points"]

    points = [
        img.rio.transform() * (tile_size / 100 * np.array(point)) for point in points
    ]

    polygon = Polygon(points)
    label = result["value"]["polygonlabels"][0]
    gdf = gpd.GeoDataFrame(
        {
            "image_name": [img.name],
            "image_bounds": [box(*img.rio.bounds())],
            "geometry": [polygon],
            "label": [label],
        },
        geometry="geometry",
        crs=img.rio.crs,
    )
    gdf = gdf.to_crs(
        crs
    )  # Convert to desired CRS only after pixel-to-coords transformation
    return gdf


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Process JSON files exported from LabelStudio into a shapefile of labels."
    )
    parser.add_argument(
        "--json_dir", type=str, required=True, help="Directory containing JSON files"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Directory where TIFF images used for labeling will be found",
    )
    parser.add_argument(
        "--shp_dir",
        type=str,
        required=True,
        help="Directory where shapefiles will be saved",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Assign directories from command line arguments
    json_dir = Path(args.json_dir)
    img_dir = Path(args.img_dir)
    shp_dir = Path(args.shp_dir)

    process_all_exports(json_dir, img_dir, shp_dir)
