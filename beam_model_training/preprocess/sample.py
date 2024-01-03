from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rxr
from preprocess.transform import get_rgb_channels
from shapely.geometry import box
from utils.helpers import create_if_not_exists, seed


def load_tile_info(tile_path):
    """Load tile and extract metadata."""
    tile = rxr.open_rasterio(tile_path)
    num_buildings = tile.attrs.get('num_buildings', 0)
    probability_score = tile.attrs.get('probability_score', 1)  # Default to 1 (high confidence)
    return tile, num_buildings, probability_score

def tile_in_settlement(tile, settlements):
    """Check if the tile overlaps with any informal settlement."""
    
    # Ensure both geometries are in the same CRS
    if tile.rio.crs != settlements.crs:
        tile = tile.rio.reproject(settlements.crs)
        
    tile_geom = box(*tile.rio.bounds())
    return any(settlements.intersects(tile_geom))

def calculate_score(num_buildings, probability_score, in_settlement):
    """Calculate a score for the tile based on the given criteria."""
    sampling_score = (1 - probability_score) if num_buildings > 10 else 0.5

    if in_settlement:
        sampling_score *= 4  # Boost for informal settlements

    return sampling_score

def sample_tiles(tile_directory, settlements_shapefile, sample_size):
    tile_dir = Path(tile_directory)
    settlements = gpd.read_file(settlements_shapefile)

    # Load tiles and calculate scores
    tile_scores = []
    for tile_path in tile_dir.glob('*.tif'):
        tile, num_buildings, probability_score = load_tile_info(tile_path)
        in_settlement = tile_in_settlement(tile, settlements)
        score = calculate_score(num_buildings, probability_score, in_settlement)
        tile_scores.append((tile_path, score))

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(tile_scores, columns=['tile_path', 'score'])

    # Normalize and sample
    df['normalized_score'] = df['score'] / df['score'].sum()
    sampled_tiles = df.sample(n=sample_size, weights='normalized_score')

    return sampled_tiles['tile_path'].tolist()

# Example usage
if __name__ == '__main__':
    seed(2022)
    input_dir = Path("C:/Users/natha/OneDrive/Desktop/Local_Training/tiles/images")
    output_dir = create_if_not_exists(input_dir.parent / "sample_png")
    sampled_tile_paths = sample_tiles("C:/Users/natha/OneDrive/Desktop/Local_Training/tiles/images", "C:/Users/natha/OneDrive/Desktop/Local_Training/Costa_Rica/PR05_Asentamientos_Informales_MIVAH_2023_2.shp", 20)
    print(sampled_tile_paths)
    for file_path in sampled_tile_paths:
        tiff_file = Path(file_path)
        png_file = output_dir / tiff_file.with_suffix('.png').name

    # Open the TIFF file and convert it to PNG
        try:
            img = get_rgb_channels(tiff_file)
            plt.imsave(png_file, np.transpose(img.values, (1,2,0)))   
            print(f"Converted {tiff_file} to {png_file}")
        except Exception as e:
            print(f"An error occurred while converting {tiff_file}: {e}")
