import shutil
import geopandas as gpd
import pandas as pd
import rioxarray as rxr
from pathlib import Path
from shapely.geometry import box
from utils.helpers import create_if_not_exists

def load_tile_info(tile_path):
    """Load tile and extract metadata."""
    tile = rxr.open_rasterio(tile_path)
    num_buildings = tile.attrs.get('num_buildings', 0)
    probability_score = tile.attrs.get('probability_score', 1)  # Default to 1 (high confidence)
    return tile, num_buildings, probability_score

def tile_in_settlement(tile, settlements):
    """Check if the tile overlaps with any informal settlement."""
    tile_geom = box(*tile.rio.bounds())
    return settlements.intersects(tile_geom).any()

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
    input_dir = Path("C:/Users/natha/OneDrive/Desktop/Local_Training/tiles/images")
    output_dir = create_if_not_exists(input_dir.parent / "sample")
    sampled_tile_paths = sample_tiles("C:/Users/natha/OneDrive/Desktop/Local_Training/tiles/images", "C:/Users/natha/OneDrive/Desktop/Local_Training/Costa_Rica/PR05_Asentamientos_Informales_MIVAH_2023_2.shp", 20)
    print(sampled_tile_paths)
    for file_path in sampled_tile_paths:
        shutil.copy2(file_path, output_dir / file_path.name)
