import gzip
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from cv2 import erode
from rasterio.features import rasterize
from shapely import wkt
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from utils.helpers import create_if_not_exists


class DataTiler:
    """Data Tiler class. This class loads from the configuration file an image directory, tile size, and tiles directory, and populates
    the tile directory with image tiles ready for data augmentation and train-test split. 
    If labels are presents in the directry, mask tiles will be created as well.
    

    Expected configuration attributes:
    - root_dir: Name of the project directory containing all training files.
    - tile_size: Size of the output tiles.
    - dirs: Directory folder paths, relative to the root directory.
        - images: Images directory.
        - labels: Labels directory.
        - tiles: Output directory for all tiles.
        - image_tiles: Output directory for image tiles.
        - mask_tiles (optional): Output directory for mask tiles.
    - erosion: true if erosion should be applied to the labels.

    Usage:
    img_tiler = DataTiler(input_dir)
    img_tiler.generate_tiles(tile_size)

    Expected output:
    Tiles saved in new sub-directory `image_tiles` and `mask_tiles` (if labels file provided).
    """
    def __init__(self, config):
            self.input_path = Path(config["root_dir"])

            # Checking for images and loading in DataArrays
            image_dir = self.input_path / config["dirs"]["images"]
            if not image_dir.exists():
                raise IOError("The directory path `images` does not point to an existing directry in `root_dir`.")
            self.images = self.load_images(image_dir)

            # Preparing tiles directory
            self.erosion = config["erosion"]
            self.output_dir = create_if_not_exists(self.input_path / config["dirs"]["tiles"], overwrite=True)
            self.dir_structure = {
                'image_tiles': create_if_not_exists(self.input_path / config["dirs"]["image_tiles"], overwrite=True),
            }
            
            # Checking for masks and loading if exist
            label_paths = list((self.input_path / config["dirs"]["labels"]).iterdir())
            valid_label_paths = [l for l in label_paths if l.suffix in ['.csv', '.shp'] or l.name.endswith('.csv.gz')]
            if not valid_label_paths:
                self.labels = None
                if len(label_paths) > 0:
                    print("Warning: Label files are not in recognized format (shp, csv). Tiling images alone.")
                else:
                    print(f"No labels file provided. Tiling images alone.")
            
            elif len(valid_label_paths) > 1:
                print(valid_label_paths)
                raise IOError("More than one labels file detected. Please provide a single labels file.")
            
            else:
                self.dir_structure['mask_tiles'] = create_if_not_exists(self.input_path / config["dirs"]["mask_tiles"], overwrite=True)
                # Loading labels from csv / shapefile.
                labels_path = valid_label_paths.pop(0)
                self.labels = self.load_labels(labels_path)
                print(f"Loaded vector labels from {labels_path.name}.")


    def load_images(self, image_dir):
        """
        Loads all GeoTIFF images in the provided image_dir with rioxarray.
        
        Parameters:
        image_dir: Images directory.
        Returns:
        xarray.DataArray: All bands stacked in a multidimensional array.
        """
        
        filepaths = [img_path for img_path in image_dir.rglob('*') if img_path.suffix.lower() in ['.tif', '.tiff']]
        
        if not filepaths:
            raise IOError(f"The directory {image_dir} does not contain any GeoTIFF images.")
        
        images = [rxr.open_rasterio(img_path, default_name=img_path.stem) for img_path in filepaths]
        # Unifying crs across images
        target_crs = images[0].rio.crs
        print("Found images:", len(images))
        return [img.rio.reproject(target_crs) for img in images]     


    def load_labels(self, labels_path, crop=True):
        """
        This loads building footprints from a vector file and stores them as an instance attribute.

        Parameters:
        labels_path: Path to labels file, in .csv, .shp or .csv.gz format.
        crop: True if labels datafame should be adapted to the size of the images dataset.

        Returns:
        GeoDataFrame: A geo dataframe containing all building labels.
        """
        if labels_path.suffix.lower() == '.csv':
            # Expecting here Google's Open Buildings Dataset format.
            df = pd.read_csv(labels_path)
            df['geometry'] = df['geometry'].apply(wkt.loads)
            buildings = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        elif labels_path.suffix.lower() == '.shp':
            buildings = gpd.read_file(labels_path)
        elif labels_path.name.endswith('.csv.gz'):
            # Unzip the .gz file and read it as csv
            with gzip.open(labels_path, 'rt') as f:
                df = pd.read_csv(f)
                df['geometry'] = df['geometry'].apply(wkt.loads)
                buildings = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        
        # Crop to adjust size to images
        if crop:
            return self.crop_labels(buildings)
        return buildings
            
    
    def crop_labels(self, buildings):
        """

        Parameters:
        buildings (GeoDataFrame): A GeoDataFrame containing the building geometries.

        Returns:
        GeoDataFrame: A GeoDataFrame containing only the building geometries that intersect with 
                    the bounding box of the input image.
        """
        bounding_boxes = [box(*img.rio.bounds()) for img in self.images]
        union_bounding_box = unary_union(bounding_boxes)
        buildings = buildings.to_crs(self.images[0].rio.crs)

        return buildings[buildings.intersects(union_bounding_box)]

    
    def generate_mask(self, image, write=False):
        """
        Generate a binary mask from a vector file (shp or geojson).

        Polygons are created from the vector labels. The mask is then created and eroded with a 3x3 kernel.

        Returns:
        numpy.ndarray: The generated mask, if 'write' is False. Otherwise, None.
        """

        # Generate the mask
        def poly_from_utm(polygon, transform):
            poly_pts = []
            poly = unary_union(polygon) 
            for i in np.array(poly.exterior.coords):
                poly_pts.append( ~ transform * tuple(i))
            new_poly = Polygon(poly_pts)
            return new_poly

        label_polygons = []
        image_size = (image.shape[1], image.shape[2])
        transform = image.rio.transform()
        crs = image.rio.crs.to_string()
        for _, row in self.labels.iterrows():
            if row['geometry'].geom_type == 'MultiPolygon':
                for label_geom in row['geometry'].geoms: 
                    # iterate over polygons within a MultiPolygon
                    label = poly_from_utm(label_geom, transform)
                    label_polygons.append(label)
            elif row['geometry'].geom_type == 'Polygon':
                label = poly_from_utm(row['geometry'], transform)
                label_polygons.append(label)
            else:
                # raise an error or skip the object
                raise TypeError("Invalid geometry type")

        if len(label_polygons) > 0:
            mask = rasterize(shapes=label_polygons, 
                             out_shape=image_size,
                             default_value=255, 
                             dtype="uint8")
        else:
            mask = np.full(image_size, 0, dtype=np.uint8)

        # Erode mask
        if self.erosion:
            kernel = np.ones((3, 3), np.uint8)
            mask = erode(mask, kernel, iterations=1)

        # Save or return mask
        mask_da = xr.DataArray(mask, dims=["y", "x"], coords={'x': image.coords['x'], 'y':image.coords['y']})
        mask_da.rio.write_crs(crs, inplace=True)
        mask_da.rio.write_transform(transform, inplace=True)

        if write:
        
            self.dir_structure['tmp'] = create_if_not_exists(self.input_path / "tmp", overwrite=True)
            mask_path = self.dir_structure['tmp'] / f"{image.name}_mask.tif"
            mask_da.rio.to_raster(mask_path)
            print(f"Saved mask for {image.name}.")
        
        return mask_da

    
    def generate_tiles(self, tile_size, write_tmp_files=False):
        """
        This method tiles both images and masks (if any) and stores them as .tif files.

        The tiled images are saved in the 'image_tiles' directory and the tiled masks (if any) are saved in the 'mask_tiles' directory.
        The naming convention for the tiled images and masks is '{original_file_name}_r{row_index}_c{column_index}.tif'.

        Parameters:
        tile_size: Size of the output tiles.
        write_tmp_files: True if the mask should be stored before tiling. 
        
        """

        for image in self.images:
            print(f"Tiling image {image.name}...")
            # Load image and corresponding mask as numpy array and retrieve their shape

            if self.labels is not None:
                mask = self.generate_mask(image, write_tmp_files)

            x_tiles = image.sizes['x'] // tile_size
            y_tiles = image.sizes['y'] // tile_size
            total_tiles = x_tiles * y_tiles

            if total_tiles == 0:
                raise IOError(f"tile_size is bigger than the input image for {image.name} ({image.sizes['x']}, {image.sizes['y']}). \
                              Please choose a smaller tile size or a different image.")

            # Cut image and mask into tiles and store them as .tif-files
            for i in range(x_tiles):
                for j in range(y_tiles):

                    img_tile = image.isel(x=slice(i*tile_size, (i+1)*tile_size), y=slice(j*tile_size, (j+1)*tile_size))
                    tile_path = self.dir_structure['image_tiles'] / f'{image.name}_r{i}_c{j}.TIF'
                    img_tile.rio.to_raster(tile_path)

                    if self.labels is not None:
                        msk_tile = mask.isel(x=slice(i*tile_size, (i+1)*tile_size), y=slice(j*tile_size, (j+1)*tile_size))
                        msk_path = self.dir_structure['mask_tiles'] / f'{image.name}_r{i}_c{j}.TIF'
                        msk_tile.rio.to_raster(msk_path)
            
            print(f"Tiled {image.name} into {total_tiles} tiles in folder `tiles/images`.")
            
            if self.labels is not None:
                print(f"Generated {total_tiles} binary mask tiles in folder `tiles/masks`.")
