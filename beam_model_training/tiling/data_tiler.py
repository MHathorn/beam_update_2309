from pathlib import Path
import shutil
import sys
import rasterio
from rasterio.features import rasterize
import numpy as np
import pandas as pd
import geopandas as gpd
import os

from shapely import wkt
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from PIL import Image
from cv2 import erode

class DataTiler:
    """Data Tiler class. This class takes in an image, tile size, and tile directory, and populates
    the tile directory with image tiles ready for data augmentation and train-test split. 

    Usage:
    img_tiler = DataTiler(image_path, output_dir, labels_path) (labels_path optional)
    img_tiler.generate_tiles(tile_size)
    """
    def __init__(self, image_path, output_dir, vector_labels=None):
            self.image_path = Path(image_path)
            self.output_dir = Path(output_dir)
            self.has_labels = (vector_labels is not None)
            self.vector_labels = Path(vector_labels) if self.has_labels else None
            self.directory_structure = {
                'image_tiles': self.create_subdirectory(output_dir / 'image_tiles'),
                'tmp': self.create_subdirectory(output_dir / 'tmp')
            }
            
            # Prepare data for tiling
            self.load_image(image_path)
            if self.has_labels:
                self.load_labels(vector_labels)
                print(f"Loaded vector labels from {self.vector_labels.name}.")
                self.directory_structure['mask_tiles'] = self.create_subdirectory(output_dir / 'mask_tiles')
            else:
                print(f"Warning: No mask file provided. Tiling images alone.")

    def load_image(self, image_path):
        """
        Loads an RGB image from self.image_path. 

        Note: Only supporting Worldview-3 so far.
        TODO: Expand to aerial, other satellites if/when needed.
        TODO: Keep georeferences throughout tiling and training.

        Returns:
            np.array: RGB bands stacked in a numpy array.
        """
        with rasterio.open(image_path) as image:
            if image.count == 8:
                self.image = image.read([4, 2, 1])
                print(f"Loaded RGB image from Worldview-3-identified imagery in {self.image_path.name}.")
            elif image.count == 4:
                self.image = image.read()
                print(f"Loaded RGB image from aerial imagery  in {self.image_path.name}.")
            else:
                raise TypeError("The imagery provided is not supported yet.")
            self.image_meta = image.meta


    def load_labels(self, vector_labels):
        """
        This loads building footprints from a vector file and stores them as an object attribute.
        TODO: Add support for other mask types beyond Open Buildings Dataset.
        """
        if vector_labels.suffix.lower() == '.csv':
            df = pd.read_csv(vector_labels)
            df['geometry'] = df['geometry'].apply(wkt.loads)
            buildings = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        elif vector_labels.suffix.lower() == '.shp':
            buildings = gpd.read_file(vector_labels)
        self.labels = self.crop_buildings(buildings)
            
    
    def crop_buildings(self, buildings):
        """
        Crop the building geometries to the bounding box of the satellite image.

        Parameters:
        buildings (GeoDataFrame): A GeoDataFrame containing the building geometries.

        Returns:
        GeoDataFrame: A GeoDataFrame containing only the building geometries that intersect with 
                    the bounding box of the input image.
        """
        with rasterio.open(self.image_path) as sat_image:
            bounds = sat_image.bounds
            bounding_box = box(*bounds)
            buildings = buildings.to_crs(sat_image.crs)

        return buildings[buildings.intersects(bounding_box)]


    def create_subdirectory(self, dir):
        """
        Create a subdirectory if it does not exist. If the directory exists and is not empty,
        it prompts the user to confirm whether they want to overwrite all files in the directory.

        Parameters:
        dir (str): The path of the directory to be created.

        Returns:
        PosixPath: The path of the created directory if successful, otherwise it terminates the program.

        Raises:
        SystemExit: If the user cancels the operation when prompted to confirm overwriting of non-empty directory.
        """
        if not os.path.exists(dir):
            os.makedirs(dir)
        elif len(os.listdir(dir)) > 0:
            confirm = input(f"Directory {dir.name} is not empty, are you sure you want to overwrite all files? (y/n): ")
            if confirm.lower() == 'y':
                shutil.rmtree(dir)  # Delete the directory and its contents
                os.makedirs(dir)   
            else:
                print("Operation cancelled by user.")
                sys.exit()
        return dir
         

    
    def generate_mask(self, write=True):
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
        image_size = (self.image_meta['height'], self.image_meta['width'])
        for _, row in self.labels.iterrows():
            if row['geometry'].geom_type == 'MultiPolygon':
                for label_geom in row['geometry'].geoms: 
                    # iterate over polygons within a MultiPolygon
                    label = poly_from_utm(label_geom, self.image_meta['transform'])
                    label_polygons.append(label)
            elif row['geometry'].geom_type == 'Polygon':
                label = poly_from_utm(row['geometry'], self.image_meta['transform'])
                label_polygons.append(label)
            else:
                # raise an error or skip the object
                raise TypeError("Invalid geometry type")

        if len(label_polygons) > 0:
            mask = rasterize(shapes=label_polygons, out_shape=image_size)
        else:
            mask = np.zeros(image_size)

        # Save or show mask
        kernel = np.ones((3, 3), np.uint8)
        mask = erode(mask, kernel, iterations=1)
        mask = mask.astype('uint8') * 255 # Change 255 to 1 if classes need to be 0 and 1
        if write:
            mask_meta = self.image_meta.copy()
            mask_meta.update({'count': 1})

            mask_path = self.directory_structure['tmp'] / f"{self.image_path.stem}_mask{self.image_path.suffix}"
            
            with rasterio.open(mask_path, 'w', **mask_meta) as dst:
                dst.write(mask, 1) # Change 255 to 1 if classes need to be 0 and 1
            print(f"Saved mask for {self.image_path.name}.")
        else:
            return mask
    
    def generate_tiles(self, tile_size):
        """
        This method tiles both images and masks (if any) and stores them as .png files.

        The tiled images are saved in the 'image_tiles' directory and the tiled masks (if any) are saved in the 'mask_tiles' directory.
        The naming convention for the tiled images and masks is '{original_file_name}_r{row_index}_c{column_index}.png'.
        
        """

        if self.image_path.suffix.lower() in ['.tif', '.tiff']:
            print(f"Tiling image {self.image_path}...")
            # Load image and corresponding mask as numpy array and retrieve their shape

            if self.has_labels:
                mask = self.generate_mask(False)
            _, x, y = self.image.shape
            print(x, y)

            x_tiles = x // tile_size
            y_tiles = y // tile_size

            # Cut image and mask into tiles and store them as .png-files
            for i in range(x_tiles):
                for j in range(y_tiles):

                    img_tile = self.image[:, i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
                    Image.fromarray(img_tile.transpose((1, 2, 0))).save(self.directory_structure['image_tiles'] / f'{self.image_path.stem}_r{i}_c{j}.png')
                    if self.has_labels:
                        msk_tile = mask[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
                        Image.fromarray(msk_tile).save(self.directory_structure['mask_tiles'] / f'{self.image_path.stem}_r{i}_c{j}.png')
            
            print(f"Tiled {self.image_path.name} into {x_tiles * y_tiles} tiles in folder `image_tiles`.")
            if self.has_labels:
                print(f"Generated {x_tiles * y_tiles} binary mask tiles in folder `mask_tiles`.")


            