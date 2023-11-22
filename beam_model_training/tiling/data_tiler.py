from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.features import rasterize
import rioxarray as rxr
from shapely import wkt
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from cv2 import erode

class DataTiler:
    """Data Tiler class. This class takes in an image, tile size, and tile directory, and populates
    the tile directory with image tiles ready for data augmentation and train-test split. 

    Usage:
    img_tiler = DataTiler(image_path, output_dir, labels_path) (labels_path optional)
    img_tiler.generate_tiles(tile_size)
    """
    def __init__(self, image_dir, output_dir, labels_path=None):
            self.images = self.load_images(image_dir)
            self.output_dir = Path(output_dir)
            if not self.output_dir.exists():
                self.create_subdir(output_dir)
            self.dir_structure = {
                'image_tiles': self.create_subdir(output_dir / 'image_tiles'),
            }
            
            # Prepare data for tiling
            if labels_path:
                self.dir_structure['mask_tiles'] = self.create_subdir(output_dir / 'mask_tiles')
                # Loading labels from csv / shapefile.
                labels_path = Path(labels_path)
                self.labels = self.load_labels(labels_path)
                print(f"Loaded vector labels from {labels_path.name}.")
                
            else:
                self.labels = None
                print(f"Warning: No mask file provided. Tiling images alone.")

    def load_images(self, image_dir):
        """
        Loads all GeoTIFF images in the provided image_dir with rioxarray
        
        Parameters:
        image_dir (PosixPath | str): Path to the directory containing images.

        Returns:
        xarray.DataArray: All bands stacked in a multidimensional array.
        """
        filepaths = [img_path for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"] for img_path in Path(image_dir).rglob(ext)]
        if not filepaths:
            raise IOError(f"The directory {image_dir} does not contain any GeoTIFF images.")
        
        images = [rxr.open_rasterio(img_path, default_name=img_path.stem) for img_path in filepaths]
        # Unifying crs across images
        target_crs = images[0].rio.crs
        return [img.rio.reproject(target_crs) for img in images]
            


    def load_labels(self, labels_path):
        """
        This loads building footprints from a vector file and stores them as an object attribute.
        TODO: Add support for other mask types beyond Open Buildings Dataset.

        Parameters:
        labels_path (PosixPath): Path to the file containing labels (csv or shp).
        """
        if labels_path.suffix.lower() == '.csv':
            # Expecting here Google's Open Buildings Dataset format.
            df = pd.read_csv(labels_path)
            df['geometry'] = df['geometry'].apply(wkt.loads)
            buildings = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        elif labels_path.suffix.lower() == '.shp':
            buildings = gpd.read_file(labels_path)
        return self.crop_labels(buildings)
            
    
    def crop_labels(self, buildings):
        """
        Crop the building geometries to the bounding box of the satellite image.

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


    def create_subdir(self, dir):
        """
        Create a subdirectory if it does not exist. If the directory exists and is not empty,
        files will get overwritten.

        Parameters:
        dir (PosixPath | str): The path of the directory to be created.

        Returns:
        PosixPath: The path of the created directory.

        """
        dir_path = Path(dir)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        elif any(dir_path.iterdir()):
            print(f"Warning: Output directory is not empty. Overwriting files.")
            shutil.rmtree(dir_path)  # Delete the directory and its contents
            dir_path.mkdir(parents=True)
        return dir_path
         

    
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
        kernel = np.ones((3, 3), np.uint8)
        mask = erode(mask, kernel, iterations=1)

        # Save or return mask
        if not write:
            return mask
        
        mask_meta = image.rio.meta.copy()
        mask_meta.update({'count': 1})

        mask_path = self.dir_structure['tmp'] / f"{image.name}_mask.tif"
        
        with rasterio.open(mask_path, 'w', **mask_meta) as dst:
            dst.write(mask, 1) 
        print(f"Saved mask for {image.name}.")

    
    def generate_tiles(self, tile_size):
        """
        This method tiles both images and masks (if any) and stores them as .png files.

        The tiled images are saved in the 'image_tiles' directory and the tiled masks (if any) are saved in the 'mask_tiles' directory.
        The naming convention for the tiled images and masks is '{original_file_name}_r{row_index}_c{column_index}.png'.
        
        """

        for image in self.images:
            print(f"Tiling image {image.name}...")
            # Load image and corresponding mask as numpy array and retrieve their shape

            if self.labels:
                mask = self.generate_mask(image)

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

                    if self.labels:
                        msk_tile = mask.isel(x=slice(i*tile_size, (i+1)*tile_size), y=slice(j*tile_size, (j+1)*tile_size))
                        msk_path = self.dir_structure['mask_tiles'] / f'{image.name}_r{i}_c{j}.TIF'
                        msk_tile.rio.to_raster(msk_path)
            
            print(f"Tiled {image.name} into {total_tiles} tiles in folder `image_tiles`.")
            
            if self.labels:
                print(f"Generated {total_tiles} binary mask tiles in folder `mask_tiles`.")


            