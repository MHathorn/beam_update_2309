# Import packages
from os.path import join

import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import mapping, Point, Polygon
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from fastai.vision.all import *
from tqdm import tqdm
import cv2

import utils.my_paths as p
from os.path import join


def create_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def tile_img_msk(image_path, tile_size: int):
    """
     Loop through all images in the given folder,splits images and corresponding masks into smaller tiles and save
     them.
    :param image_path:
    :param tile_size:
    :return:
    """
    for fn in tqdm(image_path):
        output_path = fn.parent.parent
        mask_tiles_path = join(output_path, 'mask_tiles')
        create_if_not_exists(mask_tiles_path)
        img = np.array(PILImage.create(fn))
        msk_fn = str(fn).replace('images', 'untiled masks')
        msk = np.array(PILMask.create(msk_fn))
        x, y, _ = img.shape
        # Cut tiles and save them
        for i in range(x // tile_size):
            for j in range(y // tile_size):
                img_tile = img[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size]
                msk_tile = msk[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size]
                Image.fromarray(img_tile).save(join(output_path, 'image_tiles', f'{fn.name[:-4]}_{i}_{j}.png'))
                Image.fromarray(msk_tile).save(
                    join(output_path, f'mask_tiles', f'{fn.name[:-4]}_{i}_{j}.png'))


def tile_img(image_path, output_path, tile_size: int, single=None):
    '''Tile the image into smaller tiles and save them'''
    if not single:
        for fn in tqdm(image_path):
            # Create output directory if it doesn't already exist
            create_if_not_exists(output_path)

            # Create mask for current image
            img = np.array(PILImage.create(fn))
            x, y, _ = img.shape

            # Cut tiles and save them
            for i in range(x // tile_size):
                for j in range(y // tile_size):
                    img_tile = img[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size]
                    create_if_not_exists(join(output_path, fn.name[:-4]))
                    Image.fromarray(img_tile).save(join(output_path, fn.name[:-4], f'{fn.name[:-4]}_{i}_{j}.png'))

    if single == True:
        # Create directories
        create_if_not_exists(output_path)

        # Create mask for current image
        img = np.array(PILImage.create(image_path))
        x, y, _ = img.shape

        # Cut tiles and save them
        # todo fix this to work with all types of paths!
        # fn = image_path.split("/")[-1][:-4]
        fn = image_path.split("\\")[-1][:-4]
        for i in range(x // tile_size):
            for j in range(y // tile_size):
                img_tile = img[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size]
                Image.fromarray(img_tile).save(join(output_path, fn, f'{fn}_{i}_{j}.png'))

    # Generate the mask


def poly_from_utm(polygon, transform):
    poly_pts = []
    poly = unary_union(polygon)
    for i in np.array(poly.exterior.coords):
        poly_pts.append(~ transform * tuple(i))
    new_poly = Polygon(poly_pts)
    return new_poly


def generate_mask(raster_path, shape_path, output_path=None, file_name=None):
    '''Function that generates a binary mask from a vector file (shp or geojson)
    raster_path = path to the .tif;
    shape_path = path to the shapefile or GeoJson.
    output_path = Path to save the binary mask.
    file_name = Name of the file.'''
    # Load raster
    with rasterio.open(raster_path, "r") as src:
        raster_img = src.read()
        raster_meta = src.meta
    # Load shapefile
    train_df = gpd.read_file(shape_path)
    # Verify CRS of the raster file
    if train_df.crs != src.crs:
        print(f'Raster CRS: {src.crs}, Vector CRS: {train_df.crs}.\n Convert vector and raster to the same CRS.')
    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in train_df.iterrows():
        if row['geometry'].geom_type == 'MultiPolygon':
            for p in row['geometry'].geoms: # iterate over polygons within a MultiPolygon
                poly = poly_from_utm(p, src.meta['transform'])
                poly_shp.append(poly)
        elif row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            # raise an error or skip the object
            raise TypeError("Invalid geometry type")
        
    if len(poly_shp) > 0:
        mask = rasterize(shapes=poly_shp, out_shape=im_size)
    else:
        mask = np.zeros(im_size)

    # Save or show mask after applyin erosion
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = mask.astype('uint8')
    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})
    if (output_path != None and file_name != None):
        os.chdir(output_path)
        with rasterio.open(file_name, 'w', **bin_mask_meta) as dst:
            dst.write(mask * 255, 1)  # Change 255 to 1 if classes need to be 0 and 1
    else:
        return mask


def save_masks(images, mask, maskdir):
    create_if_not_exists(maskdir)
    for image in tqdm(images):
        if image.name.endswith(('.TIF', '.tif')):
            shapes = image.name
            generate_mask(image, mask, maskdir, shapes)


