from os.path import join
import geopandas as gpd

from tiling_functions import get_image_files, save_masks, tile_img_msk, tile_img, create_if_not_exists
import utils.my_paths as p


path = p.ROOT_PATH
# Set mask path
shp_path = p.LABEL_PATH
mask_plot = gpd.read_file(shp_path)
mask_plot["geometry"].plot()

# Set directory containing images and size of tiles to produce
images_list = get_image_files(join(path, "images"))
print(f'Number of images: {len(images_list)}')

# Create masks corresponding to selected images
masks_path = join(path, 'untiled masks')
save_masks(images_list, shp_path, masks_path)
# Tile images and masks with a stride of 0 pixels
tile_size = 512
tile_img_msk(images_list, tile_size)
