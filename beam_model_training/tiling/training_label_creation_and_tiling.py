from os.path import join
import geopandas as gpd

from tiling_functions import get_image_files, save_masks, tile_img_msk, tile_img, create_if_not_exists
import utils.my_paths as p

imagery_type = 'aerial'  # "satellite"
mask_type = 'buildings'  # "settlements"
path = join(p.ROOT_PATH, imagery_type)
# Set mask path
if mask_type == "settlements":
    shp_path = join(path, 'untiled masks', 'shapefiles', 'BuildingFootprints_4326.shp')
elif mask_type == "buildings":
    shp_path = join(path, 'untiled masks', 'shapefiles', 'manually labelled', '2020_RGB_10cm_CJ_063.shp')
mask_plot = gpd.read_file(shp_path)
mask_plot["geometry"].plot()

# Set directory containing images and size of tiles to produce
images_list = get_image_files(join(path, "images"))
print(f'Number of images: {len(images_list)}')

# Create masks corresponding to selected images
masks_path = join(path, 'untiled masks', mask_type)
save_masks(images_list, shp_path, masks_path)
# Tile images and masks with a stride of 0 pixels
tile_size = 512
tile_img_msk(images_list, mask_type, tile_size)
tile_size = 500
output_path = join(path, "inference", "input")
create_if_not_exists(str(output_path))

tile_img(images_list, output_path, tile_size, single=False)

# Sanity check
print(len(get_image_files(output_path)))
print(len(images_list * 400))
