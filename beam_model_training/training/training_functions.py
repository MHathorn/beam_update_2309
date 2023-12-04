import random
from datetime import datetime
from os.path import join

import numpy as np
import pytz
import rioxarray as rxr
import xarray as xr
from fastai.vision.all import *
from IPython.display import Audio, display
from semtorch import get_segmentation_learner
from sklearn.model_selection import train_test_split

# Set path of root folder of images and masks


def map_unique_classes(file_names, is_partial=False):
    """Gather unique classes from a list of file names"""

    if is_partial and len(file_names) > 10:
        file_names = random.sample(file_names, 10) 

    # Get unique classes from file names
    masks = [rxr.open_rasterio(file_path) for file_path in file_names]
    unique_classes = np.unique([class_value for mask in masks for class_value in np.unique(mask)])

    # Convert into a dictionary mapping index to class value
    pixel_to_class = {i: class_value for i, class_value in enumerate(unique_classes)}
    return pixel_to_class


def get_mask(image_path, pixel_to_class):
    """Get mask from an image path and adjust the pixels based on p2c"""
    # new structure: 
    mask_path = str(image_path).replace("images", "masks")
    mask = rxr.open_rasterio(mask_path)
    
    for i, val in enumerate(pixel_to_class):
        mask = xr.where(mask == pixel_to_class[i], val, mask)
    mask = mask.values.reshape((mask.shape[1], mask.shape[2]))
    
    return PILMask.create(mask)


def batch_size(backbone, tile_size):
  """Automatically set batch size depending on image size and architecture used"""
  if tile_size == 512:
    batch_size_dict = {'resnet152': 2, 'resnet101': 2, 'resnet50': 2, 
                       # Change batch size for used backbone if you run into CUDA out of memory errors
                       'resnet34': 11, 'resnet18': 8, 'vgg16_bn': 2,
                       'hrnet_w18': 32, 'hrnet_w30': 32, 'hrnet_w32': 32,
                       'hrnet_w48': 18}
  elif tile_size == 256:
    batch_size_dict = {'resnet152': 2, 'resnet101': 2, 'resnet50': 2,
                       'resnet34': 11, 'resnet18': 10, 'hrnet_w18': 64}
  return batch_size_dict[backbone]


def model_notification():
    """Create notification when model training is completed"""
    for i in range(5):
        display(Audio('https://www.soundjay.com/buttons/beep-03.wav', autoplay=True))
        time.sleep(2)


def get_tile_size(tile_type):
    """Extract tile size from type of tiles passed to the model."""
    if '512' in tile_type:
        tile_size = '512'
    elif '256' in tile_type:
        tile_size = '256'
    return tile_size


def callbacks(model_dir, architecture, backbone, fit_type, timestamp):
    """Log results in CSV, show progress in graph"""
    cbs = [CSVLogger(fname=f'{model_dir}/{architecture}_{backbone}_{fit_type}_{timestamp()}.csv', append=True),
           ShowGraphCallback()]
    return cbs


def check_dataset_balance(images_dir, masks_dir, tile_size, codes, seed):
    """Check balance of the dataset."""
  
    fnames = get_image_files(images_dir) 
    lbl_names = get_image_files(masks_dir)

    # Get codes of masks
    p2c_map = map_unique_classes(lbl_names)

    # Create dataloader to check building pixels
    dls = SegmentationDataLoaders.from_label_func(images_dir, fnames, label_func=lambda x: get_mask(x, p2c_map), bs=2, codes=codes, seed=seed)

    targs = torch.zeros((0, tile_size, tile_size))
    # issue here with // execution
    for _, masks in dls[0]:
        targs = torch.cat((targs, masks.cpu()), dim=0)

    total_pixels = targs.shape[1] ** 2
    percentages = torch.count_nonzero(targs, dim=(1, 2)) / total_pixels
    plt.hist(percentages, bins=20)
    plt.ylabel('Number of tiles')
    plt.xlabel('Ratio of pixels that are of class `building`')
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['right'].set_color('none')
    plt.show()
    print(f'Mean Percentage of Pixels Belonging to Buildings: {round(percentages.mean().item(), 3)}')
    return percentages
