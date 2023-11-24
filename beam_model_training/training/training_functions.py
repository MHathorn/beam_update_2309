from fastai.vision.all import *
import numpy as np
import random
import pytz
from datetime import datetime
from semtorch import get_segmentation_learner
from IPython.display import Audio, display
from sklearn.model_selection import train_test_split
from os.path import join

import utils.my_paths as p



# Set path of root folder of images and masks
path = p.ROOT_PATH

def n_codes(fnames, is_partial=True):
    '''Gather the codes from a list of fnames'''
    vals = set()
    if is_partial:
        random.shuffle(fnames)
        fnames = fnames[:10]
    for fname in fnames:
        msk = np.array(PILMask.create(fname))
        for val in np.unique(msk):
            if val not in vals:
                vals.add(val)
    vals = list(vals)
    p2c = dict()
    for i, val in enumerate(vals):
        p2c[i] = vals[i]
    return p2c


def get_msk(fn, p2c):
    '''Grab a mask from a filename and adjust the pixels based on p2c'''
    pix2class = n_codes(lbl_names)
    # old structure: fn = f'{path}/buildings_mask_tiles/2019_10cm_RGB_BE_67/{tile_type}/{fn.stem[:-3]}lbl{fn.suffix}'
    fn = str(fn).replace('image_tiles', 'mask_tiles')
    msk = np.array(PILMask.create(fn))
    mx = np.max(msk)
    for i, val in enumerate(p2c):
        msk[msk == p2c[i]] = val
    return PILMask.create(msk)


def get_y(o):
    return get_msk(o, p2c)


def batch_size(backbone, tile_size):
  '''Automatically set batch size depending on image size and architecture used'''
  if '512' in tile_size:
    batch_size_dict = {'resnet152': 2, 'resnet101': 2, 'resnet50': 2, 
                       # Change batch size for used backbone if you run into CUDA out of memory errors
                       'resnet34': 11, 'resnet18': 8, 'vgg16_bn': 2,
                       'hrnet_w18': 32, 'hrnet_w30': 32, 'hrnet_w32': 32,
                       'hrnet_w48': 18}
  elif '256' in tile_size:
    batch_size_dict = {'resnet152': 2, 'resnet101': 2, 'resnet50': 2,
                       'resnet34': 11, 'resnet18': 10, 'hrnet_w18': 64}
  return batch_size_dict[backbone]


def timestamp():
    '''Timestamp for conducting experiments'''
    tz = pytz.timezone('Europe/Berlin')
    date = str(datetime.now(tz)).split(" ")
    date_time = f"{date[0]}_{date[1].split('.')[0][:5]}"
    return date_time


def model_notification():
    '''Create notification when model training is completed'''
    for i in range(5):
        display(Audio('https://www.soundjay.com/buttons/beep-03.wav', autoplay=True))
        time.sleep(2)


def get_tile_size(tile_type):
    if '512' in tile_type:
        tile_size = '512'
    elif '256' in tile_type:
        tile_size = '256'
    return tile_size


def check_fnames_lbls(tile_type):
    '''Get images and labels for dataloader and check whether their number is equal'''
    global fnames, lbl_names, path
    fnames = get_image_files(join(p.ROOT_PATH, 'image_tiles'))
    lbl_names = get_image_files(join(p.ROOT_PATH, 'mask_tiles'))
    if len(fnames) != len(lbl_names):
        print('ERROR: unequal number of image and mask tiles!')
    return fnames, lbl_names, path


def callbacks(model_dir, architecture, backbone, fit_type, timestamp):
    '''Log results in CSV, show progress in graph'''
    cbs = [CSVLogger(fname=f'{model_dir}/{architecture}_{backbone}_{fit_type}_{timestamp()}.csv', append=True),
           ShowGraphCallback()]
    return cbs


def check_dataset_balance(tile_type):
    '''Check, how balanced the dataset is'''
    global tile_size, p2c
    tile_size = get_tile_size(tile_type)

    # Check if there is a label for each image
    fnames, lbl_names, path = check_fnames_lbls(tile_type)
    # Get codes of masks
    p2c = n_codes(lbl_names)

    label_func = get_y


    # Create dataloader to check building pixels
    dls = SegmentationDataLoaders.from_label_func(path, fnames, label_func=label_func, bs=64, codes=p.CODES, seed=2)

    targs = torch.zeros((0, 512, 512))
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


def seed():
    # Create Seed for Reproducibility
    number_of_the_seed = 2022
    random.seed(number_of_the_seed)
    set_seed(number_of_the_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
