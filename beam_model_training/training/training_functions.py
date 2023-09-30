from fastai.vision.all import *
import numpy as np
import random
import os
import pytz
from datetime import datetime
from semtorch import get_segmentation_learner
from IPython.display import Audio, display
from sklearn.model_selection import train_test_split
from os.path import join

import utils.my_paths as p


# Custom loss functions
class CombinedLoss:
    """
    Dice and Focal combined
    """

    def __init__(self, axis=1, smooth=1., alpha=1.):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)

    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):    return x.argmax(dim=self.axis)

    def activation(self, x): return F.softmax(x, dim=self.axis)


class DualFocalLoss(nn.Module):
    """
    This loss is proposed in this paper: https://arxiv.org/abs/1909.11932
    """

    def __init__(self, ignore_lb=255, eps=1e-5, reduction='mean'):
        super(DualFocalLoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.eps = eps
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, logits, label):
        ignore = label.data.cpu() == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1).detach()

        pred = torch.softmax(logits, dim=1)
        loss = -torch.log(self.eps + 1. - self.mse(pred, lb_one_hot)).sum(dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss
        return loss


# Set path of root folder of images and masks
path = Path(join(p.ROOT_PATH, 'aerial'))


# path = Path(f'C:\Users\dmz-admin\Desktop\BEAM_training_material\Data Segmentation\aerial')

# Set codes
# codes = ['Background', 'Building']


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
    fn = str(fn).replace('image_tiles', 'buildings_mask_tiles')
    msk = np.array(PILMask.create(fn))
    mx = np.max(msk)
    for i, val in enumerate(p2c):
        msk[msk == p2c[i]] = val
    return PILMask.create(msk)


def get_msk_augmented(fn, p2c):
    '''Grab a mask from a `filename` and adjust the pixels based on `pix2class`'''
    fn = str(fn).replace('img', 'lbl')
    msk = np.array(PILMask.create(fn))
    mx = np.max(msk)
    for i, val in enumerate(p2c):
        msk[msk == p2c[i]] = val
    return PILMask.create(msk)


def get_y(o):
    return get_msk(o, p2c)


def get_y_augmented(o):
    return get_msk_augmented(o, p2c)


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


def check_fnames_lbls(tile_type, augmented=None):
    '''Get images and labels for dataloader and check whether their number is equal'''
    global fnames, lbl_names, path
    if augmented == False:
        # path = Path(f'C:\Users\dmz-admin\Desktop\BEAM_training_material\Data Segmentation\aerial')  # change this to your data
        # old structure: fnames = get_image_files(f'{path}/image_tiles/2019_10cm_RGB_BE_67/{tile_type}')
        # old structure: lbl_names = get_image_files(f'{path}/buildings_mask_tiles/2019_10cm_RGB_BE_67/{tile_type}')
        fnames = get_image_files(join(p.ROOT_PATH, 'aerial', 'image_tiles'))
        lbl_names = get_image_files(join(p.ROOT_PATH, 'aerial', 'buildings_mask_tiles'))
    elif augmented == True:
        path = Path(f'/content/drive/MyDrive/Segmentation Data/aerial/augmented/8/0.2')  # change this to your data
        fnames = get_image_files(path / 'img')
        lbl_names = get_image_files(path / 'lbl')
    if len(fnames) != len(lbl_names):
        print('ERROR: unequal number of image and mask tiles!')
    return fnames, lbl_names, path


def callbacks(model_dir, architecture, backbone, fit_type, timestamp):
    '''Log results in CSV, show progress, and stop early if dice coefficient doesn't improve for 10 epochs'''
    cbs = [CSVLogger(fname=f'{model_dir}/{architecture}_{backbone}_{fit_type}_{timestamp()}.csv', append=True),
           ShowGraphCallback()]
    return cbs


def check_dataset_balance(tile_type, augmented=None):
    '''Check, how balanced the dataset is'''
    global tile_size, p2c
    tile_size = get_tile_size(tile_type)

    # Check if there is a label for each image
    fnames, lbl_names, path = check_fnames_lbls(tile_type, augmented)
    # Get codes of masks
    p2c = n_codes(lbl_names)

    if augmented == False:
        label_func = get_y
    elif augmented == True:
        label_func = get_y_augmented

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
