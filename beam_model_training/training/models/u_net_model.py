from fastai.vision.all import *
import numpy as np
import random
import os
import pytz
from datetime import datetime
from semtorch import get_segmentation_learner
from IPython.display import Audio, display
from sklearn.model_selection import train_test_split

from training.training_functions import seed, get_tile_size, check_fnames_lbls, batch_size, n_codes, get_y, \
    get_y_augmented, callbacks, timestamp, DualFocalLoss
import utils.my_paths as p


def u_net_model_training(tile_type, backbone, fit_type, epochs, architecture='U-Net', split=.2):
    '''Create list of files and masks, a dataloader, a model, callbacks, and train final model'''
    global tile_size, p2c, loss
    tile_size = get_tile_size(tile_type)
    # Create additional image augmentations

    tfms = [*aug_transforms(mult=1.0, do_flip=True, flip_vert=True, max_rotate=40.0, min_zoom=1.0, max_zoom=1.4, max_warp=0.4),
    Normalize.from_stats(*imagenet_stats),
    Brightness(max_lighting=0.5),
    Contrast(max_lighting=0.5),
    Hue(max_hue=0.2),
    Saturation(max_lighting=0.5)]

    # Check if there is a label for each image
    fnames, lbl_names, path = check_fnames_lbls(tile_type)

    # Get codes of masks
    p2c = n_codes(lbl_names)

    # Automatically set batch size depending on image size and backbone used
    bs = batch_size(backbone, tile_size)

    dls = SegmentationDataLoaders.from_label_func(path, fnames, label_func=get_y, valid_pct=split, bs=bs,
                                                    codes=p.CODES, seed=2, batch_tfms=tfms)

    # Create model
    if backbone == 'resnet18':
        learn = unet_learner(dls, resnet18, n_out=2, loss_func=None, metrics=[Dice(), JaccardCoeff()]
                             # Dice coefficient since dataset is imbalanced
                             ).to_fp16()  # 16-bits floats, which take half the space in RAM
    elif backbone == 'resnet34':
        learn = unet_learner(dls, resnet34, n_out=2, loss_func=None, metrics=[Dice(), JaccardCoeff()]).to_fp16()
    elif backbone == 'resnet50':
        learn = unet_learner(dls, resnet50, n_out=2, loss_func=None, metrics=[Dice(), JaccardCoeff()]).to_fp16()
    elif backbone == 'resnet101':
        learn = unet_learner(dls, resnet101, n_out=2, loss_func=None, metrics=[Dice(), JaccardCoeff()]).to_fp16()

    learn.fit_one_cycle(epochs, cbs=callbacks(p.MODEL_DIR, architecture, backbone, fit_type, timestamp))
    return learn, dls
