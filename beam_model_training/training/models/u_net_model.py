from fastai.vision.all import *
import numpy as np
import random
from sklearn.model_selection import train_test_split
from training.losses import CombinedLoss, DualFocalLoss

from training.training_functions import seed, get_tile_size, check_fnames_lbls, batch_size, map_unique_classes, get_y, \
    get_y_augmented, callbacks, timestamp, DualFocalLoss
from utils.my_paths import CODES, MODEL_DIR, SEED


def u_net_model_training(tile_type, backbone, loss_function, fit_type, epochs, architecture='U-Net', split=.2):
    """Create list of files and masks, a dataloader, a model, callbacks, and trains the final U-Net model"""
    global tile_size, p2c, loss
    tile_size = get_tile_size(tile_type)
    # Create image augmentations on the fly
    tfms = [*aug_transforms(mult=1.0, do_flip=True, flip_vert=True, max_rotate=40.0, min_zoom=1.0, max_zoom=1.4, max_warp=0.4),
    Normalize.from_stats(*imagenet_stats),
    Brightness(max_lighting=0.5),
    Contrast(max_lighting=0.5),
    Hue(max_hue=0.2),
    Saturation(max_lighting=0.5)]

    # Check if there is a label for each image
    fnames, lbl_names, path = check_fnames_lbls(tile_type)

    # Get codes of masks
    p2c = map_unique_classes(lbl_names)

    # Set batch size depending on image size and backbone used
    batch_size = batch_size(backbone, tile_size)

    # Create dataloader to load images and masks
    dls = SegmentationDataLoaders.from_label_func(path, fnames, label_func=get_y, valid_pct=split,
                                                bs=batch_size, codes=CODES, seed=SEED, batch_tfms=tfms)

    # Show, which loss function is used for the experiment and set the variable accordingly
    print('loss_function: ', loss_function)
    loss_functions = {'Dual_Focal_loss': DualFocalLoss(), 'CombinedLoss': CombinedLoss(),
                        'DiceLoss': DiceLoss(), 'FocalLoss': FocalLoss(), None: None}

  # Create U-Net model with the selected backbone
    backbones = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50,
                 'resnet101': resnet101, 'vgg16_bn': vgg16_bn}
    learn = unet_learner(dls, backbones.get(backbone), n_out=2, loss_func=loss_functions.get(loss_function), metrics=[Dice(), JaccardCoeff()]).to_fp16()

  # Fit the model
    learn.fit_one_cycle(epochs, cbs=callbacks(MODEL_DIR, architecture, backbone, fit_type, timestamp))
    return learn, dls

