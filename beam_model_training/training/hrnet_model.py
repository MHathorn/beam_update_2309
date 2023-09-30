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
    get_y_augmented, callbacks, timestamp
import utils.my_paths as p


def hrnet_model_training(tile_type, backbone, fit_type, epochs, architecture='HRNet', augmented=None, split=.2,
                         bs=None):
    global tile_size, p2c, loss
    seed()

    tile_size = get_tile_size(tile_type)
    fnames, lbl_names, path = check_fnames_lbls(tile_type, augmented)

    if bs == None:
        bs = batch_size(backbone, tile_size)

    # Get codes of masks
    p2c = n_codes(lbl_names)

    if augmented == False:
        # Create function to load images and masks
        dls = SegmentationDataLoaders.from_label_func(path, fnames, label_func=get_y, bs=bs, codes=p.CODES, seed=2022,
                                                      batch_tfms=[Normalize.from_stats(*imagenet_stats)],
                                                      valid_pct=split)
    elif augmented == True:
        splitter = FuncSplitter(lambda fn: Path(fn).parent.name == 'valid')
        db = DataBlock(blocks=(ImageBlock, MaskBlock(p.CODES)), get_items=get_image_files, splitter=splitter,
                       get_y=get_y_augmented)
        dls = db.dataloaders(path / 'img', bs=bs, valid_pct=split)

    learn = get_segmentation_learner(dls, number_classes=2, segmentation_type="Semantic Segmentation",
                                     architecture_name="hrnet",
                                     backbone_name=backbone, model_dir=p.MODEL_DIR, metrics=[Dice()],
                                     splitter=trainable_params, pretrained=True).to_fp16()

    learn.fit_one_cycle(epochs, cbs=callbacks(p.MODEL_DIR, architecture, backbone, fit_type, timestamp))
    return learn, dls
