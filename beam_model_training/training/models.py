from fastai.vision.all import *
from semtorch import get_segmentation_learner
from utils.helpers import timestamp

from training.training_functions import get_tile_size, map_unique_classes, batch_size, map_unique_classes, get_mask, \
    callbacks
from utils.my_paths import CODES, SEED
from training.losses import DualFocalLoss, CombinedLoss

def train(train_dir, model_dir, tile_type, backbone, fit_type, epochs, architecture='HRNet', split=.2, bs=None, loss_function=None):
    """
    This function trains a model based on the given parameters. It supports both HRNet and U-Net architectures.
    It applies image augmentations, creates dataloaders, sets up the model, and finally trains it.

    Parameters:
    train_dir (PosixPath): Path to the model training files.
    model_dir (str): Path to save the model.
    tile_type (str): Type of the tile used for segmentation.
    backbone (str): The name of the backbone model to use.
    fit_type (str): Type of fitting method to use.
    epochs (int): Number of epochs for training.
    architecture (str, optional): Name of the architecture to use. Defaults to 'HRNet'.
    split (float, optional): Fraction of data to use as validation set. Defaults to .2.
    bs (int, optional): Batch size. If None, it will be determined automatically based on the backbone and tile size.
    loss_function (str, optional): Loss function to use for U-Net. Not applicable for HRNet.

    Returns:
    learn: Trained model.
    dls: Dataloaders used for training.
    """

    tile_size = get_tile_size(tile_type)

    tfms = [*aug_transforms(mult=1.0, do_flip=True, flip_vert=True, max_rotate=40.0, min_zoom=1.0, max_zoom=1.4, max_warp=0.4),
            Normalize.from_stats(*imagenet_stats),
            Brightness(max_lighting=0.5),
            Contrast(max_lighting=0.5),
            Hue(max_hue=0.2),
            Saturation(max_lighting=0.5)]
    
    image_files = get_image_files(train_dir / "images")
    mask_files = get_image_files(train_dir / "masks")

    # Get codes of masks
    p2c_map = map_unique_classes(mask_files)

    if bs == None:
        bs = batch_size(backbone, tile_size)

    dls = SegmentationDataLoaders.from_label_func(train_dir / "images", image_files, label_func=lambda x: get_mask(x, p2c_map), bs=bs, codes=CODES, seed=SEED,
                                                  batch_tfms=tfms,
                                                  valid_pct=split)

    if architecture.lower() == 'hrnet':
        learner = get_segmentation_learner(dls, number_classes=2, segmentation_type="Semantic Segmentation",
                                         architecture_name="hrnet",
                                         backbone_name=backbone, model_dir=model_dir, metrics=[Dice(), JaccardCoeff()],
                                         splitter=trainable_params, pretrained=True).to_fp16()
    elif architecture.lower() == 'u-net':
        loss_functions = {'Dual_Focal_loss': DualFocalLoss(), 'CombinedLoss': CombinedLoss(),
                          'DiceLoss': DiceLoss(), 'FocalLoss': FocalLoss(), None: None}
        backbones = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50,
                     'resnet101': resnet101, 'vgg16_bn': vgg16_bn}
        learner = unet_learner(dls, backbones.get(backbone), n_out=2, loss_func=loss_functions.get(loss_function), metrics=[Dice(), JaccardCoeff()]).to_fp16()

    learner.fit_one_cycle(epochs, cbs=callbacks(model_dir, architecture, backbone, fit_type, timestamp))
    return learner

