from os.path import join
from fastai.vision.all import get_image_files

from training.training_functions import check_dataset_balance
from training.models import train
import os

from utils.helpers import create_if_not_exists, seed, timestamp
from utils.my_paths import ROOT_PATH, SEED



def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    seed(SEED)

    model_dir = create_if_not_exists(ROOT_PATH / "models")
    train_dir = ROOT_PATH / "tiles/train"

    # percentages = check_dataset_balance(tile_type, augmented=False)

    tile_type = '512_with_erosion'
    backbone = 'resnet18'
    fit_type = 'one_cycle'
    epochs = 30
    architecture = 'U-Net'
    loss_function = None

    learner = train(train_dir, model_dir, tile_type, backbone, fit_type, epochs, architecture='U-Net', split=.2, bs=4, loss_function=loss_function)
    model_path = model_dir / f"{architecture}" / f"{backbone}_{timestamp()}_exported.pkl"
    learner.export(model_path)


if __name__ == '__main__':
    main()