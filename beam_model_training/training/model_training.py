from os.path import join

from tiling.tiling_functions import create_if_not_exists
from training.training_functions import check_dataset_balance
from hrnet_model import hrnet_model_training
from u_net_model import u_net_model_training
import torch
import os
import utils.my_paths as p

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

create_if_not_exists(p.MODEL_DIR)

tile_type = '512_512 stride'

percentages = check_dataset_balance(tile_type, augmented=False)

"""
Train U-Net

Unaugmented Training Data
"""

tile_type = '512_512 stride'
backbone = 'resnet18'
fit_type = 'One-Cycle'
epochs = 200

learn, dls = u_net_model_training(tile_type, backbone, fit_type, epochs, architecture='U-Net', augmented=False,
                                  split=.2)

"""Augmented Training Data"""

# tile_type = '512_512 stride'
# backbone = 'resnet18'
# fit_type = 'One-Cycle'
# epochs = 200

# learn, dls = u_net_model_training(tile_type, backbone, fit_type, epochs, architecture='U-Net', augmented=True, split=.2)

"""
Train HRNet

Unaugmented Training Data
"""

tile_type = '512_512 stride'
backbone = 'hrnet_w18'
fit_type = 'One-Cycle'
epochs = 200

learn, dls = hrnet_model_training(tile_type, backbone, fit_type, epochs, architecture='HRNet', augmented=False,
                                  split=.2)

"""Augmented Training Data"""

# tile_type = '512_512 stride'
# backbone = 'hrnet_w18'
# fit_type = 'One-Cycle'
# epochs = 200

# learn, dls = hrnet_model_training(tile_type, backbone, fit_type, epochs, architecture='HRNet', augmented=True, split=.2)

"""## Save models"""

model_save_path = join(p.ROOT_PATH, 'new_model', 'model_exported.pkl')
learn.export(model_save_path)
