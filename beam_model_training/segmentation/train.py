import random

import numpy as np
import rioxarray as rxr
import xarray as xr
from fastai.vision.all import *
from fastai.callback.tensorboard import TensorBoardCallback
from semtorch import get_segmentation_learner
from segmentation.losses import CombinedLoss, DualFocalLoss, CrossCombinedLoss
from utils.helpers import create_if_not_exists, load_config, seed, timestamp

# Set path of root folder of images and masks


class Trainer:
    def __init__(self, config):
        """
        Initialize the Trainer class with configuration settings.

        Args:
            config (dict): Configuration settings. Contains the following keys:
                - root_dir (str): The root directory containing all training files.
                - dirs (dict): Directories for various data and model files. Contains the following keys:
                    - images (str): The directory where the image tiles are stored.
                    - shapefiles (str): The directory where the shapefiles will be saved.
                    - predictions (str): The directory where the prediction files will be saved.
                    - models (str): The directory where model checkpoints are saved.
                - erosion (bool): Whether erosion has been applied to building labels in preprocessing.
                - seed (int): Seed for random number generator.
                - codes (list): List of unique codes.
                - tile_size (int): Size of each image tile.
                - test_size (float): Proportion of data to be used for testing.
                - train (dict): Training parameters. Contains the following keys:
                    - architecture (str): Architecture of the model.
                    - backbone (str): Backbone of the model.
                    - epochs (int): Number of epochs for training.
                    - loss_function (str): Loss function for training.
                    - batch_size (int): Batch size for training.

        Raises:
            ValueError: If tile size is not a positive integer.
            KeyError: If a necessary key is missing from the config dictionary.
        """

        try:
        # Load and initialize random seed
            self.load_params(config)
            seed(self.params["seed"])

            # Load learning arguments
            self.load_train_params(config)

            # Load dirs and create if needed
            path = Path(self.params["root_dir"])
            self.model_dir = create_if_not_exists(path / config["dirs"]["models"])
            self.train_dir = path / config["dirs"]["train"]
            self.images_dir = self.train_dir / "images"

            # Load masks and initialize mapping
            self.masks_dir = self.train_dir / "masks"
            self.p2c_map = self._map_unique_classes()
        except KeyError as e:
            raise KeyError(f"Config must have a value for {e} to run the Trainer.")

    def load_params(self, config):
        params_keys = ["seed", "codes", "tile_size", "test_size", "root_dir"]
        self.params = dict((k, config[k]) for k in params_keys)

        if not self.params["tile_size"] or not isinstance(self.params["tile_size"], int):
            raise ValueError("Tile size must be a positive integer.")

        if self.params["test_size"] < 0 or self.params["test_size"] > 1:
            raise ValueError("Test size must be a float between 0 and 1.")

    def load_train_params(self, config):
        train_keys = ["architecture", "backbone", "epochs", "loss_function", "batch_size"]
        self.train_params = {}
        for k in train_keys:
            if k == "batch_size":
                self.train_params[k] = config["train"].get(
                    "batch_size", 
                    self._get_batch_size(
                        self.params["tile_size"], 
                        config["train"]["backbone"]
                        )
                    )
            else:
                self.train_params[k] = config["train"].get(k)

        assert self.train_params["architecture"].lower() in ["u-net", "hrnet"]

    def _map_unique_classes(self, is_partial=False):
        """
        Map unique classes from a list of file names.

        Args:
            is_partial (bool, optional): Whether to sample a subset of the files. Defaults to False.

        Returns:
            dict: A dictionary mapping index to class value.
        """

        mask_files = get_image_files(self.masks_dir)
        if is_partial and len(mask_files) > 10:
            mask_files = random.sample(mask_files, 10) 

        # Get unique classes from each file name
        unique_classes = set()
        for file_path in mask_files:
            mask = rxr.open_rasterio(file_path)
            unique_classes.update(np.unique(mask.data))

        # Convert into a dictionary mapping index to class value
        pixel_to_class = {i: class_value for i, class_value in enumerate(unique_classes)}
        return pixel_to_class


    def _get_mask(self, image_path, pixel_to_class):
        """
        Get mask from an image path and adjust the pixels based on pixel-to-class mapping.

        Args:
            image_path (str): Path to the image.
            pixel_to_class (dict): Mapping of pixel values to classes.

        Returns:
            PILMask: The mask created from the image.
        """
        # new structure: 
        mask_path = str(image_path).replace("images", "masks")
        mask = rxr.open_rasterio(mask_path)
        
        for i, val in enumerate(pixel_to_class):
            mask = xr.where(mask == pixel_to_class[i], val, mask)
        mask = mask.values.reshape((mask.shape[1], mask.shape[2]))
        
        return PILMask.create(mask)


    def _get_batch_size(self, tile_size, backbone):
        """
        Automatically set batch size as a function of tile size and backbone used.

        Args:
            tile_size (int): Size of the tile.
            backbone (str): Name of the backbone model.

        Returns:
            int: Batch size.
        """
        if tile_size == 512:
            batch_size_dict = {'resnet152': 2, 'resnet101': 2, 'resnet50': 2, 
                            # Change batch size for used backbone if you run into CUDA out of memory errors
                            'resnet34': 11, 'resnet18': 8, 'vgg16_bn': 2,
                            'hrnet_w18': 32, 'hrnet_w30': 32, 'hrnet_w32': 32,
                            'hrnet_w48': 18}
        elif tile_size == 256:
            batch_size_dict = {'resnet152': 2, 'resnet101': 2, 'resnet50': 2,
                            'resnet34': 11, 'resnet18': 10, 'hrnet_w18': 64}
        return batch_size_dict.get(backbone, 4)


    def _callbacks(self, timestamp):
        """
        Log results in CSV, show progress in graph.

        Returns:
            list: List of callbacks.
        """
        log_dir = create_if_not_exists(self.model_dir / f'{self.train_params["architecture"]}_{timestamp}_logs')
        csv_path = str(log_dir / 'train_metrics.csv')
        tb_dir = str(log_dir / 'tb_logs/')
        cbs = [CSVLogger(fname=csv_path),
            ShowGraphCallback(),
            EarlyStoppingCallback(patience=10)]
        if self.train_params["architecture"].lower() == "u-net":
            cbs.append(TensorBoardCallback(log_dir=tb_dir))
        return cbs


    def get_y(self, x):
        """
        Get the mask for a given image. Label function for SegmentationDataLoaders.

        Args:
            x (str): Path to the image.

        Returns:
            PILMask: The mask for the image.
        """
        return self._get_mask(x, self.p2c_map)

    def check_dataset_balance(self):
        """
        Check balance of the dataset.

        Returns:
            Tensor: Percentages of pixels that belong to buildings.
        """
    
        fnames = get_image_files(self.images_dir) 
        label_func = partial(self.get_y)

        # Create dataloader to check building pixels
        dls = SegmentationDataLoaders.from_label_func(self.images_dir, fnames, label_func=label_func, bs=2, codes=self.params["codes"], seed=self.params["seed"])

        targs = torch.zeros((0, self.params["tile_size"], self.params["tile_size"]))
        # issue here with // execution
        for _, masks in dls[0]:
            targs = torch.cat((targs, masks.cpu()), dim=0)

        total_pixels = targs.shape[1] ** 2
        if total_pixels == 0:
            return 0
        
        percentages = torch.count_nonzero(targs, dim=(1, 2)) / total_pixels
        plt.hist(percentages, bins=20)
        plt.ylabel('Number of tiles')
        plt.xlabel('Ratio of pixels that are of class `building`')
        plt.gca().spines['top'].set_color('none')
        plt.gca().spines['right'].set_color('none')
        plt.show()
        print(f'Mean Percentage of Pixels Belonging to Buildings: {round(percentages.mean().item(), 3)}')
        return percentages


    def run(self):
        """
        Train a model based on the given parameters. It supports both HRNet and U-Net architectures.
        It applies image augmentations, creates dataloaders, sets up the model, and finally trains it.

        Returns:
            Learner: Trained model.
            Dataloaders: Dataloaders used for segmentation.
        """


        tfms = [*aug_transforms(mult=1.0, do_flip=True, flip_vert=True, max_rotate=40.0, min_zoom=1.0, max_zoom=1.4, max_warp=0.4),
                Normalize.from_stats(*imagenet_stats),
                Brightness(max_lighting=0.5),
                Contrast(max_lighting=0.5),
                Hue(max_hue=0.2),
                Saturation(max_lighting=0.5)]
        
        image_files = get_image_files(self.images_dir)
        self.train_params["batch_size"] = min(self.train_params["batch_size"], len(image_files))

        label_func = partial(self.get_y)

        dls = SegmentationDataLoaders.from_label_func(self.images_dir, image_files, label_func=label_func, bs=self.train_params["batch_size"], codes=self.params["codes"], seed=self.params["seed"],
                                                    batch_tfms=tfms,
                                                    valid_pct=self.params["test_size"], num_workers=0)

        if self.train_params["architecture"].lower() == 'hrnet':
            self.learner = get_segmentation_learner(dls, number_classes=2, segmentation_type="Semantic Segmentation",
                                            architecture_name="hrnet",
                                            backbone_name=self.train_params["backbone"], model_dir=self.model_dir, metrics=[Dice()]).to_fp16()
        elif self.train_params["architecture"].lower() == 'u-net':
            loss_functions = {'Dual_Focal_loss': DualFocalLoss(), 'CombinedLoss': CombinedLoss(),
                            'DiceLoss': DiceLoss(), 'FocalLoss': FocalLoss(), None: None, 'CrossCombinedLoss': CrossCombinedLoss()}
            backbones = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50,
                        'resnet101': resnet101, 'vgg16_bn': vgg16_bn}
            self.learner = unet_learner(dls, backbones.get(self.train_params["backbone"]), n_out=2, loss_func=loss_functions.get(self.train_params["loss_function"]), metrics=[Dice(), JaccardCoeff()])

        save_timestamp = timestamp()
        self.learner.fit_one_cycle(self.train_params["epochs"], cbs=self._callbacks(save_timestamp))
        model_path = self._save(save_timestamp)
        return model_path


    def _save(self, timestamp):
    
        self.model_name = f"{self.train_params['architecture']}_{timestamp}.pkl"
        model_path = str(self.model_dir / self.model_name)
        print(model_path)
        self.learner.export(model_path)

        combined_params = {**self.params, **self.train_params}
        json_path = str((self.model_dir / self.model_name).with_suffix('.json'))
        print(json_path)
        with open(json_path, 'w') as f:
            json.dump(combined_params, f)

        return model_path


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    config = load_config("test_config.yaml")
    
    trainer = Trainer(config)
    trainer.run()