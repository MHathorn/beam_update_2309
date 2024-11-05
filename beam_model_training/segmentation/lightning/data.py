import os
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
import rioxarray as rxr

from .transforms import get_training_augmentations, get_validation_augmentations

class BuildingSegmentationDataset(Dataset):
    """
    Dataset for building segmentation using albumentations and torchgeo transforms.
    """
    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        tile_size: int = 256,
        transform: Optional[A.Compose] = None,
        is_training: bool = True
    ):
        super().__init__()
        
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.tile_size = tile_size
        self.is_training = is_training
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.tif')))
        if not self.image_files:
            self.image_files = sorted(list(self.images_dir.glob('*.TIF')))
            
        if not self.image_files:
            raise ValueError(f"No TIFF files found in {self.images_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_files[index]
        mask_path = self.masks_dir / img_path.name
        
        # Load image and mask using rioxarray
        image = rxr.open_rasterio(img_path).values
        mask = rxr.open_rasterio(mask_path).values
        
        # Convert to numpy arrays in the format expected by albumentations
        image = image.transpose(1, 2, 0)  # CHW -> HWC for albumentations
        mask = mask[0]  # Take first band

        image = image.astype('float32')
        
        # Ensure image values are in [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
            
        # Apply albumentations transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']  # Now a torch tensor
            mask = transformed['mask']  # Now a torch tensor
        
        # Convert mask to binary and long dtype
        mask = (mask > 0).long()
        
        return {
            'image': image,
            'mask': mask,
            'image_path': str(img_path),
            'mask_path': str(mask_path)
        }

class BuildingSegmentationDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for building segmentation dataset.
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        config: Dict,
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.config = config
        self.pin_memory = pin_memory
        self.val_split = val_split
        
        # Calculate workers and batch size for multi-GPU
        num_gpus = torch.cuda.device_count()
        
        # Calculate number of workers per GPU
        suggested_workers = os.cpu_count() // max(num_gpus, 1)
        self.num_workers = num_workers if num_workers is not None else min(15, suggested_workers or 1)
        
        # Ensure minimum batch size per GPU
        min_batch_size = 2  # Minimum batch size per GPU
        if num_gpus > 0:
            self.batch_size = max(batch_size // num_gpus, min_batch_size)
        else:
            self.batch_size = batch_size
            
        logging.info(f"Using batch size of {self.batch_size} per GPU with {self.num_workers} workers")
        
        # Set up paths based on existing directory structure
        self.train_dir = self.data_dir / 'tiles/train'
        self.val_dir = self.data_dir / 'tiles/test'
        
        # Load parameters from config
        self.train_params = config.get('train', {})
        self.tile_size = config['tiling'].get('tile_size', 256)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        if self.checkpoint_dir.exists() and len(list(self.checkpoint_dir.glob('*'))) > 0:
            logging.warning(f"Checkpoint directory {self.checkpoint_dir} exists and is not empty.")

        
        # Load parameters from config
        self.train_params = config.get('train', {})
        self.tile_size = config['tiling'].get('tile_size', 256)
        
        # Get augmentation parameters from config
        aug_config = config.get('augmentation', {})
        self.aug_params = {
            'rotate_limit': aug_config.get('rotate_limit', 45),
            'scale_limit': aug_config.get('scale_limit', 0.2),
            'brightness_limit': aug_config.get('brightness_limit', 0.2),
            'contrast_limit': aug_config.get('contrast_limit', 0.2),
        }
        
        # Set up paths based on existing directory structure
        self.train_dir = self.data_dir / 'tiles/train'
        self.val_dir = self.data_dir / 'tiles/test'  # Using test set as validation
        
        self.train_dataset: Optional[BuildingSegmentationDataset] = None
        self.val_dataset: Optional[BuildingSegmentationDataset] = None

       
    def prepare_data(self):
        """
        Perform any necessary data preparation.
        This method is called only on 1 GPU/TPU in distributed settings.
        """
        # Check if directories exist
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")
        
        if not self.val_dir.exists():
            logging.warning(
                f"Validation directory not found: {self.val_dir}. "
                f"Will use {self.val_split:.0%} of training data for validation."
            )
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training and validation.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'fit' or stage is None:
            # Create training augmentations with config parameters
            train_transforms = get_training_augmentations(
                tile_size=self.tile_size,
                rotate_limit=self.aug_params['rotate_limit'],
                scale_limit=self.aug_params['scale_limit'],
                brightness_limit=self.aug_params['brightness_limit'],
                contrast_limit=self.aug_params['contrast_limit']
            )
            
            # Create validation transforms
            val_transforms = get_validation_augmentations()
            
            # Initialize training dataset
            self.train_dataset = BuildingSegmentationDataset(
                images_dir=self.train_dir / 'images',
                masks_dir=self.train_dir / 'masks',
                tile_size=self.tile_size,
                transform=train_transforms,
                is_training=True
            )
            
            # Initialize validation dataset
            if self.val_dir.exists():
                # Use separate validation directory if it exists
                self.val_dataset = BuildingSegmentationDataset(
                    images_dir=self.val_dir / 'images',
                    masks_dir=self.val_dir / 'masks',
                    tile_size=self.tile_size,
                    transform=val_transforms,
                    is_training=False
                )
            else:
                # Split training data for validation
                total_size = len(self.train_dataset)
                val_size = int(total_size * self.val_split)
                train_size = total_size - val_size
                
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    self.train_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
                
                # Update transforms for validation split
                self.val_dataset.dataset.transform = val_transforms
                self.val_dataset.dataset.is_training = False
                
            logging.info(f"Setup complete. Training on {len(self.train_dataset)} samples, "
                        f"validating on {len(self.val_dataset)} samples.")
    
    def train_dataloader(self) -> DataLoader:
        """Create the train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # Drop incomplete batches for better batch norm statistics
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_normalization_stats(self):
        """Get the normalization statistics used in the transforms."""
        return {
            'mean': [0.485, 0.456, 0.406],  # ImageNet stats - consider calculating from your data
            'std': [0.229, 0.224, 0.225]
        }