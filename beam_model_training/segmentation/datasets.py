import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import torchgeo.transforms as transforms
from torchvision import transforms as T

from utils.base_class import BaseClass
from utils.helpers import seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class BuildingSegmentationDataset(Dataset):
    """
    Dataset for building segmentation that handles both binary and edge-aware segmentation.
    
    Attributes:
        data_dir (Path): Directory containing image and mask data
        image_dir (Path): Directory containing image files
        mask_dir (Path): Directory containing mask files
        transform (Optional[object]): Transform to apply to images and masks
        has_edge (bool): Whether to use edge-aware segmentation
        image_files (List[Path]): List of image file paths
        mask_files (List[Path]): List of corresponding mask file paths
    """
    
    def __init__(
        self,
        data_dir: Path,
        transform: Optional[object] = None,
        has_edge: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Base directory containing 'images' and 'masks' subdirectories
            transform: Optional transforms to apply to both image and mask
            has_edge: Whether to use edge-aware segmentation
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.mask_dir = self.data_dir / 'masks'
        self.transform = transform
        self.has_edge = has_edge
        
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise ValueError(f"Both {self.image_dir} and {self.mask_dir} must exist")
        
        # Get all image files and their corresponding masks
        self.image_files = sorted(list(self.image_dir.glob('*.TIF')))
        self.mask_files = sorted(list(self.mask_dir.glob('*.TIF')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No TIF files found in {self.image_dir}")
        
        # Verify matching files
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(
                f"Number of images ({len(self.image_files)}) and "
                f"masks ({len(self.mask_files)}) don't match"
            )
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Dictionary containing:
                - 'image': Normalized image tensor (C×H×W)
                - 'mask': Binary or multi-class mask tensor (H×W)
                - 'edge': Edge mask tensor if has_edge=True (H×W)
        """
        # Load image with rasterio
        with rasterio.open(self.image_files[idx]) as src:
            image = src.read()  # CxHxW format
            
        # Load mask
        with rasterio.open(self.mask_files[idx]) as src:
            mask = src.read()  # CxHxW format
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        # Normalize image to [0, 1] range
        image = image / 255.0
        
        if self.has_edge:
            # Assume first channel is building interior, second is edge
            building_mask = mask[0]
            edge_mask = mask[1]
            
            # Create final mask where:
            # 0: background
            # 1: building interior
            # 2: building edge
            combined_mask = torch.zeros_like(building_mask)
            combined_mask[building_mask > 0] = 1
            combined_mask[edge_mask > 0] = 2
            mask = combined_mask
        else:
            # For binary segmentation, convert to binary mask
            mask = (mask[0] > 0).long()
        
        # Apply transforms if specified
        if self.transform:
            transformed = self.transform({
                'image': image,
                'mask': mask
            })
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask
        }


class BuildingSegmentationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for building segmentation.
    
    Handles data loading, transforms, and train/val/test splits.
    """
    
    def __init__(
        self,
        train_dir: Path,
        val_dir: Path,
        batch_size: int = 8,
        num_workers: int = 4,
        has_edge: bool = False,
        augmentation_params: Optional[Dict] = None
    ):
        """
        Initialize the DataModule.
        
        Args:
            train_dir: Directory containing training data
            val_dir: Directory containing validation data
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            has_edge: Whether to use edge-aware segmentation
            augmentation_params: Dictionary of augmentation parameters
        """
        super().__init__()
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.has_edge = has_edge
        self.augmentation_params = augmentation_params or {}
        
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training and validation.
        
        Args:
            stage: Optional stage parameter (fit/test)
        """
        # Define transforms
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
        if stage == 'fit' or stage is None:
            self.train_dataset = BuildingSegmentationDataset(
                self.train_dir,
                transform=self.train_transform,
                has_edge=self.has_edge
            )
            
            self.val_dataset = BuildingSegmentationDataset(
                self.val_dir,
                transform=self.val_transform,
                has_edge=self.has_edge
            )
            
            logging.info(f"Training dataset size: {len(self.train_dataset)}")
            logging.info(f"Validation dataset size: {len(self.val_dataset)}")
    
    def _get_train_transforms(self):
        """
        Get training transforms including augmentations.
        """
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=self.augmentation_params.get('rotate', 45)
            ),
            transforms.ColorJitter(
                brightness=self.augmentation_params.get('brightness', 0.2),
                contrast=self.augmentation_params.get('contrast', 0.2),
                saturation=self.augmentation_params.get('saturation', 0.2)
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_val_transforms(self):
        """
        Get validation transforms (normalization only).
        """
        return transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class DatasetManager(BaseClass):
    """
    Manager class for dataset creation and verification.
    Handles configuration loading and dataset preparation.
    """
    
    def __init__(self, project_dir: str, config_name: Optional[str] = None):
        """
        Initialize the DatasetManager.
        
        Args:
            project_dir: Path to the project directory
            config_name: Optional name of config file
        """
        super().__init__(project_dir, config_name)
        
        # Load configurations
        self._load_params()
        seed(self.params["seed"])
        
        # Setup directories
        data_dirs = ["train_images", "train_masks", "val_images", "val_masks"]
        super().load_dir_structure(read_dirs=data_dirs)
        
    def _load_params(self):
        """Load and validate parameters from config file."""
        # Required parameters
        required_keys = ["seed", "codes", "val_size"]
        self.params = {k: self.config.get(k) for k in required_keys}
        
        # Training parameters
        train_keys = [
            "batch_size",
            "num_workers",
            "augmentation"
        ]
        self.train_params = {k: self.config.get("train", {}).get(k) for k in train_keys}
        
        # Set defaults
        self.train_params.setdefault("batch_size", 8)
        self.train_params.setdefault("num_workers", 4)
        self.train_params.setdefault("augmentation", {})
        
        # Validate required parameters
        for k in required_keys:
            if not self.params.get(k):
                raise ValueError(f"Please provide a configuration value for `{k}`")
                
    def create_data_module(self) -> BuildingSegmentationDataModule:
        """
        Create and return a BuildingSegmentationDataModule instance.
        
        Returns:
            Configured BuildingSegmentationDataModule
        """
        has_edge = "edge" in self.params["codes"]
        
        return BuildingSegmentationDataModule(
            train_dir=self.train_images_dir.parent,
            val_dir=self.val_images_dir.parent,
            batch_size=self.train_params["batch_size"],
            num_workers=self.train_params["num_workers"],
            has_edge=has_edge,
            augmentation_params=self.train_params["augmentation"]
        )
        
    def verify_datasets(self):
        """
        Verify dataset integrity and print summary statistics.
        """
        logging.info("Verifying dataset integrity...")
        
        data_module = self.create_data_module()
        data_module.setup()
        
        train_size = len(data_module.train_dataset)
        val_size = len(data_module.val_dataset)
        
        logging.info(f"Found {train_size} training samples")
        logging.info(f"Found {val_size} validation samples")
        
        # Sample a few images to verify loading
        try:
            sample_batch = next(iter(data_module.train_dataloader()))
            logging.info(
                f"Successfully loaded batch with shapes: "
                f"images={sample_batch['image'].shape}, "
                f"masks={sample_batch['mask'].shape}"
            )
        except Exception as e:
            logging.error(f"Error loading sample batch: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and verify datasets for building segmentation."
    )
    parser.add_argument(
        "-d",
        "--project_dir",
        type=str,
        help="The project directory.",
        required=True
    )
    parser.add_argument(
        "-c",
        "--config_name",
        type=str,
        help="The configuration file name. If missing, the constructor will look for a single file in the project directory."
    )
    
    args = parser.parse_args()
    
    try:
        dataset_manager = DatasetManager(args.project_dir, args.config_name)
        dataset_manager.verify_datasets()
    except Exception as e:
        logging.error(f"Error in dataset preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main()