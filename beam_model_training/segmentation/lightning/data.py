from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import rioxarray as rxr

class BuildingSegmentationDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for building segmentation dataset.
    Maintains compatibility with existing directory structure and configuration.
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        config: Dict,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        Initialize the DataModule.
        
        Args:
            data_dir: Root directory containing the data
            config: Configuration dictionary
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Load parameters from config
        self.train_params = config.get('train', {})
        self.tile_size = config['tiling'].get('tile_size', 256)
        
        # Set up paths based on existing directory structure
        self.train_dir = self.data_dir / 'tiles/train'
        self.val_dir = self.data_dir / 'tiles/test'
        
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training and validation.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = BuildingSegmentationDataset(
                images_dir=self.train_dir / 'images',
                masks_dir=self.train_dir / 'masks',
                tile_size=self.tile_size,
                transform=None  # We'll add transforms later
            )
            
            self.val_dataset = BuildingSegmentationDataset(
                images_dir=self.val_dir / 'images',
                masks_dir=self.val_dir / 'masks',
                tile_size=self.tile_size,
                transform=None  # We'll add transforms later
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create the train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

class BuildingSegmentationDataset(Dataset):
    """
    Dataset for building segmentation that maintains compatibility with existing data structure.
    """
    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        tile_size: int = 256,
        transform = None
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing image tiles
            masks_dir: Directory containing mask tiles
            tile_size: Size of the tiles
            transform: Transforms to apply to images and masks
        """
        super().__init__()
        
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.tile_size = tile_size
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.tif')))
        if not self.image_files:
            self.image_files = sorted(list(self.images_dir.glob('*.TIF')))
            
        if not self.image_files:
            raise ValueError(f"No TIFF files found in {self.images_dir}")
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample to fetch
            
        Returns:
            Dictionary containing the image and mask tensors
        """
        img_path = self.image_files[index]
        mask_path = self.masks_dir / img_path.name
        
        # Load image and mask using rioxarray
        image = rxr.open_rasterio(img_path).values
        mask = rxr.open_rasterio(mask_path).values
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        
        # Normalize mask to binary (0, 1) and keep as integer
        mask = torch.from_numpy(mask[0])  # Take first band
        mask = (mask > 0).long()  # Convert 255 to 1, keep as integer
        
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        return {
            'image': image,
            'mask': mask,
            'image_path': str(img_path),
            'mask_path': str(mask_path)
        }