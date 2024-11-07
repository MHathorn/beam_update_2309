import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Optional

def get_training_augmentations(
    tile_size: int = 256,
    rotate_limit: int = 45,
    scale_limit: float = 0.1,  # Reduced
    brightness_limit: float = 0.1,  # Reduced
    contrast_limit: float = 0.1  # Reduced
) -> A.Compose:
    return A.Compose([
        # Spatial augmentations (more conservative)
        A.RandomRotate90(p=0.3),
        A.Flip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            interpolation=1,
            border_mode=0,
            value=0,
            mask_value=0,
            p=0.3
        ),
        
        # Basic color augmentations only
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.3
        ),
        
        # Remove weather effects for now
        
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_validation_augmentations() -> A.Compose:
    """Get the default validation augmentation pipeline (normalization only)."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])