import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Optional

def get_training_augmentations(
    tile_size: int = 256,
    rotate_limit: int = 45,
    scale_limit: float = 0.2,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2
) -> A.Compose:
    """Get the default training augmentation pipeline using albumentations."""
    return A.Compose([
        # Spatial augmentations
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, 
            scale_limit=scale_limit, 
            rotate_limit=rotate_limit, 
            interpolation=1, 
            border_mode=0,
            value=0,
            mask_value=0,
            p=0.5
        ),
        
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
        ], p=0.3),
        
        # Weather/Atmospheric effects
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.5),
        ], p=0.2),
        
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