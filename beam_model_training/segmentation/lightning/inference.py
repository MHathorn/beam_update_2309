import logging
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import BuildingSegmentationModule
from .transforms import get_validation_augmentations

class InferenceModule(BuildingSegmentationModule):
    """Building segmentation inference module with test-time augmentation support."""
    
    def __init__(self, config: Dict, tta_transforms: Optional[List] = None):
        super().__init__(config)
        
        # Initialize TTA configuration
        self.tta_config = config.get('tta', {})
        self.tta_enabled = self.tta_config.get('enabled', False)
        self.tta_transforms = self._get_tta_transforms() if self.tta_enabled else []
        
        # Inference settings
        self.inference_config = config.get('inference', {})
        self.confidence_threshold = self.inference_config.get('confidence_threshold', 0.5)
        
        # Set model to eval mode
        self.eval()
        self.freeze()

    def _get_tta_transforms(self) -> List[A.Compose]:
        """Get list of albumentations TTA transforms"""
        transforms = []
        base_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Original transform
        transforms.append(base_transform)
        
        # Add configured transforms
        if self.tta_config.get('flip_horizontal', True):
            transforms.append(A.Compose([
                A.HorizontalFlip(p=1.0),
                *base_transform
            ]))
            
        if self.tta_config.get('flip_vertical', True):
            transforms.append(A.Compose([
                A.VerticalFlip(p=1.0),
                *base_transform
            ]))
            
        if self.tta_config.get('rotate_90', True):
            transforms.append(A.Compose([
                A.Rotate(limit=(90, 90), p=1.0),
                *base_transform
            ]))
            
        return transforms

    def _inverse_transform(self, x: torch.Tensor, transform_idx: int) -> torch.Tensor:
        """Apply inverse transform to predictions"""
        if transform_idx == 0:  # Original
            return x
        elif transform_idx == 1:  # Horizontal flip
            return torch.flip(x, dims=[-1])
        elif transform_idx == 2:  # Vertical flip
            return torch.flip(x, dims=[-2])
        elif transform_idx == 3:  # 90 degree rotation
            return torch.rot90(x, k=-1, dims=[-2, -1])
        return x

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        images = batch['image']
        all_logits = []
        tta_preds = []
        
        tta_config = self.config.get('tta', {})
        if not tta_config.get('enabled', False):
            return self._basic_prediction(images, batch)
            
        transforms_config = tta_config.get('transforms', {})
        transforms = []

        # Base prediction
        with torch.no_grad():
            logits = self.forward(images)
            all_logits.append(logits)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            tta_preds.append(preds)

        # Build transform list from config
        if transforms_config.get('flip_horizontal', False):
            transforms.append((
                lambda x: torch.flip(x, dims=[-1]),
                lambda x: torch.flip(x, dims=[-1])
            ))
        
        if transforms_config.get('flip_vertical', False):
            transforms.append((
                lambda x: torch.flip(x, dims=[-2]),
                lambda x: torch.flip(x, dims=[-2])
            ))
            
        if transforms_config.get('rotate_90', False):
            transforms.append((
                lambda x: torch.rot90(x, k=1, dims=[-2, -1]),
                lambda x: torch.rot90(x, k=-1, dims=[-2, -1])
            ))
            
        if transforms_config.get('rotate_180', False):
            transforms.append((
                lambda x: torch.rot90(x, k=2, dims=[-2, -1]),
                lambda x: torch.rot90(x, k=-2, dims=[-2, -1])
            ))
            
        if transforms_config.get('diagonal_flip', False):
            transforms.append((
                lambda x: x.transpose(-2, -1),
                lambda x: x.transpose(-2, -1)
            ))
            
        if transforms_config.get('scale_down', False): #not currently functional
            transforms.append((
                lambda x: F.interpolate(x, scale_factor=0.8, mode='bilinear', align_corners=True),
                lambda x: F.interpolate(x, size=images.shape[-2:], mode='bilinear', align_corners=True)
            ))
            
        if transforms_config.get('scale_up', False): #not currently functional
            transforms.append((
                lambda x: F.interpolate(x, scale_factor=1.2, mode='bilinear', align_corners=True),
                lambda x: F.interpolate(x, size=images.shape[-2:], mode='bilinear', align_corners=True)
            ))

        # Apply transforms
        for transform_fn, inverse_fn in transforms:
            transformed = transform_fn(images)
            with torch.no_grad():
                logits = self.forward(transformed)
                logits = inverse_fn(logits)
                all_logits.append(logits)
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                tta_preds.append(preds)

        # Stack and average predictions
        all_logits = torch.stack(all_logits)
        tta_preds = torch.stack(tta_preds)
        avg_logits = all_logits.mean(dim=0)
        probs = F.softmax(avg_logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]

        result = {
            'predictions': predictions,
            'probabilities': probs,
            'confidences': confidence,
            'image_path': batch['image_path'],
            'image': images,
            'tta_predictions': tta_preds
        }

        # Add uncertainty metrics if configured
        if tta_config.get('uncertainty', {}).get('compute_entropy', False):
            result['entropy'] = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
        
        if tta_config.get('uncertainty', {}).get('compute_variance', False):
            result['variance'] = torch.var(torch.stack([F.softmax(l, dim=1) for l in all_logits]), dim=0).mean(1)

        return result


class InferenceVisualizer:
    """Enhanced visualization tools for building segmentation inference."""
    
    def __init__(self, config: dict):
        self.config = config
        self.class_colors = config.get('visualization', {}).get('colors', {
            0: [0, 0, 0],       # Background (black)
            1: [255, 0, 0],     # Building (red)
            2: [0, 255, 0]      # Edge (green)
        })
        

    def visualize_batch_predictions(
        self,
        batch_predictions: Dict[str, torch.Tensor],
        output_dir: Path,
        max_samples: int = 4
    ) -> None:
        """Visualize predictions for a batch"""
        for idx in range(min(len(batch_predictions['predictions']), max_samples)):
            self.visualize_tta_sample(
                image=batch_predictions['image'][idx],
                tta_preds=batch_predictions['tta_predictions'][:, idx],
                final_pred=batch_predictions['predictions'][idx],
                confidence=batch_predictions['confidences'][idx],
                save_dir=output_dir,
                sample_idx=idx
            )

    def visualize_tta_sample(
        self,
        image: torch.Tensor,           # (C, H, W)
        tta_preds: torch.Tensor,       # (T, H, W)
        final_pred: torch.Tensor,      # (H, W)
        confidence: torch.Tensor,      # (H, W)
        save_dir: Path,
        sample_idx: int
    ) -> None:
        """Create comprehensive TTA visualization for a single sample"""
        sample_dir = save_dir / f"sample_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_denorm = (image * std + mean).cpu().numpy()
        image_denorm = np.transpose(image_denorm, (1, 2, 0))
        image_denorm = np.clip(image_denorm, 0, 1)
        
        # Create figure with subplots
        n_transforms = len(tta_preds)
        fig = plt.figure(figsize=(15, 10))
        
        # Original image
        ax1 = plt.subplot(2, n_transforms + 1, 1)
        ax1.imshow(image_denorm)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # TTA predictions
        for idx in range(n_transforms):
            ax = plt.subplot(2, n_transforms + 1, idx + 2)
            pred = tta_preds[idx].cpu().numpy()  # Already correct shape (H, W)
            colored_pred = self._colorize_prediction(pred)
            ax.imshow(colored_pred)
            ax.set_title(f'TTA {idx}')
            ax.axis('off')
        
        # Final prediction and confidence
        ax_final = plt.subplot(2, n_transforms + 1, n_transforms + 2)
        colored_final = self._colorize_prediction(final_pred.cpu().numpy())
        ax_final.imshow(colored_final)
        ax_final.set_title('Final Prediction')
        ax_final.axis('off')
        
        ax_conf = plt.subplot(2, n_transforms + 1, 2 * n_transforms + 2)
        conf_map = ax_conf.imshow(confidence.cpu().numpy(), cmap='magma')
        ax_conf.set_title('Confidence')
        plt.colorbar(conf_map, ax=ax_conf)
        ax_conf.axis('off')
        
        plt.tight_layout()
        plt.savefig(sample_dir / 'tta_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _colorize_prediction(self, pred: np.ndarray) -> np.ndarray:
        """Debug version of colorize prediction"""
        pred = pred.squeeze()
        rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
        
        print(f"Prediction shape before colorizing: {pred.shape}")
        print(f"Unique values in prediction: {np.unique(pred)}")
        
        for class_idx, color in self.class_colors.items():
            mask = pred == class_idx
            if mask.any():
                print(f"Found pixels for class {class_idx}")
            rgb[mask] = color
            
        return rgb



