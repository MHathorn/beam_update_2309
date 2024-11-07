import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics

class BuildingSegmentationModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Load model parameters
        self.batch_size = self.config['train'].get('batch_size', 32)  # Get batch size from train config
        self.learning_rate = self.config['train'].get('learning_rate', 1e-3)
        self.codes = self.config.get('codes', ['background', 'building', 'edge'])
        self.num_classes = len(self.codes)
        
        # Create model using smp
        self.model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=self.num_classes
        )
        
        # Initialize metrics
        self.train_dice = torchmetrics.Dice(num_classes=self.num_classes, average='macro')
        self.val_dice = torchmetrics.Dice(num_classes=self.num_classes, average='macro')
        self.train_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=self.num_classes)
        self.val_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=self.num_classes)

        if 'class_weights' in config:
            self.class_weights = torch.tensor(config['class_weights']).float()
        else:
            self.class_weights = None    

        self.save_hyperparameters(config)    

    def _log_images(self, batch, preds, stage='train', num_images=4):
        """Log images with overlaid predictions"""
        # Take only the first few images
        images = batch['image'][:num_images]
        masks = batch['mask'][:num_images]
        predictions = preds[:num_images]
        
        # Convert predictions to categorical
        predictions = torch.argmax(predictions, dim=1)
        
        # Create a color-coded visualization
        def _create_mask_visualization(mask):
            # Create RGB mask with different colors for each class
            vis = torch.zeros((mask.shape[0], 3, mask.shape[1], mask.shape[2]), device=mask.device)
            vis[:, 0] = (mask == 1) * 1.0  # Red for buildings
            vis[:, 1] = (mask == 2) * 1.0  # Green for edges
            return vis
            
        pred_vis = _create_mask_visualization(predictions)
        true_vis = _create_mask_visualization(masks)
        
        # Log to TensorBoard
        for idx in range(num_images):
            self.logger.experiment.add_image(
                f'{stage}_sample_{idx}/image',
                images[idx],
                self.current_epoch
            )
            self.logger.experiment.add_image(
                f'{stage}_sample_{idx}/prediction',
                pred_vis[idx],
                self.current_epoch
            )
            self.logger.experiment.add_image(
                f'{stage}_sample_{idx}/ground_truth',
                true_vis[idx],
                self.current_epoch
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch['image'], batch['mask']
        y_hat = self(x)
        
        # Move class weights to correct device if they exist
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(y_hat.device)
        
        # Calculate loss
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        
        # Get predictions
        preds = torch.argmax(y_hat, dim=1)
        
        # Calculate metrics
        dice = getattr(self, f'{stage}_dice')(preds, y)
        iou = getattr(self, f'{stage}_iou')(preds, y)
        
        # Log everything
        self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_dice', dice, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_iou', iou, prog_bar=True, sync_dist=True)

        # Calculate per-class metrics
        for i, code in enumerate(self.codes):
            class_mask = (y == i)
            if class_mask.sum() > 0:  # Only calculate if class exists in batch
                class_pred = (preds == i)
                class_intersection = (class_pred & class_mask).sum()
                class_union = (class_pred | class_mask).sum()
                class_iou = class_intersection.float() / (class_union + 1e-8)
                self.log(f'{stage}_iou_{code}', class_iou, sync_dist=True)
        
        # Log learning rate
        if stage == 'train':
            self.log('learning_rate', self.optimizers().param_groups[0]['lr'], sync_dist=True)
        
        # Log images periodically
        if self.current_epoch % 5 == 0:  # Every 5 epochs
            self._log_images(batch, y_hat, stage)

        return loss    

    def on_train_epoch_end(self):
        """Log histograms of model parameters"""
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'val')

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer