import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union


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
        """Log images with properly colored overlaid predictions"""
        # Take only the first few images
        images = batch['image'][:num_images]
        masks = batch['mask'][:num_images]
        predictions = torch.argmax(preds[:num_images], dim=1)
        
        # Denormalize images (reverse ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        denormalized_images = images * std + mean

        # Clamp values to valid range [0, 1]
        denormalized_images = torch.clamp(denormalized_images, 0, 1)

        
        # Create color-coded visualization
        def _create_mask_visualization(mask):
            # Create RGB mask (B,3,H,W)
            vis = torch.zeros((mask.shape[0], 3, mask.shape[1], mask.shape[2]), device=mask.device)
            
            # Background = black (already zeros)
            # Buildings = red
            vis[:, 0] = (mask == 1).float()
            # Edges = green (if using 3 classes)
            if self.num_classes > 2:
                vis[:, 1] = (mask == 2).float()
                
            return vis
        
        # Create visualizations
        pred_vis = _create_mask_visualization(predictions)
        true_vis = _create_mask_visualization(masks)
        
        # Log to TensorBoard
        for idx in range(num_images):
            # Log original image
            self.logger.experiment.add_image(
                f'{stage}_sample_{idx}/image',
                images[idx],
                self.current_epoch
            )
            
            # Log prediction and ground truth side by side
            self.logger.experiment.add_image(
                f'{stage}_sample_{idx}/pred_vs_true',
                torch.cat([pred_vis[idx], true_vis[idx]], dim=2),  # Concatenate horizontally
                self.current_epoch
            )
            
            # Add histograms of predictions for this image
            if idx == 0:  # Just for first image
                self.logger.experiment.add_histogram(
                    f'{stage}_predictions_dist',
                    predictions[idx].float(),
                    self.current_epoch
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str, batch_idx: Union[int, str]) -> torch.Tensor:
        x, y = batch['image'], batch['mask']
        
        # Convert batch_idx to int if it's a string
        batch_idx_int = int(batch_idx)
        
        # Debug every few batches

        
        y_hat = self(x)
        

        
        weights = self.class_weights.to(y_hat.device) if self.class_weights is not None else None

        # Calculate loss
        loss = F.cross_entropy(y_hat, y, weight=weights)
        
        # Get predictions
        preds = torch.argmax(y_hat, dim=1)
        
        # Calculate metrics
        dice = getattr(self, f'{stage}_dice')(preds, y)
        iou = getattr(self, f'{stage}_iou')(preds, y)
        
        # Log everything
        self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_dice', dice, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_iou', iou, prog_bar=True, sync_dist=True)
        
        # Log images periodically
        if batch_idx_int == 0 and self.current_epoch % 5 == 0:
            self._log_images(batch, y_hat, stage)
        
        return loss

    def on_train_epoch_end(self):
        """Log histograms of model parameters"""
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'train', batch_idx)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'val', batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01  # Add weight decay
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['train']['scheduler']['max_lr'],
            epochs=self.config['train']['epochs'],
            steps_per_epoch=self.trainer.estimated_stepping_batches // self.config['train']['epochs'],
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }