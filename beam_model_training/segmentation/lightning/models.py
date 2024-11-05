import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import torchmetrics

class BuildingSegmentationModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])
        
        # Load training parameters from config
        self.batch_size = config.get('batch_size', 32)
        self.train_params = config.get('train', {})
        self.learning_rate = self.train_params.get('learning_rate', 1e-3)
        self.max_epochs = self.train_params.get('epochs', 100)
        
        # Initialize metrics with num_classes=2 for binary segmentation
        self.train_dice = torchmetrics.Dice(num_classes=2, average='micro')
        self.val_dice = torchmetrics.Dice(num_classes=2, average='micro')
        
        self.train_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=2)
        self.val_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=2)
        
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=2, average='micro')
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=2, average='micro')
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=2, average='micro')
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=2, average='micro')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for the model.
        """
        x, y = batch['image'], batch['mask']
        y_hat = self(x)
        
        # Cross entropy loss expects class indices as targets
        loss = F.cross_entropy(y_hat, y)
        
        # Calculate metrics
        preds = torch.argmax(y_hat, dim=1)
        
        # Keep predictions and targets as integers
        dice_score = self.train_dice(preds, y)
        iou_score = self.train_iou(preds, y)
        precision = self.train_precision(preds, y)
        recall = self.train_recall(preds, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log('train_dice', dice_score, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log('train_iou', iou_score, on_step=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log('train_precision', precision, on_step=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step for the model.
        """
        x, y = batch['image'], batch['mask']
        y_hat = self(x)
        
        val_loss = F.cross_entropy(y_hat, y)
        
        # Calculate metrics
        preds = torch.argmax(y_hat, dim=1)
        
        # Keep as integer tensors
        dice_score = self.val_dice(preds, y)
        iou_score = self.val_iou(preds, y)
        precision = self.val_precision(preds, y)
        recall = self.val_recall(preds, y)
        
        # Log metrics
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log('val_dice', dice_score, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)  
        self.log('val_iou', iou_score, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log('val_precision', precision, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log('val_recall', recall, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.max_epochs,
            steps_per_epoch=self.trainer.estimated_stepping_batches,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def get_progress_bar_dict(self) -> Dict[str, Any]:
        """Customize items to display in progress bar."""
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items