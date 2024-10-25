import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.base_class import BaseClass
from utils.helpers import seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class BuildingSegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning module for building segmentation.
    Supports both binary and edge-aware segmentation with multiple architectures.
    """
    
    def __init__(
        self,
        architecture: str = "unet",
        backbone: str = "resnet18",
        learning_rate: float = 1e-4,
        has_edge: bool = False,
        **kwargs
    ):
        """
        Initialize the model.
        
        Args:
            architecture: Model architecture ('unet' or 'hrnet')
            backbone: Backbone model for encoder
            learning_rate: Initial learning rate
            has_edge: Whether to use edge-aware segmentation
            **kwargs: Additional arguments for model configuration
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.architecture = architecture.lower()
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.has_edge = has_edge
        
        # Number of output classes (2 for binary, 3 for edge-aware)
        self.num_classes = 3 if has_edge else 2
        
        # Initialize model architecture
        self.model = self._create_model()
        
        # Initialize loss functions
        self.building_criterion = nn.CrossEntropyLoss()
        if has_edge:
            # Optional: Add class weights if needed
            self.edge_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 2.0]))
    
    def _create_model(self) -> nn.Module:
        """
        Create the segmentation model based on specified architecture.
        
        Returns:
            nn.Module: Initialized model
        """
        if self.architecture == "unet":
            return smp.Unet(
                encoder_name=self.backbone,
                encoder_weights="imagenet",
                in_channels=3,
                classes=self.num_classes
            )
        elif self.architecture == "hrnet":
            return smp.HRNet(
                encoder_name=self.backbone,
                in_channels=3,
                classes=self.num_classes
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def _calculate_loss(
        self,
        outputs: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate the loss for the current batch.
        
        Args:
            outputs: Model outputs
            masks: Ground truth masks
            
        Returns:
            Tuple containing:
                - Total loss
                - Dictionary of separate losses
        """
        if self.has_edge:
            # Split loss for building interior and edges
            building_loss = self.building_criterion(outputs, masks)
            edge_loss = self.edge_criterion(outputs, masks)
            total_loss = building_loss + edge_loss
            
            return total_loss, {
                'building_loss': building_loss.item(),
                'edge_loss': edge_loss.item()
            }
        else:
            # Binary segmentation loss
            loss = self.building_criterion(outputs, masks)
            return loss, {'building_loss': loss.item()}
    
    def _calculate_metrics(
        self,
        outputs: torch.Tensor,
        masks: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate metrics for the current batch.
        
        Args:
            outputs: Model outputs
            masks: Ground truth masks
            
        Returns:
            Dictionary of metrics
        """
        # Convert outputs to predictions
        preds = outputs.argmax(dim=1)
        
        # Calculate IoU
        intersection = torch.logical_and(preds == 1, masks == 1).sum()
        union = torch.logical_or(preds == 1, masks == 1).sum()
        iou = (intersection / (union + 1e-8)).item()
        
        # Calculate Dice coefficient
        dice = (2 * intersection / (union + intersection + 1e-8)).item()
        
        metrics = {
            'iou': iou,
            'dice': dice
        }
        
        if self.has_edge:
            # Calculate edge metrics
            edge_intersection = torch.logical_and(preds == 2, masks == 2).sum()
            edge_union = torch.logical_or(preds == 2, masks == 2).sum()
            edge_iou = (edge_intersection / (edge_union + 1e-8)).item()
            metrics['edge_iou'] = edge_iou
        
        return metrics
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Execute training step."""
        images, masks = batch['image'], batch['mask']
        outputs = self(images)
        
        loss, separate_losses = self._calculate_loss(outputs, masks)
        metrics = self._calculate_metrics(outputs, masks)
        
        # Log all metrics
        self.log('train_loss', loss, prog_bar=True)
        for name, value in {**separate_losses, **metrics}.items():
            self.log(f'train_{name}', value, prog_bar=True)
        
        return loss
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Execute validation step."""
        images, masks = batch['image'], batch['mask']
        outputs = self(images)
        
        loss, separate_losses = self._calculate_loss(outputs, masks)
        metrics = self._calculate_metrics(outputs, masks)
        
        # Log all metrics
        self.log('val_loss', loss, prog_bar=True)
        for name, value in {**separate_losses, **metrics}.items():
            self.log(f'val_{name}', value, prog_bar=True)
        
        return {'val_loss': loss, **metrics}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


class ModelManager(BaseClass):
    """
    Manager class for model creation and verification.
    Handles configuration loading and model setup.
    """
    
    def __init__(self, project_dir: str, config_name: Optional[str] = None):
        super().__init__(project_dir, config_name)
        
        # Load configurations
        self._load_params()
        seed(self.params["seed"])
        
        # Setup directories
        model_dirs = ["models"]
        if self.train_params["finetune"]:
            model_dirs.append("base_model")
        super().load_dir_structure(read_dirs=model_dirs)
    
    def _load_params(self):
        """Load and validate parameters from config file."""
        # Required parameters
        required_keys = ["seed", "codes"]
        self.params = {k: self.config.get(k) for k in required_keys}
        
        # Training parameters
        train_keys = [
            "architecture",
            "backbone",
            "learning_rate",
            "finetune"
        ]
        self.train_params = {k: self.config.get("train", {}).get(k) for k in train_keys}
        
        # Set defaults
        self.train_params.setdefault("architecture", "unet")
        self.train_params.setdefault("backbone", "resnet18")
        self.train_params.setdefault("learning_rate", 1e-4)
        self.train_params.setdefault("finetune", False)
        
        # Validate required parameters
        for k in required_keys:
            if not self.params.get(k):
                raise ValueError(f"Please provide a configuration value for `{k}`")
    
    def create_model(self) -> BuildingSegmentationModel:
        """
        Create and return a BuildingSegmentationModel instance.
        
        Returns:
            Configured BuildingSegmentationModel
        """
        has_edge = "edge" in self.params["codes"]
        
        return BuildingSegmentationModel(
            architecture=self.train_params["architecture"],
            backbone=self.train_params["backbone"],
            learning_rate=self.train_params["learning_rate"],
            has_edge=has_edge
        )
    
    def verify_model(self):
        """Verify model creation and forward pass."""
        logging.info("Verifying model setup...")
        
        model = self.create_model()
        
        # Create dummy input
        batch_size = 2
        channels = 3
        height = width = 256
        dummy_input = torch.randn(batch_size, channels, height, width)
        
        try:
            # Test forward pass
            with torch.no_grad():
                output = model(dummy_input)
            
            logging.info(
                f"Model verification successful. "
                f"Output shape: {output.shape}"
            )
            
            num_params = sum(p.numel() for p in model.parameters())
            logging.info(f"Total number of parameters: {num_params:,}")
            
        except Exception as e:
            logging.error(f"Error in model verification: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Verify model architecture and configuration."
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
        model_manager = ModelManager(args.project_dir, args.config_name)
        model_manager.verify_model()
    except Exception as e:
        logging.error(f"Error in model setup: {str(e)}")
        raise

if __name__ == "__main__":
    main()