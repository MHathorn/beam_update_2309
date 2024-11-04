import argparse
import logging
from pathlib import Path
import yaml

import pytorch_lightning as pl
import torch
import torch.nn as nn

from segmentation.lightning.data import BuildingSegmentationDataModule
from segmentation.lightning.models import BuildingSegmentationModule

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class SimpleCNN(nn.Module):
    """Simple CNN for testing the Lightning setup"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.conv3(x)

def test_lightning_setup(project_dir: Path, config_path: Path):
    """
    Test the Lightning infrastructure setup.
    
    Args:
        project_dir: Path to the project directory
        config_path: Path to the configuration file
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded configuration from {config_path}")
    
    # Initialize data module
    data_module = BuildingSegmentationDataModule(
        data_dir=project_dir,
        config=config,
        batch_size=2,  # Small batch size for testing
        num_workers=0   # No multiprocessing for testing
    )
    
    logging.info("Initializing data module...")
    data_module.setup()
    
    # Test data loading
    try:
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        # Try to load a batch
        batch = next(iter(train_loader))
        logging.info(f"Successfully loaded a batch with keys: {batch.keys()}")
        logging.info(f"Image shape: {batch['image'].shape}")
        logging.info(f"Mask shape: {batch['mask'].shape}")
        
    except Exception as e:
        logging.error(f"Error during data loading: {str(e)}")
        raise
    
    # Initialize model
    num_classes = len(config.get('codes', ['background', 'building']))
    model = SimpleCNN(num_classes=num_classes)
    
    # Initialize Lightning module
    lightning_module = BuildingSegmentationModule(
        model=model,
        config=config
    )
    
    logging.info("Initializing trainer...")
    
    # Initialize trainer with minimal settings
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='auto',
        devices=1,
        max_steps=5,  # Only run a few steps for testing
        enable_progress_bar=True,
        logger=False  # Disable logging for testing
    )
    
    logging.info("Starting test training loop...")
    
    try:
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        logging.info("Successfully completed test training loop!")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Lightning infrastructure setup")
    parser.add_argument(
        "-d",
        "--project_dir",
        type=str,
        help="The project directory.",
        required=True
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="Path to the configuration file",
        required=True
    )
    
    args = parser.parse_args()
    
    test_lightning_setup(
        project_dir=Path(args.project_dir),
        config_path=Path(args.config_path)
    )