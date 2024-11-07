import argparse
import logging
from pathlib import Path
import yaml

import torch
import pytorch_lightning as pl
from segmentation.lightning.data import BuildingSegmentationDataModule
from segmentation.lightning.models import BuildingSegmentationModule

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_model(project_dir: Path, config_path: Path):
    """
    Train the segmentation model using PyTorch Lightning.
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded configuration from {config_path}")
    
    # Initialize data module with batch size from config
    data_module = BuildingSegmentationDataModule(
        data_dir=project_dir,
        config=config,
        batch_size=config['train'].get('batch_size', 32),
        num_workers=4
    )
    
    logging.info("Initializing data module...")
    data_module.setup()
    
    # Initialize Lightning module
    lightning_module = BuildingSegmentationModule(config=config)

    steps_per_epoch = len(data_module.train_dataset) // (config['train']['batch_size'] * torch.cuda.device_count())
    log_every_n_steps = max(1, min(steps_per_epoch // 4, 50))  # Log 4 times per epoch or every 50 steps, whichever is smaller

    
    logging.info("Initializing trainer...")
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['train'].get('epochs', 100),
        accelerator='auto',
        devices='auto',
        strategy='ddp',
        precision='16-mixed',
        enable_progress_bar=True,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=str(project_dir / 'lightning_logs'),
            name='segmentation_training'
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=str(project_dir / 'checkpoints'),
                filename='{epoch}-{val_dice:.3f}',
                save_top_k=3,
                mode='max',
                monitor='val_dice'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.EarlyStopping(
                monitor='val_dice',
                mode='max',
                patience=10,
                min_delta=0.001
            )
        ]
    )
    
    logging.info("Starting training...")
    
    try:
        trainer.fit(
            model=lightning_module,
            datamodule=data_module
        )
        # Run final validation
        logging.info("Running final validation...")
        val_results = trainer.validate(
            model=lightning_module,
            datamodule=data_module
        )
        logging.info(f"Final validation results: {val_results}")
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model with Lightning")
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
    
    train_model(
        project_dir=Path(args.project_dir),
        config_path=Path(args.config_path)
    )