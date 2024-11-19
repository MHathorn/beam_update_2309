import argparse
import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import yaml
import rioxarray as rxr
import xarray as xr

from segmentation.lightning.data import BuildingSegmentationDataset
from segmentation.lightning.inference import InferenceModule, InferenceVisualizer
from segmentation.lightning.transforms import get_validation_augmentations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: Path) -> dict:
    """Load configuration from yaml file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def run_inference(
    project_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Optional[Path] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """
    Run inference on test dataset.
    
    Args:
        project_dir: Root project directory
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save predictions (defaults to project_dir/predictions)
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        device: Device to run inference on
    """
    # Load config
    config = load_config(config_path)
    
    # Setup output directory
    output_dir = output_dir or project_dir / 'predictions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup test dataset
    test_dir = project_dir / 'tiles/test'
    test_dataset = BuildingSegmentationDataset(
        images_dir=test_dir / 'images',
        masks_dir=test_dir / 'masks',  # Optional for inference
        transform=get_validation_augmentations(),
        is_training=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Important: keep order for geospatial reference
        pin_memory=True
    )
    
    # Initialize inference module
    model = InferenceModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        strict=True
    )
    
    # Setup trainer for inference
    trainer = Trainer(
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
        precision='16-mixed',  # Use half precision for faster inference
        logger=False,  # Disable logging for inference
    )
    
    # Run predictions
    logging.info("Starting inference...")
    predictions = trainer.predict(model, test_loader)

        # Visualize predictions
    # Create visualization directory
    output_viz_dir = output_dir / 'visualizations'
    output_viz_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer once
    visualizer = InferenceVisualizer(config)

    # Process each batch
    for batch_idx, batch_pred in enumerate(predictions):
        visualizer.visualize_batch_predictions(
            batch_predictions=batch_pred,
            output_dir=output_viz_dir / f'batch_{batch_idx}',  # Organize by batch
            max_samples=4
        )

    logging.info(f"Predictions type: {type(predictions)}")
    logging.info(f"Number of batches: {len(predictions)}")
    if predictions:
        logging.info(f"First batch keys: {predictions[0].keys()}")
        logging.info(f"First batch predictions shape: {predictions[0]['predictions'].shape}")
    
    # Process and save predictions
    save_predictions(predictions, output_dir, config)
    
def save_predictions(predictions, output_dir: Path, config: dict) -> None:
    """Save predictions with visualizations."""

    logging.info(f"Output directory: {output_dir}")
    
    # Initialize directories
    pred_dir = output_dir / 'predictions'
    prob_dir = output_dir / 'probabilities' if config.get('inference', {}).get('save_probabilities', False) else None
    conf_dir = output_dir / 'confidence'
    viz_dir = output_dir / 'visualizations' if config.get('visualization', {}).get('enabled', False) else None
    
    for dir_path in [d for d in [pred_dir, prob_dir, conf_dir, viz_dir] if d is not None]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")
    
    total_samples = sum(len(batch['predictions']) for batch in predictions)
    logging.info(f"Processing {total_samples} samples")
    
    for batch_idx, batch in enumerate(predictions):
        logging.info(f"Processing batch {batch_idx}")
        for idx in range(len(batch['predictions'])):
            pred = batch['predictions'][idx].cpu().numpy()
            probs = batch['probabilities'][idx].cpu().numpy()
            conf = batch['confidences'][idx].cpu().numpy()
            image_path = batch['image_path'][idx]
            stem = Path(image_path).stem
            
            logging.info(f"Processing sample {stem}")
            
            try:
                with rxr.open_rasterio(image_path) as src:
                    # Save prediction
                    pred_path = pred_dir / f"{stem}_pred.tif"
                    logging.info(f"Saving prediction to {pred_path}")
                    
                    pred_da = xr.DataArray(
                        pred,
                        dims=('y', 'x'),
                        coords={'y': src.y, 'x': src.x}
                    )
                    pred_da.rio.write_crs(src.rio.crs, inplace=True)
                    pred_da.rio.write_transform(src.rio.transform(), inplace=True)
                    pred_da.rio.to_raster(str(pred_path))  # Convert path to string
                    
                    logging.info(f"Successfully saved prediction to {pred_path}")
                    
                    # Save probabilities if enabled
                    if prob_dir:
                        prob_path = prob_dir / f"{stem}_probs.tif"
                        prob_da = xr.DataArray(
                            probs,
                            dims=('class', 'y', 'x'),
                            coords={
                                'class': range(probs.shape[0]),
                                'y': src.y,
                                'x': src.x
                            }
                        )
                        prob_da.rio.write_crs(src.rio.crs, inplace=True)
                        prob_da.rio.write_transform(src.rio.transform(), inplace=True)
                        prob_da.rio.to_raster(str(prob_path))
                        
                    # Save confidence map
                    conf_path = conf_dir / f"{stem}_conf.tif"
                    conf_da = xr.DataArray(
                        conf,
                        dims=('y', 'x'),
                        coords={'y': src.y, 'x': src.x}
                    )
                    conf_da.rio.write_crs(src.rio.crs, inplace=True)
                    conf_da.rio.write_transform(src.rio.transform(), inplace=True)
                    conf_da.rio.to_raster(str(conf_path))
                    
            except Exception as e:
                logging.error(f"Error processing {stem}: {e}")
                raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run building segmentation inference")
    parser.add_argument("--project_dir", type=Path, required=True, help="Project directory")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=Path, help="Output directory (optional)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help="Device to run inference on")
    
    args = parser.parse_args()
    
    run_inference(
        project_dir=args.project_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )