import argparse
import logging
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import yaml
from tqdm import tqdm

from segmentation.lightning.data import BuildingSegmentationDataset
from segmentation.lightning.transforms import get_validation_augmentations
from segmentation.lightning.map_generator import MapGeneratorModule
from segmentation.lightning.inference import InferenceVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_boundaries(boundaries_path: Optional[Path], crs: Optional[str] = None) -> Optional[gpd.GeoDataFrame]:
    """Load and process boundary shapefile."""
    if not boundaries_path:
        return None
        
    boundaries = gpd.read_file(boundaries_path)
    if crs and boundaries.crs != crs:
        boundaries = boundaries.to_crs(crs)
    return boundaries

def run_inference(
    project_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    boundaries_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """
    Run enhanced building segmentation inference with geospatial processing.
    """
    # Load config
    config = yaml.safe_load(config_path.read_text())
    
    # Setup output directories
    output_dir = output_dir or project_dir / 'predictions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output subdirectories
    pred_dir = output_dir / 'predictions'
    shape_dir = output_dir / 'shapefiles'
    conf_dir = output_dir / 'confidence'
    viz_dir = output_dir / 'visualizations'
    
    for dir_path in [pred_dir, shape_dir, conf_dir, viz_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load boundaries if provided
    boundaries_gdf = load_boundaries(
        boundaries_path,
        config.get('geospatial', {}).get('crs')
    )
    
    # Setup test dataset
    test_dir = project_dir / 'tiles/test'
    test_dataset = BuildingSegmentationDataset(
        images_dir=test_dir / 'images',
        masks_dir=test_dir / 'masks',
        transform=get_validation_augmentations(),
        is_training=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    # Initialize map generator module
    model = MapGeneratorModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        strict=True
    )
    
    # Setup trainer
    trainer = Trainer(
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
        precision='16-mixed',
        logger=False
    )
    
    # Run predictions
    logging.info("Starting inference...")
    predictions = trainer.predict(model, test_loader)
    
    # Initialize visualizer
    visualizer = InferenceVisualizer(config)
    
    # Process predictions
    logging.info("Processing predictions...")
    
    # Save individual tile predictions and visualizations
    for batch_idx, batch_pred in enumerate(predictions):
        # Save visualizations
        visualizer.visualize_batch_predictions(
            batch_predictions=batch_pred,
            output_dir=viz_dir / f'batch_{batch_idx}',
            max_samples=4
        )
        
        # Save processed predictions
        for proc_pred in batch_pred['processed']:
            # Save prediction masks
            proc_pred['mask'].rio.to_raster(
                pred_dir / f"{Path(proc_pred['image_path']).stem}_pred.tif"
            )
            
            # Save confidence maps
            proc_pred['confidence'].rio.to_raster(
                conf_dir / f"{Path(proc_pred['image_path']).stem}_conf.tif"
            )
            
            # Save individual shapefiles
            if proc_pred['polygons'] is not None and len(proc_pred['polygons']) > 0:
                proc_pred['polygons'].to_file(
                    shape_dir / f"{Path(proc_pred['image_path']).stem}_buildings.shp"
                )
    
    # Merge predictions if we have multiple tiles
    if len(predictions) > 1:
        logging.info("Merging predictions across tiles...")
        merged_gdf = model.merge_predictions(predictions, boundaries_gdf)
        
        # Save merged shapefile
        output_shapefile = shape_dir / "merged_buildings.shp"
        merged_gdf.to_file(output_shapefile)
        logging.info(f"Saved merged predictions to {output_shapefile}")
    
    logging.info("Inference completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run building segmentation inference")
    parser.add_argument("--project_dir", type=Path, required=True, help="Project directory")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--boundaries", type=Path, help="Path to boundaries shapefile (optional)")
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
        boundaries_path=args.boundaries,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )