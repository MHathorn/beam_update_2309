import argparse
import logging
import yaml
from pathlib import Path
import torch
from segmentation.lightning.data import BuildingSegmentationDataModule

def test_pipeline(project_dir: Path, config_path: Path):
    """
    Test the data pipeline using the validation script.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded configuration from {config_path}")
    
    # Initialize data module
    data_module = BuildingSegmentationDataModule(
        data_dir=project_dir,
        config=config,
        batch_size=4,  # Small batch size for testing
        num_workers=0  # Single worker for easier debugging
    )
    
    def validate_data_pipeline(datamodule, num_batches=5):
        """
        Validate the data pipeline by checking for common issues.
        """
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        def analyze_loader(loader, name):
            print(f"\n=== Analyzing {name} loader ===")
            image_ranges = []
            class_counts = {0: 0, 1: 0, 2: 0}
            
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                    
                images = batch['image']
                masks = batch['mask']
                paths = batch['image_path']  # Add paths for debugging
                
                # Check image statistics
                img_min = images.min().item()
                img_max = images.max().item()
                img_mean = images.mean().item()
                img_std = images.std().item()
                
                image_ranges.append((img_min, img_max))
                
                # Update class counts
                for c in range(3):
                    class_counts[c] += (masks == c).sum().item()
                
                print(f"\nBatch {i}:")
                print(f"Images from: {paths}")
                print(f"Image range: [{img_min:.3f}, {img_max:.3f}]")
                print(f"Image mean: {img_mean:.3f}, std: {img_std:.3f}")
                print(f"Image tensor shape: {images.shape}")
                print(f"Mask tensor shape: {masks.shape}")
                print(f"Unique mask values: {torch.unique(masks).tolist()}")
                
                # Check for NaN/Inf values
                if torch.isnan(images).any():
                    print("WARNING: Found NaN values in images!")
                    print("NaN locations:", torch.nonzero(torch.isnan(images)))
                
                if torch.isinf(images).any():
                    print("WARNING: Found Inf values in images!")
                    print("Inf locations:", torch.nonzero(torch.isinf(images)))
                
                # Save problematic batches for inspection
                if img_min < -5.0 or img_max > 5.0:
                    print(f"WARNING: Unusual image range detected in batch {i}!")
                    print("Affected files:", paths)
                
            total_pixels = sum(class_counts.values())
            print("\nClass distribution:")
            for c, count in class_counts.items():
                percentage = (count / total_pixels) * 100
                print(f"Class {c}: {percentage:.2f}%")
            
            print("\nImage range statistics across batches:")
            print(f"Min range: {min(r[0] for r in image_ranges):.3f}")
            print(f"Max range: {max(r[1] for r in image_ranges):.3f}")
                
        analyze_loader(train_loader, "Training")
        analyze_loader(val_loader, "Validation")

    # Run the validation
    try:
        logging.info("Starting pipeline validation...")
        validate_data_pipeline(data_module)
        logging.info("Pipeline validation completed successfully!")
    except Exception as e:
        logging.error(f"Error during pipeline validation: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data pipeline")
    parser.add_argument(
        "-d",
        "--project_dir",
        type=str,
        help="The project directory",
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
    
    test_pipeline(
        project_dir=Path(args.project_dir),
        config_path=Path(args.config_path)
    )