
import os
from itertools import islice
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from segmentation.infer import MapGenerator
from utils.helpers import create_if_not_exists, load_config
from segmentation.train import Trainer
from fastai.vision.all import load_learner


class Evaluator:
    def __init__(self, config):

        try:
            path = Path(config["root_dir"])
        
            self.images_dir = path / config["dirs"]["test"] / "images"
            self.masks_dir = path / config["dirs"]["test"] / "masks"
            self.shp_dir = create_if_not_exists(path / config["dirs"]["shapefiles"])
            self.predict_dir = create_if_not_exists(path / config["dirs"]["predictions"])
            self.output_dir = create_if_not_exists(path / config["dirs"]["eval"])
            self.config = config

            self.generate_preds = True

            infer_args = config["test"]
            model_path = path / config["dirs"]["models"] / infer_args["model_name"]
            if not model_path.exists():
                raise ValueError(f"Couldn't find model under {model_path}.")
            self.model = load_learner(model_path)
        except KeyError as e:
            raise KeyError(f"Config must have a value for {e}.")

    def overlay_shapefiles_on_images(self, n_images, show=False):
        for shapefile in islice(self.shp_dir.iterdir(), n_images):
            if shapefile.name.endswith('.shp'):
                # Construct the corresponding image file path
                image_file = shapefile.name.replace('_predicted.shp', '.tif') # Assuming the image extension is .jpg
                image_path = self.images_dir / image_file

                if image_path.exists():
                    # Read the shapefile
                    gdf = gpd.read_file(shapefile)

                    # Open the image
                    with Image.open(image_path) as img:
                        draw = ImageDraw.Draw(img)

                        # Overlay each geometry in the shapefile
                        for geometry in gdf.geometry:
                            if geometry.geom_type == 'Polygon':
                                # Convert the polygon coordinates to image pixel coordinates
                                # This step needs customization based on the coordinate system of your shapefiles and images
                                polygon = [(x, y) for x, y in zip(geometry.exterior.coords.xy[0], geometry.exterior.coords.xy[1])]
                                draw.polygon(polygon, outline="red")

                        # Display the image
                        if show == True:
                            plt.imshow(img)
                            plt.title(image_file)
                            plt.show()
                        else:
                            output_path = self.output_dir / f"eval_pred_{image_file}"
                            img.save(output_path)
                            print(f"Image file {output_path.name} written to `{output_path.parent.name}`.")

                else:
                    print(f"Image file {image_file} not found.")


    def compute_metrics(self):
        sum_intersect = 0
        sum_total_pred = 0
        sum_total_truth = 0
        sum_union = 0
        sum_xor = 0

        num_files = 0

        for groundtruth_path in self.masks_dir.iterdir():
            pred_mask_path = self.predict_dir / (groundtruth_path.stem +'_inference.tif')

            if pred_mask_path.exists():
                # Load ground truth mask

                gt_mask = np.array(Image.open(groundtruth_path).convert('L')) / 255  # Convert to binary (0 and 1)
                gt_mask = gt_mask.astype(int)

                # Load predicted mask
                with rasterio.open(pred_mask_path) as file:
                    pred_mask = file.read(1)
                    pred_mask = (pred_mask > 0.5).astype(int)


                # Accumulate values for metrics
                intersect = np.sum(pred_mask * gt_mask)
                sum_intersect += intersect
                sum_total_pred += np.sum(pred_mask)
                sum_total_truth += np.sum(gt_mask)
                sum_union += np.sum(pred_mask) + np.sum(gt_mask) - intersect
                sum_xor += np.sum(pred_mask == gt_mask)

                num_files += 1
            else:
                print(f"Predicted mask for {groundtruth_path} not found.")

        if num_files > 0:
            # Compute aggregated metrics
            precision = sum_intersect / sum_total_pred
            recall = sum_intersect / sum_total_truth
            accuracy = sum_xor / (sum_union + sum_xor - sum_intersect)
            dice = 2 * sum_intersect / (sum_total_pred + sum_total_truth)
            iou = sum_intersect / sum_union

            # Create a DataFrame with the aggregated metrics
            metrics = {
                "Precision": [round(precision, 3)],
                "Recall": [round(recall, 3)],
                "Accuracy": [round(accuracy, 3)],
                "Dice": [round(dice, 3)],
                "IoU": [round(iou, 3)]
            }
            return pd.DataFrame(metrics)
        else:
            return pd.DataFrame()

    def evaluate(self, n_images=10):
        if self.generate_preds:
            map_gen = MapGenerator(self.config)
            map_gen.create_tile_inferences()
        self.overlay_shapefiles_on_images(n_images)
        metrics = self.compute_metrics()
        output_file_path = self.output_dir / 'metrics.csv'
        metrics.to_csv(output_file_path)

if __name__ == "__main__":
    config = load_config("base_config.yaml")
    evaluator = Evaluator(config)
    metrics_df = evaluator.evaluate()