
import os
from datetime import datetime
from itertools import islice
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from fastai.vision.all import load_learner
from PIL import Image, ImageDraw
from segmentation.infer import MapGenerator
from segmentation.train import Trainer
from utils.helpers import (create_if_not_exists, crs_to_pixel_coords,
                           load_config)


class Evaluator:
    """
    A class used to evaluate the performance of a segmentation model.

    ...

    Methods
    -------
    overlay_shapefiles_on_images(n_images, show=False):
        Overlays shapefiles on corresponding images and saves or displays the result.
    compute_metrics():
        Computes precision, recall, accuracy, dice coefficient, and IoU for the model's predictions.
    evaluate(n_images=10):
        Generates predictions, overlays them on images, computes metrics, and saves the results.
    """

    def __init__(self, config):
        """
        Constructs all the necessary attributes for the Evaluator object.

        Parameters
        ----------
            config : dict
                Configuration parameters.
        """

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
            self.model_name = infer_args["model_name"]
            model_path = path / config["dirs"]["models"] / self.model_name
            if not model_path.exists():
                raise ValueError(f"Couldn't find model under {model_path}.")
            self.model = load_learner(model_path)
        except KeyError as e:
            raise KeyError(f"Config must have a value for {e}.")

    def overlay_shapefiles_on_images(self, n_images, show=False):
        """
        Overlays shapefiles on corresponding images and saves or displays the result.

        Parameters
        ----------
            n_images : int
                Number of images to process.
            show : bool, optional
                Whether to display the images or save them to disk (default: False).
        """
        shapefiles = [f for f in self.shp_dir.iterdir() if f.name.endswith('.shp')]
        for shapefile in islice(shapefiles, 0, n_images):
            if shapefile.name.endswith('.shp'):
                # Construct the corresponding image file path
                image_file = shapefile.name.replace('_predicted.shp', '.tif') 
                image_path = self.images_dir / image_file

                if image_path.exists():
                    # Read the shapefile
                    gdf = gpd.read_file(shapefile)

                    # Open the image
                    with rasterio.open(image_path) as ds:
                        transform = ds.transform

                        img = Image.fromarray(ds.read().transpose((1,2,0)))
                        draw = ImageDraw.Draw(img)

                        for geometry in gdf.geometry:
                            # Convert the polygon coordinates to image pixel coordinates and overlay to img
                            if geometry.geom_type == 'Polygon':
                                polygon = [crs_to_pixel_coords(x, y, transform) for x, y in zip(geometry.exterior.coords.xy[0], geometry.exterior.coords.xy[1])]
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
        """
        Computes precision, recall, accuracy, dice coefficient, and IoU for the model's predictions.

        Returns
        -------
            pd.DataFrame
                A dataframe containing the computed metrics.
        """
        sum_intersect = 0
        sum_total_pred = 0
        sum_total_truth = 0
        sum_union = 0
        sum_xor = 0

        num_files = 0

        gt_images = [f for f in self.masks_dir.iterdir() if f.suffix.lower() in ['.tif', '.tiff']]
        for groundtruth_path in gt_images:
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
                "EvalTimestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        """
        Generates predictions, overlays them on images, computes metrics, and saves the results.

        Parameters
        ----------
            n_images : int, optional
                Number of overlays to generate (default: 10).

        """
        if self.generate_preds:
            map_gen = MapGenerator(self.config)
            map_gen.create_tile_inferences()
        self.overlay_shapefiles_on_images(n_images)
        metrics = self.compute_metrics()
        output_file_path = self.output_dir / self.model_name.replace('.pkl', '_metrics.csv')
        if output_file_path.exists():
            df = pd.read_csv(output_file_path)
            df = df.append(metrics, ignore_index=True)
        else:
            df = metrics
        df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    config = load_config("base_config.yaml")
    evaluator = Evaluator(config)
    evaluator.evaluate(n_images=0)