import argparse
from datetime import datetime
from itertools import islice
import logging

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
from fastai.vision.all import load_learner
from PIL import Image, ImageDraw

from segmentation.infer import MapGenerator
from segmentation.train import Trainer
from utils.base_class import BaseClass
from utils.helpers import crs_to_pixel_coords, seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Evaluator(BaseClass):
    """
    A class used to evaluate the performance of a segmentation model.

    Attributes
    ----------
    Args:
    project_dir : str
        Path to the project directory, containing one or more models, as well as images and masks for evaluation.
    config_name : str (optional)
        The configuration file name. If missing, the constructor will look for a single file in the project directory.
    generate_preds : bool
        Flag indicating whether predictions should be generated during evaluation.
    model_path:
        The path to the model, if not set by the config file.

    Methods
    -------
    overlay_shapefiles_on_images(n_images, show=False):
        Overlays shapefiles on corresponding images and saves or displays the result.
    compute_metrics():
        Computes precision, recall, accuracy, dice coefficient, and IoU for the model's predictions.
    evaluate(n_images=10):
        Generates predictions, overlays them on images, computes metrics, and saves the results.
    """

    def __init__(
        self,
        project_dir,
        config_name=None,
        generate_preds=False,
        model_path=None,
    ):
        """
        Constructs all the necessary attributes for the Evaluator object.

        Parameters
        ----------
        project_dir : str
            Path to the project directory, containing one or more models, as well as images and masks for evaluation.
        config_name : str
            The configuration file name. If missing, the constructor will look for a single file in the project directory.
        generate_preds : bool
            Flag indicating whether predictions should be generated during evaluation.
        model_path:
            The path to the model, if not set by the config file.
        """

        super().__init__(project_dir, config_name)
        seed(self.config["seed"])

        self.generate_preds = generate_preds
        read_dirs = ["test_images", "test_masks", "models", "eval"]
        write_dirs = []
        if self.generate_preds:
            write_dirs += ["shapefiles", "predictions"]
        else:
            read_dirs += ["shapefiles", "predictions"]
        super().load_dir_structure(read_dirs=read_dirs, write_dirs=write_dirs)
        if len(list(self.predictions_dir.iterdir())) == 0 and not self.generate_preds:
            raise FileNotFoundError(
                "Predictions directory is empty. Set `generate_preds` to true to generate predictions."
            )
        if model_path:
            self.model_version = Path(model_path).stem
        else:
            try:
                self.model_version = self.config["model_version"]
                model_path = super().load_model_path(self.model_version)
            except KeyError as e:
                raise KeyError(f"Config must have a value for {e}.")

        self.model = load_learner(model_path)

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
        shapefiles = [
            f for f in self.shapefiles_dir.iterdir() if f.name.endswith(".shp")
        ]
        for shapefile in islice(shapefiles, 0, n_images):
            # Construct the corresponding image file path
            image_file = shapefile.name.replace("_predicted.shp", ".TIF")
            image_path = self.test_images_dir / image_file

            if image_path.exists():
                # Read the shapefile
                gdf = gpd.read_file(shapefile)

                # Open the image
                with rasterio.open(image_path) as ds:
                    transform = ds.transform

                    img = Image.fromarray(ds.read().transpose((1, 2, 0)))
                    draw = ImageDraw.Draw(img)

                    for geometry in gdf.geometry:
                        # Convert the polygon coordinates to image pixel coordinates and overlay to img
                        if geometry.geom_type == "Polygon":
                            polygon = [
                                crs_to_pixel_coords(x, y, transform)
                                for x, y in zip(
                                    geometry.exterior.coords.xy[0],
                                    geometry.exterior.coords.xy[1],
                                )
                            ]
                            draw.polygon(polygon, outline="red")

                    # Display the image
                    if show == True:
                        plt.imshow(img)
                        plt.title(image_file)
                        plt.show()
                    else:
                        output_path = self.eval_dir / f"eval_pred_{image_file}".replace(
                            ".TIF", ".png"
                        )
                        img.save(output_path)
                        logging.info(
                            f"Image file {output_path.name} written to `{output_path.parent.name}`."
                        )

            else:
                logging.warning(f"Image file {image_file} not found.")

    def compute_metrics(self, map_gen, iou_threshold=0.5):
        """
        Computes precision, recall, accuracy, dice coefficient, and IoU for the model's predictions.

        Parameters
        ----------
        map_gen : MapGenerator
            An instance of MapGenerator to convert masks to shapefiles.
        iou_threshold : float, optional
            The IoU threshold to consider when calculating building-level precision and recall (default: 0.5).

        Returns
        -------
        pd.DataFrame
            A dataframe containing the computed metrics.
        """

        def calculate_building_iou(poly1, poly2):
            intersection = poly1.intersection(poly2).area
            union = poly1.union(poly2).area
            return intersection / union

        sum_intersect = 0
        sum_total_pred = 0
        sum_total_truth = 0
        sum_union = 0
        sum_xor = 0
        buildings_intersect = 0
        buildings_total_pred = 0
        buildings_total_truth = 0

        num_files = 0

        gt_images = [
            f
            for f in self.test_masks_dir.iterdir()
            if f.suffix.lower() in [".tif", ".tiff"]
        ]
        for groundtruth_path in gt_images:
            pred_mask_path = self.predictions_dir / (
                groundtruth_path.stem + "_INFERENCE.TIF"
            )

            if pred_mask_path.exists():

                # Load ground truth mask
                gt_mask = rxr.open_rasterio(
                    groundtruth_path, default_name=groundtruth_path.name
                )
                gt_gdf = map_gen.create_shp_from_mask(gt_mask)
                gt_values = (
                    (gt_mask / 255).astype(int).values
                )  # Convert to binary (0 and 1)

                # Load predicted mask
                pred_mask = rxr.open_rasterio(
                    pred_mask_path, default_name=pred_mask_path.name
                )
                pred_gdf = map_gen.create_shp_from_mask(pred_mask)
                pred_values = (pred_mask > 0.5).astype(int).values

                # Accumulate values for metrics
                intersect = np.sum(pred_values * gt_values)
                sum_intersect += intersect
                sum_total_pred += np.sum(pred_values)
                sum_total_truth += np.sum(gt_values)
                sum_union += np.sum(pred_values) + np.sum(gt_values) - intersect
                sum_xor += np.sum(pred_values == gt_values)
                buildings_total_pred += len(pred_gdf)
                buildings_total_truth += len(gt_gdf)

                for pred_poly in pred_gdf.geometry:
                    for gt_poly in gt_gdf.geometry:
                        iou = calculate_building_iou(pred_poly, gt_poly)
                        if iou >= iou_threshold:
                            buildings_intersect += 1

                num_files += 1
            else:
                logging.warning(f"Predicted mask for {groundtruth_path} not found.")

        if num_files > 0:
            # Compute aggregated metrics
            precision = sum_intersect / sum_total_pred
            recall = sum_intersect / sum_total_truth
            accuracy = sum_xor / (sum_union + sum_xor - sum_intersect)
            dice = 2 * sum_intersect / (sum_total_pred + sum_total_truth)
            iou = sum_intersect / sum_union
            building_precision = buildings_intersect / buildings_total_pred
            building_recall = buildings_intersect / buildings_total_truth

            # Create a DataFrame with the aggregated metrics
            metrics = {
                "EvalTimestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ModelVersion": self.model_version,
                "Precision": [round(precision, 3)],
                "Recall": [round(recall, 3)],
                "Accuracy": [round(accuracy, 3)],
                "Dice": [round(dice, 3)],
                "IoU": [round(iou, 3)],
                f"Building Precision @ {iou_threshold} IoU": [
                    round(building_precision, 3)
                ],
                f"Building Recall @ {iou_threshold} IoU": [round(building_recall, 3)],
            }
            return pd.DataFrame(metrics)
        else:
            return pd.DataFrame()

    def evaluate(self, n_images=10, iou_threshold=0.5):
        """
        Generates predictions if required, overlays them on images, computes metrics, and saves the results.

        Parameters
        ----------
        n_images : int, optional
            Number of overlays to generate (default: 10).
        iou_threshold : float, optional
            The IoU threshold to consider when calculating building-level precision and recall (default: 0.5).
        """
        map_gen = MapGenerator(
            self.project_dir, self.config_name, generate_preds=self.generate_preds
        )
        if self.generate_preds:
            map_gen.create_tile_inferences()
        self.overlay_shapefiles_on_images(n_images)
        metrics = self.compute_metrics(map_gen, iou_threshold)
        output_file_path = self.eval_dir / (self.model_version + "_metrics.csv")
        if output_file_path.exists():
            try:
                df = pd.read_csv(output_file_path)
                df = pd.concat([df, metrics], ignore_index=True)
            except pd.errors.EmptyDataError:
                df = metrics
        else:
            df = metrics
        df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the segmentation model performance."
    )
    parser.add_argument(
        "-d", "--project_dir", type=str, help="The project directory.", required=True
    )  # required
    parser.add_argument(
        "-c",
        "--config_name",
        type=str,
        help="The configuration file name. If missing, the constructor will look for a single file in the project directory.",
    )  # optional
    parser.add_argument(
        "--generate_preds",
        default=False,
        action="store_true",
        help="Flag indicating whether predictions should be generated during evaluation. Defaults to False.",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=10,
        help="Number of images to overlay shapefiles on. Defaults to 10.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="The IoU threshold for computing metrics. Defaults to 0.5.",
    )

    args = parser.parse_args()

    evaluator = Evaluator(
        project_dir=args.project_dir,
        config_name=args.config_name,
        generate_preds=args.generate_preds,
    )
    evaluator.evaluate(n_images=args.n_images, iou_threshold=args.iou_threshold)
