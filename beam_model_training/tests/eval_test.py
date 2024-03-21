import os
from unittest import mock
import numpy as np
import pytest
import shutil
import ssl
import time
from segmentation.eval import Evaluator
from segmentation.infer import MapGenerator
import xarray as xr
import rioxarray as rxr
import yaml

from fastai.vision.all import get_files
from pathlib import Path

from preprocess.data_tiler import DataTiler
from preprocess.transform import gen_train_test
from segmentation.train import Trainer


base_config = {
    "config_name": "satellite_unet",
    "config_value": {
        "seed": 2022,
        "codes": ["Background", "Building"],
        "test_size": 0.2,
        "tiling": {"tile_size": 256, "erosion": True, "distance_weighting": False},
        "train": {
            "architecture": "u-net",
            "backbone": "resnet18",
            "epochs": 10,
            "loss_function": None,
            "batch_size": 8,
        },
    },
}


mock_configs = {
    "satellite_unet": base_config,
    # "aerial_unet": create_config("aerial_unet", base_config),
}


def create_mock_mask(src, polygons):
    """
    Creates a binary mask DataArray given the source DataArray with multiple bands and a list of polygons.

    Parameters
    ----------
    src : xarray.DataArray
        The source DataArray with multiple bands.
    polygons : list of tuples
        Each tuple contains the coordinates for the top-left and bottom-right points of a rectangle.

    Returns
    -------
    xarray.DataArray
        A binary mask DataArray with a single band.
    """
    # create the numpy array with the polygons
    mask = np.zeros((src.sizes["y"], src.sizes["x"]), dtype=np.uint8)
    for top_left, bottom_right in polygons:
        mask[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] = 255

    # create a DataArray from the numpy array
    mask_da = xr.DataArray(
        mask, dims=("y", "x"), coords={"y": src.coords["y"], "x": src.coords["x"]}
    )
    mask_da = mask_da.rio.write_crs(src.rio.crs)
    mask_da = mask_da.rio.write_transform(src.rio.transform())

    return mask_da


class TestEvaluator:

    @pytest.fixture(
        scope="class", params=mock_configs.values(), ids=mock_configs.keys()
    )
    def mock_config(self, request: pytest.FixtureRequest):
        return request.param

    @pytest.fixture
    def evaluator(self, mock_config, tmp_path_factory):
        name, config = mock_config["config_name"], mock_config["config_value"]
        project_dir = Path("beam_model_training/tests") / name.split("_")[0]
        tmp_path = tmp_path_factory.mktemp("eval")
        test_dir = tmp_path / name
        config_name = f"{name}_config.yaml"

        # Generate tiles once in the main directory
        tile_size = config["tiling"]["tile_size"]
        tiles_path = project_dir / Trainer.DIR_STRUCTURE["image_tiles"]
        if not tiles_path.exists():
            data_tiler = DataTiler(project_dir, "test_config.yaml")
            data_tiler.generate_tiles(tile_size)
            gen_train_test(project_dir, test_size=config["test_size"])

        try:
            # Copy the entire file structure from project_dir to the temporary directory.
            if project_dir.exists():
                for item in project_dir.iterdir():
                    s = project_dir / item.name
                    d = test_dir / item.name
                    if s.is_dir():
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)

            # Write the configuration data to the temporary YAML file.
            with open(test_dir / config_name, "w") as file:
                yaml.dump(config, file)

            with mock.patch(
                "segmentation.infer.load_learner"
            ) as mock_load_learner, mock.patch(
                "segmentation.eval.MapGenerator"
            ) as MockMapGenerator:
                mock_load_learner.return_value = None
                MockMapGenerator.return_value = MapGenerator(
                    test_dir, config_name, generate_preds=True, model_path="mocked.pkl"
                )
                evaluator = Evaluator(test_dir, config_name, generate_preds=True)

            # Creating mock tiles with known polygon locations.
            gt_polygons = [((30, 30), (70, 70))]
            pred_polygons = [((50, 50), (90, 90))]
            test_image_files = get_files(
                evaluator.test_images_dir, extensions=[".tif", ".tiff"]
            )

            for img_path in test_image_files[:2]:
                with rxr.open_rasterio(img_path) as src:
                    gt_mask = create_mock_mask(src, gt_polygons)
                    gt_mask_path = evaluator.test_masks_dir / img_path.name
                    gt_mask.rio.to_raster(gt_mask_path)
                    gt_mask.name = img_path.stem

                    pred_mask = create_mock_mask(src, pred_polygons)
                    pred_mask_path = (
                        evaluator.predictions_dir / f"{img_path.stem}_INFERENCE.TIF"
                    )
                    pred_mask.name = img_path.stem
                    pred_mask.rio.to_raster(pred_mask_path)

                    shp_path = (
                        evaluator.shapefiles_dir / f"{img_path.stem}_predicted.shp"
                    )
                    vector_df = evaluator.map_generator.create_shp_from_mask(pred_mask)
                    vector_df.to_file(shp_path, driver="ESRI Shapefile")
            for img_path in test_image_files[2:]:
                os.remove(img_path)
            yield evaluator

        finally:
            # Clean up after tests.
            time.sleep(3)
            shutil.rmtree(test_dir)

    def test_overlay_shapefiles_on_images(self, evaluator):
        evaluator.overlay_shapefiles_on_images(n_images=2)
        overlay_files = get_files(evaluator.eval_dir, extensions=[".png"])
        assert (
            len(overlay_files) == 2
        ), f"Two images should have been created with overlay, {len(overlay_files)} found."

    def test_compute_metrics_high_threshold(self, evaluator):
        metrics = evaluator.compute_metrics(iou_threshold=0.5)
        assert metrics.iloc[0].Precision == 0.25, "Precision should be 0.25."
        assert metrics.iloc[0].Recall == 0.25, "Precision should be 0.25."
        assert metrics.iloc[0].Accuracy == 0.963, "Accuracy should be 0.963."
        assert metrics.iloc[0].Dice == 0.25, "Dice coefficient should be 0.25."
        assert metrics.iloc[0].IoU == 0.143, "IoU coefficient should be 0.143."
        assert (
            metrics.iloc[0]["Building Precision @ 0.5 IoU"] == 0
        ), "Building precision @ 0.5 IoU should be 0."
        assert (
            metrics.iloc[0]["Building Recall @ 0.5 IoU"] == 0
        ), "Building recall @ 0.5 IoU should be 0."

    def test_compute_metrics_low_threshold(self, evaluator):
        metrics = evaluator.compute_metrics(iou_threshold=0.1)
        assert metrics.iloc[0].Precision == 0.25, "Precision should be 0.25."
        assert metrics.iloc[0].Recall == 0.25, "Precision should be 0.25."
        assert metrics.iloc[0].Accuracy == 0.963, "Accuracy should be 0.963."
        assert metrics.iloc[0].Dice == 0.25, "Dice coefficient should be 0.25."
        assert metrics.iloc[0].IoU == 0.143, "IoU coefficient should be 0.143."
        assert (
            metrics.iloc[0]["Building Precision @ 0.1 IoU"] == 1
        ), "Building precision @ 0.1 IoU should be 1."
        assert (
            metrics.iloc[0]["Building Recall @ 0.1 IoU"] == 1
        ), "Building recall @ 0.1 IoU should be 1."
