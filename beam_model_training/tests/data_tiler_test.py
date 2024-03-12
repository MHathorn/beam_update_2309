import geopandas as gpd
import pytest
import rasterio
import shutil
import time
import types

from collections.abc import Generator
from pathlib import Path

from preprocess.data_tiler import DataTiler
from tests.fixture_config import create_config
import yaml

base_config = {
    "config_name": "satellite_config",
    "config_value": {
        "seed": 2022,
        "tiling": {"tile_size": 256, "erosion": True, "distance_weighting": False},
    },
}

mock_configs = {
    name: create_config(name, base_config)
    for name in [
        "aerial_training",
        "aerial_inference",
        "aerial_single",
        "satellite_training",
        "satellite_inference",
    ]
}

mock_configs["satellite_weighting"] = create_config(
    "satellite_weighting", base_config, distance_weighting=True
)


class Test_DataTiler:

    @pytest.fixture(
        scope="class", params=mock_configs.values(), ids=mock_configs.keys()
    )
    def mock_config(self, request):
        return request.param

    @pytest.fixture
    def data_tiler(self, mock_config, tmp_path_factory):
        name, config = mock_config["config_name"], mock_config["config_value"]
        tmp_path = tmp_path_factory.mktemp("data_tiler_test")
        input_path = Path("beam_model_training/tests") / name.split("_")[0]
        test_dir = tmp_path / name
        config_name = f"{name}_config.yaml"

        test_dir.mkdir(parents=True, exist_ok=True)
        with open(test_dir / config_name, "w") as file:
            yaml.dump(config, file)

        # Copy labels expect for inference tests
        (test_dir / DataTiler.DIR_STRUCTURE["labels"]).mkdir(
            parents=True, exist_ok=True
        )
        if not name.endswith("inference"):
            for file in (input_path / DataTiler.DIR_STRUCTURE["labels"]).iterdir():
                shutil.copy2(file, test_dir / DataTiler.DIR_STRUCTURE["labels"])

        # Copy images
        (test_dir / DataTiler.DIR_STRUCTURE["images"]).mkdir(
            parents=True, exist_ok=True
        )
        image_files = list((input_path / DataTiler.DIR_STRUCTURE["images"]).iterdir())
        copy_count = 1 if name.endswith("single") else len(image_files)
        for file in image_files[:copy_count]:
            shutil.copy2(file, test_dir / DataTiler.DIR_STRUCTURE["images"])
        try:
            yield DataTiler(test_dir, config_name)
        finally:
            # Clean up after tests.
            time.sleep(3)
            shutil.rmtree(test_dir)

    def test_load_images(self, name, data_tiler):
        assert isinstance(
            data_tiler.images_generator, (Generator, types.GeneratorType)
        ), f"{name}: Images generator should be a generator."

    def test_load_labels(self, name, data_tiler):
        if name.endswith("inference"):
            assert (
                data_tiler.labels is None
            ), f"{name}: Expecting empty labels for inference."
        else:
            assert isinstance(
                data_tiler.labels, gpd.GeoDataFrame
            ), f"{name}: Object is not a GeoDataFrame"

    def test_crop_buildings(self, name, data_tiler):
        if name.endswith("single"):
            labels_path = data_tiler.root_dir / "labels"
            valid_label_paths = [
                l
                for l in labels_path.glob("*")
                if l.suffix in [".csv", ".shp"] or l.name.endswith(".csv.gz")
            ]
            input_labels = data_tiler._load_labels(valid_label_paths)
            input_label_bbox = input_labels.total_bounds
            cropped_labels_bbox = data_tiler.labels.to_crs("EPSG:4326").total_bounds

            assert (
                cropped_labels_bbox[2] - cropped_labels_bbox[0]
                < input_label_bbox[2] - input_label_bbox[0]
            ), f"{name}: Widths of the cropped labels should be reduced."
            assert (
                cropped_labels_bbox[3] - cropped_labels_bbox[1]
                < input_label_bbox[3] - input_label_bbox[1]
            ), f"{name}: Heights of the cropped labels should be reduced."

    def test_generate_tiles(self, name, data_tiler):
        data_tiler.generate_tiles(tile_size=512)
        image_files = list(data_tiler.image_tiles_dir.glob("*.TIF"))

        expected_files = 4 if name.endswith("single") else 8
        assert len(image_files) == expected_files

        if not name.endswith("inference"):
            mask_files = list(data_tiler.mask_tiles_dir.glob("*.TIF"))

            assert len(image_files) == len(
                mask_files
            ), f"{name}: Number of images and mask tiles do not match."

            image_file = image_files[0]
            mask_file = data_tiler.mask_tiles_dir / image_file.name
            assert mask_file.exists()

            with rasterio.open(image_file) as image, rasterio.open(mask_file) as mask:
                assert image.crs == mask.crs, f"{name}: Image and mask CRS do not match"
                assert (
                    image.transform == mask.transform
                ), f"{name}: Image and mask transform do not match"

        if name.endswith("weighting"):
            weight_files = list(data_tiler.weight_tiles_dir.glob("*.TIF"))
            assert len(image_files) == len(
                weight_files
            ), f"{name}: Number of images and weight tiles do not match."
        else:
            weight_tiles_dir = (
                data_tiler.root_dir + data_tiler.DIR_STRUCTURE["weight_tiles"]
            )
            assert (
                not weight_tiles_dir.exists()
            ), f"The directory {weight_tiles_dir} should not exist."
