import shutil
import time
from pathlib import Path

import geopandas as gpd
import pytest
import rasterio
from preprocess.data_tiler import DataTiler


class Test_DataTiler:

    @pytest.fixture(
        scope="class",
        params=[
            "aerial_training",
            "aerial_inference",
            "aerial_single",
            "satellite_training",
            "satellite_inference",
        ],
    )
    def name(self, request):
        return request.param

    @pytest.fixture
    def data_tiler(self, name, tmp_path_factory):

        tmp_path = tmp_path_factory.mktemp("data_tiler_test")
        input_path = Path("beam_model_training/tests") / name.split("_")[0]
        dir = tmp_path / name
        config = {"root_dir": dir}

        dir.mkdir(parents=True, exist_ok=True)
        (dir / DataTiler.DIR_STRUCTURE["labels"]).mkdir(parents=True, exist_ok=True)
        # Copy labels expect for inference tests
        if not name.endswith("inference"):
            for file in (input_path / DataTiler.DIR_STRUCTURE["labels"]).iterdir():
                shutil.copy2(file, dir / DataTiler.DIR_STRUCTURE["labels"])

        (dir / DataTiler.DIR_STRUCTURE["images"]).mkdir(parents=True, exist_ok=True)
        image_files = list((input_path / DataTiler.DIR_STRUCTURE["images"]).iterdir())
        copy_count = 1 if name.endswith("single") else len(image_files)
        for file in image_files[:copy_count]:
            shutil.copy2(file, dir / DataTiler.DIR_STRUCTURE["images"])
        try:
            yield DataTiler(config)
        finally:
            # Clean up after tests.
            time.sleep(3)
            shutil.rmtree(dir)

    def test_load_images(self, name, data_tiler):
        assert data_tiler.images, f"{name}: Images object should not be empty."
        expected_length = 1 if name.endswith("single") else 2
        assert (
            len(data_tiler.images) == expected_length
        ), f"{name}: Expecting a list of {expected_length} images."

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
            labels_path = (
                data_tiler.input_path
                / "labels"
                / (
                    "AERIAL_TEST_LABELS.shp"
                    if name.startswith("aerial")
                    else "SAT_TEST_LABELS.csv"
                )
            )
            input_labels = data_tiler.load_labels(labels_path, crop=False)
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
        data_tiler.generate_tiles(tile_size=512, write_tmp_files=True)

        image_dir = data_tiler.dir_structure.get("image_tiles")
        image_files = list(image_dir.glob("*.TIF"))

        expected_files = 4 if name.endswith("single") else 8
        assert len(image_files) == expected_files

        if not name.endswith("inference"):
            mask_dir = data_tiler.dir_structure.get("mask_tiles")
            mask_files = list(mask_dir.glob("*.TIF"))

            assert len(image_files) == len(
                mask_files
            ), f"{name}: Number of images and mask tiles do not match."

            for image_file in image_files:
                mask_file = mask_dir / image_file.name
                assert mask_file.exists()

                with rasterio.open(image_file) as image, rasterio.open(
                    mask_file
                ) as mask:
                    assert (
                        image.crs == mask.crs
                    ), f"{name}: Image and mask CRS do not match"
                    assert (
                        image.transform == mask.transform
                    ), f"{name}: Image and mask transform do not match"
