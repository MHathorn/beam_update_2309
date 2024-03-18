import pytest
import shutil
import ssl
import time
from segmentation.infer import MapGenerator
import xarray
import yaml

from fastai.vision.all import PILMask, get_files
from pathlib import Path
from typing import Any
from unittest.mock import patch

from preprocess.data_tiler import DataTiler
from preprocess.transform import gen_train_test
from segmentation.train import Trainer
from tests.fixture_config import create_config


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
    # "satellite_hrnet_w18": create_config(
    #     "satellite_hrnet_w18", base_config, architecture="HRNet", backbone="hrnet_w18"
    # ),
    # "aerial_hrnet_w18": create_config(
    #     "aerial_hrnet_w18",
    #     base_config,
    #     architecture="HRNet",
    #     backbone="hrnet_w18",
    # ),
}


ssl._create_default_https_context = ssl._create_unverified_context


class TestMapGenerator:

    @pytest.fixture(
        scope="class", params=mock_configs.values(), ids=mock_configs.keys()
    )
    def mock_config(self, request: pytest.FixtureRequest):
        return request.param

    @pytest.fixture
    def map_generator(
        self, mock_config: Any, request: pytest.FixtureRequest, tmp_path_factory
    ):
        name, config = mock_config["config_name"], mock_config["config_value"]
        project_dir = Path("beam_model_training/tests") / name.split("_")[0]
        tmp_path = tmp_path_factory.mktemp("trainer_test")
        test_dir = tmp_path / name
        config_name = f"{name}_config.yaml"

        # Generate tiles once in the main directory
        tiles_path = project_dir / Trainer.DIR_STRUCTURE["image_tiles"]
        if not tiles_path.exists():
            data_tiler = DataTiler(project_dir, "test_config.yaml")
            data_tiler.generate_tiles(config["tiling"]["tile_size"])
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

            # Creating learner and map_generator
            trainer = Trainer(test_dir)
            trainer.set_learner()
            map_generator = MapGenerator(test_dir, generate_preds=True)
            map_generator.learner = trainer.learner
            yield map_generator

        finally:
            # Clean up after tests.
            time.sleep(3)
            shutil.rmtree(test_dir)

    def test_single_inference(self, mock_config, map_generator: MapGenerator):

        # Mocking the get_image_files and rxr.open_rasterio functions
        # You might need to adjust this part based on your actual implementation

        image_files = get_files(
            map_generator.test_images_dir, extensions=[".tif", ".tiff"]
        )

        prediction = map_generator.single_tile_inference(
            image_files[0], write_shp=False
        )

        tile_size = mock_config["config_value"]["tiling"]["tile_size"]

        assert isinstance(prediction, xarray.DataArray)
        assert prediction.shape == (
            tile_size,
            tile_size,
        )

    def test_parallel_inferences(self, map_generator: MapGenerator):
        map_generator.create_tile_inferences(write_shp=True, parallel=False)
        image_files = get_files(
            map_generator.test_images_dir, extensions=[".tif", ".tiff"]
        )
        inference_files = get_files(
            map_generator.predictions_dir, extensions=[".tif", ".tiff"]
        )
        assert len(inference_files) == len(
            image_files
        ), "The number of predictions doesn't match the number of test images."
