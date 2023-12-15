import shutil
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastai.vision.all import PILMask
from preprocess.data_tiler import DataTiler
from preprocess.transform import gen_train_test
from segmentation.train import Trainer
from utils.helpers import create_if_not_exists

default_config = {
        "config_name": "satellite_config",
        "config_value": {
            "seed": 42,
            "codes": ["Background", "Building"],
            "tile_size": 256,
            "test_size": 0.2,
            "train": {
                "architecture": "u-net",
                "backbone": "resnet34",
                "fit_type": "one_cycle",
                "epochs": 10,
                "loss_function": "CombinedLoss",
                "batch_size": 8
            },
            "root_dir": "beam_model_training/tests/satellite",
            "dirs": {
                "models": "models",
                "train": "train",
                "tiles": "tiles",
                "image_tiles": "tiles/images",
                "images": "images",
                "labels": "labels",
                "mask_tiles": "tiles/masks",
                "test": "test",
            },
        }
    }

mock_configs = {"default": default_config,
                "aerial_images": dict(default_config, root_dir="beam_model_training/tests/aerial")
                }

class TestTrainer:
    
    @pytest.fixture(scope="class", params=mock_configs.values(), ids=mock_configs.keys())
    def config(self, request: pytest.FixtureRequest):
            return request.param["config_value"]

    @pytest.fixture
    def trainer(self, config: Any):
        input_path = Path(config["root_dir"])
        tiles_path = input_path / "tiles"
        if not tiles_path.exists():
            data_tiler = DataTiler(config)
            data_tiler.generate_tiles(config["tile_size"])
            gen_train_test(config["root_dir"], config["dirs"], test_size=config["test_size"])
        try:
            yield Trainer(config)
        finally:
            # Clean up after tests.
            time.sleep(3)
            shutil.rmtree(input_path / config["dirs"]["tiles"])
            shutil.rmtree(input_path / config["dirs"]["test"])
            shutil.rmtree(input_path / config["dirs"]["train"])
            shutil.rmtree(input_path / config["dirs"]["models"])

    def test_map_unique_classes(self, trainer: Trainer):
         
        # Mocking the get_image_files and rxr.open_rasterio functions
        # You might need to adjust this part based on your actual implementation

        result = trainer._map_unique_classes()

        assert isinstance(result, dict), "Result should be a dictionary"
        print(result)
        assert len(result.keys()) == 2, "Result should contain 4 unique classes"
        assert result == {0: 0, 1: 255}, "Result should map index to class value correctly"

        # Test with is_partial=True
        result_partial = trainer._map_unique_classes(is_partial=True)
        assert isinstance(result_partial, dict), "Result should be a dictionary"

    def test_get_mask_shape(self, config, trainer):
        pixel_to_class = {0:0, 1: 255}
        image_path = next(trainer.images_dir.iterdir())
        mask = trainer._get_mask(image_path, pixel_to_class)
        assert isinstance(mask, PILMask)
        assert mask.size == (config["tile_size"], config["tile_size"])