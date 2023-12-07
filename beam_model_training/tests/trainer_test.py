import shutil
import time
from pathlib import Path
import numpy as np

import pytest
from unittest.mock import patch
from preprocess.data_tiler import DataTiler
from preprocess.transform import gen_train_test
from segmentation.train import Trainer

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
            }
        }
    }

mock_configs = [default_config]

class TestTrainer:
    
    @pytest.fixture(scope="class", params=mock_configs)
    def config(self, request):
            return request.param["config_value"]

    @pytest.fixture
    def trainer(self, config):
        input_path = Path(config["root_dir"])
        # data_tiler = DataTiler(input_path, input_path / "labels")
        # data_tiler.generate_tiles(config["tile_size"])
        # gen_train_test(input_path / config["dirs"]["tiles"], test_size=config["test_size"])
        try:
            yield Trainer(config)
        finally:
            # Clean up after tests.
            time.sleep(3)
            shutil.rmtree(input_path / "tiles")

    def test_map_unique_classes(self, trainer):
         
        # Mocking the get_image_files and rxr.open_rasterio functions
        # You might need to adjust this part based on your actual implementation

        result = trainer._map_unique_classes()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result) == 4, "Result should contain 4 unique classes"
        assert result == {0: 1, 1: 2, 2: 3, 3: 4}, "Result should map index to class value correctly"

        # Test with is_partial=True
        result_partial = self._map_unique_classes(is_partial=True)

        assert isinstance(result_partial, dict), "Result should be a dictionary"