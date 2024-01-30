import shutil
import ssl
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastai.vision.all import PILMask
from preprocess.data_tiler import DataTiler
from preprocess.transform import gen_train_test
from segmentation.train import Trainer

default_config = {
        "config_name": "satellite_config",
        "config_value": {
            "seed": 2022,
            "codes": ["Background", "Building"],
            "tile_size": 512,
            "test_size": 0.2,
            "train": {
                "architecture": "u-net",
                "backbone": "resnet18",
                "epochs": 10,
                "loss_function": None,
                "batch_size": 8
            },
            "root_dir": "satellite",
        }
    }
mock_configs = {
    "sat_unet": default_config,
    "aerial_unet": dict(default_config, root_dir="aerial"),
    "sat_hrnet_w18": dict(default_config, architecture="HRNet", backbone="hrnet_w18"),
    "aerial_hrnet_w18": dict(default_config, root_dir="aerial", architecture="HRNet", backbone="hrnet_w18")
}


ssl._create_default_https_context = ssl._create_unverified_context

class TestTrainer:
    
    @pytest.fixture(scope="class", params=mock_configs.values(), ids=mock_configs.keys())
    def config(self, request: pytest.FixtureRequest):
            return request.param["config_value"]

    @pytest.fixture
    def trainer(self, config: Any):
        current_dir = Path(__file__).parent.resolve()
        config["root_dir"] = current_dir / config["root_dir"]
        tiles_path = config["root_dir"] / Trainer.DIR_STRUCTURE["image_tiles"]
        if not tiles_path.exists():
            data_tiler = DataTiler(config)
            data_tiler.generate_tiles(config["tiling"]["tile_size"])
            gen_train_test(config["root_dir"], test_size=config["test_size"])
        try:
            yield Trainer(config)
        finally:
            # Clean up after tests.
            time.sleep(3)
            shutil.rmtree(config["root_dir"] / Trainer.DIR_STRUCTURE["models"])

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
        assert mask.size == (config["tiling"]["tile_size"], config["tiling"]["tile_size"])

    def test_run(self, trainer):
        with patch('segmentation.train.get_segmentation_learner') as MockGetSegmentationLearner, \
            patch('segmentation.train.unet_learner') as MockUnetLearner, \
            patch('segmentation.train.Trainer._save') as MockSave:
            
            mock_get_segmentation_learner = MockGetSegmentationLearner.return_value
            mock_unet_learner = MockUnetLearner.return_value

            trainer.run()

            # Assert that either get_segmentation_learner or unet_learner was called once
            assert MockGetSegmentationLearner.call_count + MockUnetLearner.call_count == 1

            # Assert that the fit_one_cycle method was called on the learner instance
            if trainer.architecture.lower() == 'hrnet':
                mock_get_segmentation_learner.fit_one_cycle.assert_called_once()
            elif trainer.architecture.lower() == 'u-net':
                mock_unet_learner.fit_one_cycle.assert_called_once()

            MockSave.assert_called_once()
