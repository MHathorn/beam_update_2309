import pytest
import shutil
import ssl
import time
import yaml

from fastai.vision.all import PILMask
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
    "aerial_unet": create_config("aerial_unet", base_config),
    "satellite_hrnet_w18": create_config(
        "satellite_hrnet_w18", base_config, architecture="HRNet", backbone="hrnet_w18"
    ),
    "aerial_hrnet_w18": create_config(
        "aerial_hrnet_w18",
        base_config,
        architecture="HRNet",
        backbone="hrnet_w18",
    ),
}


ssl._create_default_https_context = ssl._create_unverified_context


class TestTrainer:

    @pytest.fixture(
        scope="class", params=mock_configs.values(), ids=mock_configs.keys()
    )
    def mock_config(self, request: pytest.FixtureRequest):
        return request.param

    @pytest.fixture
    def trainer(
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
            yield Trainer(test_dir, config_name)
        finally:
            # Clean up after tests.
            time.sleep(3)
            shutil.rmtree(test_dir)

    def test_map_unique_classes(self, trainer: Trainer):

        # You might need to adjust this part based on your actual implementation

        result = trainer._map_unique_classes()

        assert isinstance(result, dict), "Result should be a dictionary"
        assert len(result.keys()) == 2, "Result should contain 2 unique classes"
        assert result == {
            0: 0,
            1: 255,
        }, "Result should map index to class value correctly"

        # Test with is_partial=True
        result_partial = trainer._map_unique_classes(is_partial=True)
        assert isinstance(result_partial, dict), "Result should be a dictionary"

    def test_get_mask_shape(self, mock_config, trainer):
        pixel_to_class = {0: 0, 1: 255}
        tile_size = mock_config["config_value"]["tiling"]["tile_size"]
        image_path = next(trainer.train_images_dir.iterdir())
        mask = trainer._get_mask(image_path, pixel_to_class)
        assert isinstance(mask, PILMask)
        assert mask.size == (
            tile_size,
            tile_size,
        )

    def test_run(self, trainer):
        with patch(
            "segmentation.train.get_segmentation_learner"
        ) as MockGetSegmentationLearner, patch(
            "segmentation.train.unet_learner"
        ) as MockUnetLearner, patch(
            "segmentation.train.Trainer._save"
        ) as MockSave:

            mock_get_segmentation_learner = MockGetSegmentationLearner.return_value
            mock_unet_learner = MockUnetLearner.return_value

            trainer.run()

            # Assert that either get_segmentation_learner or unet_learner was called once
            assert (
                MockGetSegmentationLearner.call_count + MockUnetLearner.call_count == 1
            )

            # Assert that the fit_one_cycle method was called on the learner instance
            if trainer.train_params["architecture"].lower() == "hrnet":
                mock_get_segmentation_learner.fit_one_cycle.assert_called_once()
            elif trainer.train_params["architecture"].lower() == "u-net":
                mock_unet_learner.fit_one_cycle.assert_called_once()

            MockSave.assert_called_once()
