import pytest
from segmentation.lightning.data import BuildingSegmentationDataModule
from segmentation.lightning.models import BuildingSegmentationModule

def test_data_module_setup(mock_config, tmp_path):
    data_module = BuildingSegmentationDataModule(tmp_path, mock_config)
    data_module.setup()
    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None