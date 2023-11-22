import pytest
from tiling.data_tiler import DataTiler
import os
from pathlib import Path

class Test_DataTiler:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        print("\nSetup...")
    
        # Create subdirectories for image_dir and output_dir
        path = os.getcwd()
        test_dirs = {
            "aerial": Path("beam_model_training/tests/aerial"),
            "satellite": Path("beam_model_training/tests/satellite")
        }

        self.data_tilers = {name: DataTiler(dir / "images", dir / "output") for name, dir in test_dirs.items()}


    def teardown_method(self, method):
        print("\nTearing down test...")
        # Remove the image file
        for data_tiler in self.data_tiler.values():
            if data_tiler.output_dir.exists():
                data_tiler.output_dir.rmdir()

    def test_load_images(self):
        for data_tiler in self.data_tilers.values():
            assert data_tiler.images  # This will check if images is not empty


    # Add code to the following test methods
    def test_load_labels(self):
        for data_tiler in self.data_tilers.values():
            assert data_tiler.labels is None

    def test_crop_buildings(self):
        pass

    def test_create_subdir(self):
        for data_tiler in self.data_tilers.values():
            assert (data_tiler.output_dir / 'tmp').exists()
            assert (data_tiler.output_dir / 'image_tiles').exists()
            assert not (data_tiler.output_dir / 'mask_tiles').exists()

    def test_generate_mask(self):
        pass
        
    def test_generate_tiles(self):
        # Assuming generate_tiles takes tile_size as an argument and populates 'image_tiles' directory with tile images
        for data_tiler in self.data_tilers.values():
            data_tiler.generate_tiles(tile_size=512)

            image_tiles_dir = data_tiler.output_dir / 'image_tiles'
            image_files = list(image_tiles_dir.glob('*.TIF'))

            assert len(image_files) == 8
