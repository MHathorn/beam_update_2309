import shutil
import pytest
from tiling.data_tiler import DataTiler
import geopandas as gpd
from pathlib import Path
import rasterio

class Test_DataTiler:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        print("\nSetup...")
    
        # Create subdirectories for image_dir and output_dir
        test_dirs = {
            "aerial": Path("beam_model_training/tests/aerial"),
            "satellite": Path("beam_model_training/tests/satellite")
        }

        self.data_tilers = {name: DataTiler(dir) for name, dir in test_dirs.items()}


    def teardown_method(self, method):
        print("\nTearing down test...")
        # Remove the image file
        for data_tiler in self.data_tilers.values():
            output_path = data_tiler.input_path / "tiles"
            shutil.rmtree(output_path)

    def test_load_images(self):
        for data_tiler in self.data_tilers.values():
            assert data_tiler.images  # This will check if images is not empty


    # Add code to the following test methods
    def test_load_labels(self):
        for data_tiler in self.data_tilers.values():
            assert isinstance(data_tiler.labels, gpd.GeoDataFrame), "Object is not a GeoDataFrame"


    def test_crop_buildings(self):
        pass

    def test_create_subdir(self):
        for data_tiler in self.data_tilers.values():
            assert (data_tiler.input_path / "tiles").exists()
            assert (data_tiler.input_path / 'tiles/images').exists()
            if data_tiler.labels is not None:
                assert (data_tiler.input_path / 'tiles/masks').exists()

    def test_generate_mask(self):    

        pass


    def test_generate_tiles(self):
        
        # Assuming generate_tiles takes tile_size as an argument and populates 'image_tiles' directory with tile images
        for data_tiler in self.data_tilers.values():
            data_tiler.generate_tiles(tile_size=512, write_tmp_files=True)

            image_dir = data_tiler.dir_structure.get('image_tiles')
            image_files = list(image_dir.glob('*.TIF'))

            assert len(image_files) == 8

            mask_dir = data_tiler.dir_structure.get('mask_tiles')
            mask_files = list(image_dir.glob('*.TIF'))
    
            assert len(image_files) == len(mask_files), "Number of images and label masks does not match."
    
            for image_file in image_files:
                # Remove '_mask.tif' from the end of the corresponding mask file name
                mask_file_name = f"{image_file.stem}_mask.tif"
                mask_file = mask_dir / mask_file_name
        
                assert mask_file.exists()
        
                with rasterio.open(image_file) as image, rasterio.open(mask_file) as mask:
                    assert image.crs == mask.crs, "Image and mask CRS do not match"
                    assert image.transform == mask.transform, "Image and mask transform do not match"
    