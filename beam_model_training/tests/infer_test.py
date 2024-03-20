import numpy as np
import pytest
import shutil
import ssl
import time
from segmentation.infer import MapGenerator
import xarray as xr
import rioxarray as rxr
import yaml

from fastai.vision.all import get_files
import geopandas as gpd
from pathlib import Path

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
}


ssl._create_default_https_context = ssl._create_unverified_context


class TestMapGenerator:

    @pytest.fixture(
        scope="class", params=mock_configs.values(), ids=mock_configs.keys()
    )
    def mock_config(self, request: pytest.FixtureRequest):
        return request.param

    @pytest.fixture
    def boundaries_path(self, mock_config):
        name = mock_config["config_name"]
        return f"AOIs/{name.split('_')[0]}_test_polygons.shp"

    @pytest.fixture
    def map_generator(self, mock_config, tmp_path_factory):
        name, config = mock_config["config_name"], mock_config["config_value"]
        project_dir = Path("beam_model_training/tests") / name.split("_")[0]
        tmp_path = tmp_path_factory.mktemp("map_gen")
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
            trainer = Trainer(test_dir, config_name)
            trainer.set_learner()
            map_generator = MapGenerator(test_dir, config_name, generate_preds=True)
            map_generator.learner = trainer.learner
            yield map_generator

        finally:
            # Clean up after tests.
            time.sleep(3)
            shutil.rmtree(test_dir)

    def test_create_shp_with_single_building(self, map_generator: MapGenerator):
        image_files = get_files(
            map_generator.test_images_dir, extensions=[".tif", ".tiff"]
        )
        mask_da = rxr.open_rasterio(image_files[0]).isel(band=0)

        mask_da.name = "test_mask_single"
        # Create a binary mask with a central square of 1s
        size_y, size_x = (
            mask_da.shape[-2],
            mask_da.shape[-1],
        )
        center_square_size = 20
        mask_array = np.zeros((size_y, size_x), dtype=np.uint8)
        start_y = size_y // 2 - center_square_size // 2
        start_x = size_x // 2 - center_square_size // 2
        end_y = start_y + center_square_size
        end_x = start_x + center_square_size
        mask_array[start_y:end_y, start_x:end_x] = 1
        mask_da.data = mask_array

        gdf = map_generator.create_shp_from_mask(mask_da)
        assert not gdf.empty, "GeoDataFrame should not be empty."
        assert len(gdf) == 1, "GeoDataFrame should contain exactly one polygon."

    def test_create_shp_with_no_building(self, map_generator: MapGenerator):
        image_files = get_files(
            map_generator.test_images_dir, extensions=[".tif", ".tiff"]
        )
        mask_da = rxr.open_rasterio(image_files[0]).isel(band=0)

        mask_da.name = "test_mask_none"
        # Create a binary mask with a central square of 1s
        size_y, size_x = (
            mask_da.shape[-2],
            mask_da.shape[-1],
        )
        center_square_size = 5
        mask_array = np.zeros((size_y, size_x), dtype=np.uint8)
        mask_da.data = mask_array

        gdf = map_generator.create_shp_from_mask(mask_da)
        assert gdf.empty, "GeoDataFrame should be empty."

    def test_generate_map_with_no_AOIs(self, map_generator):

        # Call the function under test
        map_generator.generate_map_from_images()
        image_files = get_files(
            map_generator.test_images_dir, extensions=[".tif", ".tiff"]
        )
        inference_files = get_files(
            map_generator.predictions_dir, extensions=[".tif", ".tiff"]
        )
        assert len(inference_files) == len(
            image_files
        ), "The number of predictions doesn't match the number of test images."

    def test_generate_map_with_no_overlap(self, boundaries_path, map_generator):

        # Call the function under test
        polygons_path = map_generator.project_dir / boundaries_path
        boundaries_gdf = gpd.read_file(polygons_path)

        # Directory where the image tiles are located
        image_tiles_dir = (
            map_generator.project_dir / map_generator.DIR_STRUCTURE["image_tiles"]
        )

        # New sub-directory for filtered images
        filtered_images_dir = map_generator.project_dir / "filtered_images"
        filtered_images_dir.mkdir(exist_ok=True)

        # Filter and copy images with "IMAGE_1024A" in their name to the new sub-directory
        for img in image_tiles_dir.iterdir():
            if "IMAGE_1024A" in img.name:
                shutil.copy(str(img), str(filtered_images_dir))

        with pytest.raises(FileNotFoundError) as errinfo:
            map_generator.generate_map_from_images(
                images_dir=filtered_images_dir,
                boundaries_gdf=boundaries_gdf,
                primary_key="id",
            )
        assert str(filtered_images_dir) in str(
            errinfo.value
        ), "This should raise an error and mention the tiles directory."

    def test_generate_map_with_overlap(self, boundaries_path, map_generator):

        # Call the function under test
        polygons_path = map_generator.project_dir / boundaries_path
        images_dir = (
            map_generator.project_dir / map_generator.DIR_STRUCTURE["image_tiles"]
        )
        boundaries_gdf = gpd.read_file(polygons_path)
        map_generator.generate_map_from_images(
            images_dir=images_dir, boundaries_gdf=boundaries_gdf, primary_key="id"
        )
        shapefiles = get_files(map_generator.shapefiles_dir, extensions=[".shp"])
        assert len(shapefiles) == 1, "Map generation should create a single shapefile"

    def test_generate_map_wrong_primary_key(self, boundaries_path, map_generator):

        # Call the function under test
        polygons_path = map_generator.project_dir / boundaries_path
        boundaries_gdf = gpd.read_file(polygons_path)
        wrong_key = "settlement_id"
        images_dir = (
            map_generator.project_dir / map_generator.DIR_STRUCTURE["image_tiles"]
        )
        with pytest.raises(KeyError) as excinfo:
            map_generator.generate_map_from_images(
                images_dir=images_dir,
                boundaries_gdf=boundaries_gdf,
                primary_key=wrong_key,
            )
        assert str(wrong_key) in str(
            excinfo.value
        ), "KeyError should specify the wrong key value."

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

        assert isinstance(prediction, xr.DataArray)
        assert prediction.shape == (
            tile_size,
            tile_size,
        )

        # Open the original image file to compare its metadata with the prediction
        with rxr.open_rasterio(image_files[0]) as src:
            # Check if the CRS of the prediction matches the image CRS
            assert (
                prediction.rio.crs == src.rio.crs
            ), "Prediction CRS does not match image CRS."

            # Check if the spatial extent of the prediction matches the image
            assert (
                src.rio.bounds() == prediction.rio.bounds()
            ), "Prediction bounds do not match image bounds."

    def test_single_inference_with_empty_boundaries(
        self, mock_config, map_generator: MapGenerator
    ):

        # Mocking the get_image_files and rxr.open_rasterio functions
        # You might need to adjust this part based on your actual implementation

        image_files = get_files(
            map_generator.test_images_dir, extensions=[".tif", ".tiff"]
        )

        prediction = map_generator.single_tile_inference(
            image_files[0],
            boundaries_gdf=gpd.GeoDataFrame(geometry=[]),
            write_shp=False,
        )

        tile_size = mock_config["config_value"]["tiling"]["tile_size"]

        assert isinstance(prediction, xr.DataArray)
        assert prediction.shape == (
            tile_size,
            tile_size,
        )

        # Open the original image file to compare its metadata with the prediction
        with rxr.open_rasterio(image_files[0]) as src:
            # Check if the CRS of the prediction matches the image CRS
            assert (
                prediction.rio.crs == src.rio.crs
            ), "Prediction CRS does not match image CRS."

            # Check if the spatial extent of the prediction matches the image
            assert (
                src.rio.bounds() == prediction.rio.bounds()
            ), "Prediction bounds do not match image bounds."

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
