import logging
import gzip
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from cv2 import erode
from rasterio.features import rasterize
from shapely import wkt
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from utils.base_class import BaseClass


class DataTiler(BaseClass):
    """Data Tiler class. This class loads from the configuration file an image directory,
    tile size, and tiles directory, and populates the tile directory with image
    tiles ready for data augmentation and train-test split.

    Expected configuration attributes:
    ├── input_dir
    │ ├── images // A list of files in GeoTIFF format.
    │ │ ├── image.tiff
    │ ├── labels // A shapefile or csv file (in Google Open Buildings Dataset format) containing all labels
                    for the included images.
    │ │ ├── label.shp (optional)
    If labels are presents in the directry, mask tiles will be created as well.


    Expected configuration attributes:
    - root_dir: Name of the project directory containing all training files.
    - tile_size: Size of the output tiles.
    - erosion: true if erosion should be applied to the labels.

    Usage:
    img_tiler = DataTiler(config)
    img_tiler.generate_tiles(tile_size)

    Expected output:
    Tiles saved in new sub-directory `image_tiles` and `mask_tiles` (if labels file provided).
    """

    def __init__(self, config):

        self.root_dir = Path(config["root_dir"])
        self.crs = None
        self.spatial_resolution = None
        write_dirs = ["image_tiles"]

        self.tiling_params = self.load_tiling_params(config)

        # Checking for images and loading in DataArrays
        images_dir = self.root_dir / self.DIR_STRUCTURE["images"]
        if not images_dir.exists():
            raise IOError(
                "The directory path `images` does not point to an existing directry in `root_dir`."
            )
        self.images_generator = self.load_images(images_dir)

        # Checking for masks and loading if exist
        labels_dir = self.root_dir / self.DIR_STRUCTURE["labels"]
        valid_label_paths = [
            l
            for l in labels_dir.glob("*")
            if l.suffix in [".csv", ".shp"] or l.name.endswith(".csv.gz")
        ]

        if not valid_label_paths:
            self.labels = None
            print(
                "No labels file provided. Tiling images alone."
                if not labels_dir.exists() or len(list(labels_dir.iterdir())) == 0
                else "Warning: Label files are not in recognized format (shp, csv). Tiling images alone."
            )
        else:
            write_dirs += ["mask_tiles", "label_tiles"]
            if self.tiling_params["distance_weighting"]:
                write_dirs.append("weight_tiles")
            # Loading labels from csv / shapefile.
            self.labels = self.load_labels(valid_label_paths)

        super().__init__(config, write_dirs=write_dirs)

    def load_tiling_params(self, config):
        tiling_keys = ["distance_weighting", "tile_labels", "erosion", "tile_size"]
        return {k: config["tiling"].get(k) for k in tiling_keys}

    def load_images(self, image_dir):
        """
        Loads all GeoTIFF images in the provided image_dir with rioxarray.

        Parameters:
        image_dir: Images directory.
        Yields:
        xarray.DataArray: Single band from a GeoTIFF image.
        """

        filepaths = [
            img_path
            for img_path in image_dir.rglob("*")
            if img_path.suffix.lower() in [".tif", ".tiff"]
        ]

        if not filepaths:
            raise IOError(
                f"The directory {image_dir} does not contain any GeoTIFF images."
            )
        self.n_images = len(filepaths)

        for img_path in filepaths:
            yield rxr.open_rasterio(img_path, default_name=img_path.stem)

    def load_labels(self, labels_files):
        """
        This loads building footprints from one or more vector files
        and stores them as an instance attribute.

        Parameters:
        labels_files: Path to labels file, in .csv, .shp or .csv.gz format.
        crop: True if labels datafame should be adapted to the size of the images dataset.

        Returns:
        GeoDataFrame: A geo dataframe containing all building labels.
        """

        def load_from_gob(csv_path):
            """
            Loading function from Google Open Buildings dataset.
            Expected format: CSV.
            Ref: https://sites.research.google/open-buildings/
            """
            df = pd.read_csv(csv_path)
            df["geometry"] = df["geometry"].apply(wkt.loads)
            return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        buildings = pd.DataFrame()

        for labels_path in labels_files:
            print(f"Loading label file {labels_path.name}..")
            if labels_path.suffix.lower() == ".csv":
                df = load_from_gob(labels_path)

            elif labels_path.name.endswith(".csv.gz"):
                # Unzip the .gz file and read it as csv
                with gzip.open(labels_path, "rt") as f:
                    df = load_from_gob(f)

            # Load from shapefile
            elif labels_path.suffix.lower() == ".shp":
                df = gpd.read_file(labels_path)

            buildings = pd.concat([buildings, df])

        print("Deduplicating..")
        buildings = buildings.drop_duplicates()

        print(f"Loaded {len(buildings)} labels for the imaged region.")
        return buildings

    def crop_labels(self, image):
        """
        This method crops labels based on the bounding box of the input image.

        Parameters:
        image: The input image whose bounding box will be used to crop the labels.

        Returns:
        GeoDataFrame: GeoDataFrame cropped to the image boundaries.
        """

        bounding_box = box(*image.rio.bounds())
        union_bounding_box = unary_union(bounding_box)

        return self.labels[self.labels.intersects(union_bounding_box)]

    def write_da_to_raster(self, data, name, directory):
        """Write data array to raster file in tiff format."""
        data_path = directory / name
        data.rio.to_raster(data_path)

    def generate_mask(self, image, labels, write=False):
        """
        Generate a binary mask from a vector file (shp or geojson).

        Polygons are created from the vector labels. The mask is then created and
        eroded with a 3x3 kernel.

        Returns:
        numpy.ndarray: The generated mask, if 'write' is False. Otherwise, None.
        """

        # Generate the mask
        def poly_from_utm(polygon, transform):
            if polygon.is_empty:
                return []
            elif polygon.geom_type == "MultiPolygon":
                return [
                    Polygon(
                        [~transform * tuple(i) for i in np.array(geom.exterior.coords)]
                    )
                    for geom in polygon.geoms
                ]
            elif polygon.geom_type == "Polygon":
                return [
                    Polygon(
                        [
                            ~transform * tuple(i)
                            for i in np.array(polygon.exterior.coords)
                        ]
                    )
                ]
            else:
                raise TypeError("Invalid geometry type")

        # Generate data array
        def create_data_array(data, transform, image):
            data_da = xr.DataArray(
                data,
                dims=["y", "x"],
                coords={"x": image.coords["x"], "y": image.coords["y"]},
            )
            data_da = data_da.rio.write_crs(self.crs)
            data_da = data_da.rio.write_transform(transform)

            return data_da

        image_size = (image.shape[1], image.shape[2])
        transform = image.rio.transform()

        # if self.tiling_params["erosion"]:
        #     labels['geometry'] = labels['geometry'].buffer(-spatial_resolution * 1)

        label_polygons = sum(
            labels["geometry"].apply(poly_from_utm, args=(transform,)), []
        )  # converting all to lists of polygons, then concatenating.
        mask = np.full(image_size, 0, dtype=np.uint8)
        weights = np.full(image_size, 0, dtype=np.uint8)

        if len(label_polygons) > 0:
            mask = rasterize(
                shapes=label_polygons,
                out_shape=image_size,
                default_value=255,
                dtype="uint8",
            )

            if self.tiling_params["erosion"]:
                kernel = np.ones((3, 3), np.uint8)
                mask = erode(mask, kernel, iterations=1)

            if self.tiling_params["distance_weighting"]:
                edge_polygons = [poly.boundary for poly in label_polygons]
                weights = rasterize(
                    shapes=edge_polygons,
                    out_shape=image_size,
                    default_value=255,
                    dtype="uint8",
                )
                weights = gaussian_filter(weights, sigma=0.5) * 200

        mask_da = create_data_array(mask, transform, image)
        weights_da = create_data_array(weights, transform, image)

        if write:
            tmp_dir = BaseClass.create_if_not_exists(
                self.root_dir / "tmp", overwrite=True
            )
            self.write_da_to_raster(mask_da, f"{image.name}_mask.tif", tmp_dir)
            if self.tiling_params["distance_weighting"]:
                self.write_da_to_raster(weights_da, f"{image.name}_edges.tif", tmp_dir)

        return mask_da, weights_da

    def save_tile_shapefile(self, labels, tile_geom, shp_name):
        "Save a clipped version of self.labels containing only the polygons of a given tile."
        clipped_labels = gpd.clip(labels, tile_geom)
        clipped_labels = clipped_labels[clipped_labels.geometry.type == "Polygon"]
        # Save clipped labels as a shapefile
        clipped_labels_path = self.label_tiles_dir / shp_name
        clipped_labels.to_file(clipped_labels_path)

    def generate_tiles(self, tile_size=0, write_tmp_files=False):
        """
        This method tiles both images and masks (if any) and stores them as .tif files.

        The tiled images are saved in the 'image_tiles' directory and the tiled masks (if any) are saved in the 'mask_tiles' directory.
        The naming convention for the tiled images and masks is '{original_file_name}_r{row_index}_c{column_index}.png'.

        Parameters:
        tile_size: Size of the output tiles.
        write_tmp_files: True if the mask should be stored before tiling.

        """

        if tile_size == 0:
            tile_size = self.tiling_params["tile_size"]

        for idx, image in enumerate(self.images_generator):
            print(f"Tiling image {image.name}.. ({idx+1}/{self.n_images})")
            # Load image and corresponding mask as numpy array and retrieve their shape

            # Fix CRS with first image.
            print("Preparing inputs..")
            if self.crs is None:
                self.crs = image.rio.crs
                if self.labels and self.labels.crs != self.crs:
                    self.labels = self.labels.to_crs(self.crs)
            elif image.rio.crs != self.crs:
                image = image.rio.reproject(self.crs)

            # Prepare labels for training.
            if self.labels is not None:
                labels = self.crop_labels(image)
                if not labels.empty:
                    mask, weights = self.generate_mask(image, labels, write_tmp_files)
                else:
                    print(
                        f"No intersecting labels found for {image.name}. Skipping image."
                    )
                    continue

            x_tiles = image.sizes["x"] // tile_size
            y_tiles = image.sizes["y"] // tile_size
            total_tiles = x_tiles * y_tiles

            if total_tiles == 0:
                raise IOError(
                    f"tile_size is bigger than the input image for {image.name} \
                    ({image.sizes['x']}, {image.sizes['y']}). \
                              Please choose a smaller tile size or a different image."
                )

            pbar = tqdm(total=total_tiles, desc=f"Tiling {image.name}", leave=True)

            # Cut image. mask and weights into tiles and store them as .tif-files
            for i in range(x_tiles):
                for j in range(y_tiles):

                    tile_name = f"{image.name}_r{j}_c{i}.TIF"
                    img_tile = image.isel(
                        x=slice(i * tile_size, (i + 1) * tile_size),
                        y=slice(j * tile_size, (j + 1) * tile_size),
                    )
                    tile_geom = box(*img_tile.rio.bounds())
                    self.write_da_to_raster(img_tile, tile_name, self.image_tiles_dir)

                    if self.labels is not None:
                        msk_tile = mask.isel(
                            x=slice(i * tile_size, (i + 1) * tile_size),
                            y=slice(j * tile_size, (j + 1) * tile_size),
                        )
                        self.write_da_to_raster(
                            msk_tile, tile_name, self.mask_tiles_dir
                        )

                        if self.tiling_params["distance_weighting"]:
                            weights_tile = weights.isel(
                                x=slice(i * tile_size, (i + 1) * tile_size),
                                y=slice(j * tile_size, (j + 1) * tile_size),
                            )
                            self.write_da_to_raster(
                                weights_tile, tile_name, self.weight_tiles_dir
                            )

                        # Save labels in the appropriate folder.
                        if self.tiling_params["tile_labels"]:
                            self.save_tile_shapefile(labels, tile_geom, tile_name)
                    pbar.update(1)
            pbar.close()
