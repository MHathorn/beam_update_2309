import logging
import gzip
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

from cv2 import erode
from cv2 import dilate
from rasterio.features import rasterize
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.affinity import scale
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from utils.base_class import BaseClass
from utils.helpers import seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataTiler(BaseClass):
    """
    Data Tiler class that processes geospatial imagery and labels into tiles.

    This class is responsible for loading images and optional label data, generating
    mask and weight tiles if required, and saving these tiles for further processing.

    Attributes:
        project_dir (Path): The project directory containing all training files.
        crs (CRS): Coordinate Reference System of the input images.
        spatial_resolution (float): Spatial resolution of the images.
        tiling_params (dict): Parameters related to tiling such as distance weighting,
                              whether to tile labels, erosion, and tile size.

    Methods:
        _load_tiling_params: Loads tiling parameters from the configuration.
        _load_images: Loads GeoTIFF images from a specified directory.
        _load_labels: Loads building footprints from vector files into a GeoDataFrame.
        _crop_labels: Crops labels based on the bounding box of an input image.
        _write_da_to_raster: Writes a DataArray to a raster file in TIFF format.
        generate_mask: Generates a binary mask from vector labels.
        generate_tiles: Tiles images and masks, and stores them as TIFF files.

    Usage:
        img_tiler = DataTiler(project_dir, config_name)
        img_tiler.generate_tiles(tile_size)

    Expected output:
        Tiles saved in sub-directories `image_tiles`, `mask_tiles` (if labels provided),
        and `weight_tiles` (if distance weighting is enabled).
    """

    def __init__(self, project_dir, config_name=None):
        """
        Initializes the DataTiler with the relevant configuration settings.

        Args:
            project_dir : str
                Path to the project directory, containing images, config file and labels (if training project).
            config_name : str
                The configuration file name. If missing, the constructor will look for a single file in the project directory.
        """

        super().__init__(project_dir, config_name)
        self.tiling_params = self._load_tiling_params(self.config)
        seed(self.config["seed"])

        self.crs = None
        self.spatial_resolution = None
        write_dirs = ["image_tiles"]

        images_dir = self.project_dir / self.DIR_STRUCTURE["images"]
        dsm_dir = None
        if not images_dir.exists():
            raise IOError(
                "The directory path `images` does not point to an existing directory in `project_dir`."
            )

        if self.config["tiling"].get("dsm", False):
            dsm_dir = self.project_dir / self.DIR_STRUCTURE.get("dsm", "dsm")
            if not dsm_dir.exists():
                raise IOError(
                    "The directory path `dsm` does not point to an existing directory in `project_dir`."
                )

        self.images_generator = self._load_images(images_dir, dsm_dir)

        labels_dir = self.project_dir / self.DIR_STRUCTURE["labels"]
        valid_label_paths = [
            l
            for l in labels_dir.glob("*")
            if l.suffix in [".csv", ".shp"] or l.name.endswith(".csv.gz")
        ]

        if not valid_label_paths:
            self.labels = None
            logging.warning(
                "No labels file provided. Tiling images alone."
                if not labels_dir.exists() or len(list(labels_dir.iterdir())) == 0
                else "Label files are not in recognized format (shp, csv). Tiling images alone."
            )
        else:
            write_dirs += ["mask_tiles"]
            if self.tiling_params["distance_weighting"]:
                write_dirs.append("weight_tiles")
            self.labels = self._load_labels(valid_label_paths)

        super().load_dir_structure(write_dirs=write_dirs)

    def _load_tiling_params(self, config):
        """
        Loads tiling parameters from the configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing tiling parameters.

        Returns:
            dict: Dictionary containing tiling parameters such as distance weighting,
                  whether to tile labels, erosion, and tile size.
        """
        tiling_keys = ["distance_weighting", "erosion", "tile_size", "dsm"]
        return {k: config["tiling"].get(k) for k in tiling_keys}

    def _load_images(self, image_dir, dsm_dir):
            """
            Loads all GeoTIFF images in the provided directory using rioxarray.

            Args:
                image_dir (Path): Directory containing GeoTIFF images.
                dsm_dir (Path): Directory containing DSM (Digital Surface Model) files.

            Yields:
                DataArray: Single band from a GeoTIFF image as an xarray DataArray.

            Raises:
                IOError: If the directory does not contain any GeoTIFF images.
                ValueError: If the filename format is unexpected.
                IOError: If the DSM file is not found for an image.
            """
            rgb_files = [
                img_path
                for img_path in image_dir.rglob("*")
                if img_path.suffix.lower() in [".tif", ".tiff"]
            ]

            if not rgb_files:
                raise IOError(
                    f"The directory {image_dir} does not contain any GeoTIFF images."
                )
            self.n_images = len(rgb_files)

            for img_path in rgb_files:
                image = rxr.open_rasterio(img_path, default_name=img_path.stem)

                # Check if image is not uint8 and convert if necessary
                if image.dtype != np.uint8:
                    logging.info(f"Converting {img_path.name} to uint8...")
                                # Check if clipping is necessary
                    min_val = image.min().item()
                    max_val = image.max().item()
                    
                    if min_val < 0 or max_val > 255:
                        logging.warning(f"Image {img_path.name} contains values outside the 0-255 range. "
                                        f"Min: {min_val}, Max: {max_val}. Clipping will be performed.")
                        image = image.clip(0, 255)
                    
                    image = image.astype(np.uint8)


                if self.config["tiling"].get("dsm", False):
                    # Extract the relevant part of the filename to match DSM naming convention
                    parts = img_path.stem.split('_')
                    if len(parts) < 2:
                        raise ValueError(f"Unexpected filename format: {img_path.stem}")
                    dsm_name = f"{parts[-2]}_{parts[-1]}_elev.tif"
                    dsm_path = dsm_dir / dsm_name
                    if not dsm_path.exists():
                        raise IOError(f"DSM file {dsm_path} not found for image {img_path}")
                    dsm = rxr.open_rasterio(dsm_path, default_name=dsm_name)
                    dsm = dsm.rio.reproject_match(image)
                    image = xr.concat([image, dsm], dim="band")
                yield image

    def _load_labels(self, labels_files):
        """
        Crops labels based on the bounding box of the input image.

        Args:
            image (DataArray): The input image whose bounding box will be used to crop the labels.

        Returns:
            GeoDataFrame: A GeoDataFrame cropped to the image boundaries.
        """

        def _load_from_gob(csv_path):
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
            logging.info(f"Loading label file {labels_path.name}..")
            if labels_path.suffix.lower() == ".csv":
                df = _load_from_gob(labels_path)

            elif labels_path.name.endswith(".csv.gz"):
                # Unzip the .gz file and read it as csv
                with gzip.open(labels_path, "rt") as f:
                    df = _load_from_gob(f)

            # Load from shapefile
            elif labels_path.suffix.lower() == ".shp":
                df = gpd.read_file(labels_path)

            buildings = pd.concat([buildings, df])

        logging.info("Deduplicating..")
        buildings = buildings.drop_duplicates()

        logging.info(f"Loaded {len(buildings)} labels for the imaged region.")
        return buildings

    def _crop_labels(self, image):
        """
        Crops labels based on the bounding box of the input image.

        Args:
            image (DataArray): The input image whose bounding box will be used to crop the labels.

        Returns:
            GeoDataFrame: A GeoDataFrame cropped to the image boundaries.
        """

        bounding_box = box(*image.rio.bounds())
        union_bounding_box = unary_union(bounding_box)

        return self.labels[self.labels.intersects(union_bounding_box)]

    def _write_da_to_raster(self, data, name, directory):
        """
        Writes a DataArray to a raster file in TIFF format.

        Args:
            data (DataArray): The data array to write to file.
            name (str): The name of the output file.
            directory (Path): The directory where the output file will be saved.
        """
        data_path = directory / name
        data.rio.to_raster(data_path)

    def generate_mask(self, image, labels, write=False, shrink_factor=0.995):
        """
        Generates a binary mask from vector labels and optionally writes it to disk.

        Polygons are created from the vector labels, optionally shrunk, and then rasterized.
        The mask is eroded with a 3x3 kernel if erosion is enabled. Distance weights are calculated
        if distance weighting is enabled.

        Args:
            image (DataArray): The image associated with the labels.
            labels (GeoDataFrame): The labels used to generate the mask.
            write (bool): If True, the generated mask and weights are written to disk.
            shrink_factor (float): Factor by which to shrink the polygons.

        Returns:
            tuple: A tuple containing the mask and weights as DataArrays.
        """
    
        # Function to shrink polygons
        def _shrink_polygon(polygon, shrink_factor):
            if polygon.is_empty:
                return polygon
            elif polygon.geom_type == "MultiPolygon":
                return MultiPolygon([scale(geom, xfact=shrink_factor, yfact=shrink_factor, origin='centroid') for geom in polygon.geoms])
            elif polygon.geom_type == "Polygon":
                return scale(polygon, xfact=shrink_factor, yfact=shrink_factor, origin='centroid')
            else:
                raise TypeError("Invalid geometry type")

        # Generate the mask
        def _poly_from_utm(polygon, transform):
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
        def _create_data_array(data, transform, image):
            
            if data.ndim == 3:
                data_da = xr.DataArray(
                    data,
                    dims=["band", "y", "x"],
                    coords={
                        "band": ["building", "edge"],
                        "x": image.coords["x"],
                        "y": image.coords["y"]
                    },
                )
            else:
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

        # Avoid SettingWithCopyError by creating a copy of the labels DataFrame
        labels_copy = labels.copy()
        
        # Shrink the polygons before rasterizing
        labels_copy.loc[:, 'geometry'] = labels_copy['geometry'].apply(_shrink_polygon, shrink_factor=shrink_factor)
    
        building_polygons = sum(
            labels_copy["geometry"].apply(_poly_from_utm, args=(transform,)), []
        )  # converting all to lists of polygons, then concatenating.
        edge_polygons = [poly.boundary for poly in building_polygons]
        building_mask = np.full(image_size, 0, dtype=np.uint8)
        weights = np.full(image_size, 0, dtype=np.uint8)
        edge_mask = np.full(image_size, 0, dtype=np.uint8)

        if len(building_polygons) > 0:
            building_mask = rasterize(
                shapes=building_polygons,
                out_shape=image_size,
                default_value=255,
                dtype="uint8",
            )
            edge_mask = rasterize(
            shapes=edge_polygons,
            out_shape=image_size,
            default_value=255,
            dtype="uint8",
            )

            # Eroding masks, as proposed in https://arxiv.org/abs/2107.12283
            if self.tiling_params["erosion"]:
                kernel = np.ones((3, 3), np.uint8)
                building_mask = erode(building_mask, kernel, iterations=1)
                edge_mask = dilate(edge_mask, kernel, iterations=1)

            # Generating gaussian weights, as proposed in https://arxiv.org/abs/2107.12283
            if self.tiling_params["distance_weighting"]:
                edge_polygons = [poly.boundary for poly in building_polygons]
                weights = rasterize(
                    shapes=edge_polygons,
                    out_shape=image_size,
                    default_value=0,
                    dtype="float32",
                )
                weights = gaussian_filter(weights, sigma=0.5)
                weights = weights / weights.max()

            #ensure that building mask and edge mask are not overlapping
            building_mask[edge_mask == 255] = 0
            # Combine into a three-class mask
            combined_mask = np.stack([building_mask, edge_mask], axis=0)

        mask_da = _create_data_array(combined_mask, transform, image)
        weights_da = _create_data_array(weights, transform, image)

        if write:
            tmp_dir = BaseClass.create_if_not_exists(
                self.project_dir / "tmp", overwrite=True
            )
            self._write_da_to_raster(mask_da, f"{image.name}_mask.tif", tmp_dir)
            if self.tiling_params["distance_weighting"]:
                self._write_da_to_raster(weights_da, f"{image.name}_edges.tif", tmp_dir)

        return mask_da, weights_da

    def generate_tiles(self, tile_size=0, write_tmp_files=False):
        """
        Tiles both images and masks (if any) and stores them as TIFF files.

        The tiled images are saved in the 'image_tiles' directory and the tiled masks
        (if any) are saved in the 'mask_tiles' directory. Optionally, weight tiles are
        saved in the 'weight_tiles' directory if distance weighting is enabled.

        Args:
            tile_size (int): Size of the output tiles. If 0, uses the size from tiling_params.
            write_tmp_files (bool): If True, intermediate mask files are stored before tiling.
        """

        if tile_size == 0:
            tile_size = self.tiling_params["tile_size"]

        for idx, image in enumerate(self.images_generator):
            logging.info(f"Tiling image {image.name}.. ({idx+1}/{self.n_images})")
            # Load image and corresponding mask as numpy array and retrieve their shape

            # Fix CRS with first image.
            logging.info("Preparing inputs..")
            if self.crs is None:
                self.crs = image.rio.crs
                if self.labels is not None and self.labels.crs != self.crs:
                    self.labels = self.labels.to_crs(self.crs)
            elif image.rio.crs != self.crs:
                image = image.rio.reproject(self.crs)

            # Prepare labels for training.
            if self.labels is not None:
                labels = self._crop_labels(image)
                if labels.empty:
                    logging.warning(
                        f"No intersecting labels found for {image.name}. Skipping image."
                    )
                    continue
                mask, weights = self.generate_mask(image, labels, write_tmp_files)

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
                    self._write_da_to_raster(img_tile, tile_name, self.image_tiles_dir)

                    if self.labels is not None:
                        msk_tile = mask.isel(
                            x=slice(i * tile_size, (i + 1) * tile_size),
                            y=slice(j * tile_size, (j + 1) * tile_size),
                        )
                        self._write_da_to_raster(
                            msk_tile, tile_name, self.mask_tiles_dir
                        )

                        if self.tiling_params["distance_weighting"]:
                            weights_tile = weights.isel(
                                x=slice(i * tile_size, (i + 1) * tile_size),
                                y=slice(j * tile_size, (j + 1) * tile_size),
                            )
                            self._write_da_to_raster(
                                weights_tile, tile_name, self.weight_tiles_dir
                            )

                    pbar.update(1)
            pbar.close()
