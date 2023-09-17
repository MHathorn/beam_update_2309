__author__ = "Wassim Brahim"
__copyright__ = "Copyright 2022, UNITAC"
__email__ = "wassim.brahim@un.org"
__status__ = "Production"

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from os.path import join, exists
import uvicorn
from fastai.vision.all import *
import numpy as np
from os import listdir, makedirs, remove, stat
import glob
import shutil
from datetime import datetime
from time import sleep
import pathlib
from natsort import os_sorted
import rasterio
import rasterio.features
import shapely.geometry
import geopandas as gpd

from log_management import log

loaded_model = None
input_names = []
output_folder = ""
tile_dir = "./image_tiles"
tile_size = 1000
predicted_dir = "./predicted_images"
image_shape_x = None
image_shape_y = None
origins = ["http://localhost:8080"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def p2c(*_):
    pass


def get_msk(*_):
    pass


def get_y_augmented(*_):
    pass


@app.on_event("startup")
async def startup_event():
    """
    start up event for the whole process to check hardware checkup are proper.
    """
    log.info("GPU available: " + str(torch.cuda.is_available()))
    log.info(torch.cuda.device(0))


@app.get("/ping")
def ping_pong():
    """
    Function to help the frontend note when the backend is ready. It is a health check.
    """
    headers = {"Access-Control-Allow-Origin": "*"}
    return JSONResponse(
        content={"started": True}, status_code=status.HTTP_200_OK, headers=headers
    )


@app.post("/exit")
def exit_process():
    """
    function to exit the process clean
    """
    sleep(3000)
    os._exit(0)


@app.get("/uploadImages/")
def uploadImages(folder: str):
    """
    function to get images from frontend in a loop from the folder_path.
    """
    global input_names
    log.info("Images will imported from: " + folder)
    input_names = glob.glob(join(folder, "*.tif"))
    input_names.extend(glob.glob(join(folder, "*.tiff")))
    headers = {"Access-Control-Allow-Origin": "*"}
    try:
        if len(input_names) > 0:
            msg_log = "\n".join(input_names)
            log.info(f"Input Images are:\n{msg_log}")
            content = {"selectedImages": input_names}
            return JSONResponse(
                content=content, status_code=status.HTTP_200_OK, headers=headers
            )
        else:
            log.warn(f"No images found in {folder}")
            return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, headers=headers)
    except ValueError as err:
        log.warn(f"Unexpected {err=}, {type(err)=}")


@app.get("/loadModel/")
def load_model(model: str):
    """
    Loads the by default model, otherwise, loads the model is the parameter
    """
    global loaded_model
    # model = './models/exported-model.pkl' if not model else model
    log.info(f"The model is: {model}")
    try:
        loaded_model = load_learner(
            model,
            cpu=False if torch.cuda.is_available() else True,
            pickle_module=pickle,
        )
        log.info(f"Model {model} loaded successfully")
    except OSError as err:
        log.warn(f"Unexpected {err=}, {type(err)=}")
    content = {"selectedModel": model}
    headers = {"Access-Control-Allow-Origin": "*"}
    return JSONResponse(
        content=content, status_code=status.HTTP_200_OK, headers=headers
    )


@app.get("/loadOutputDir/")
def load_output(folder: str):
    """
    Loads the output folder on selection. The output folder is the one where we store the output shape files.
    """
    global output_folder
    log.info(f"The Shape files will be exported to: {output_folder}")
    output_folder = folder
    content = {"selectedOutput": output_folder}
    headers = {"Access-Control-Allow-Origin": "*"}
    return JSONResponse(
        content=content, status_code=status.HTTP_200_OK, headers=headers
    )


def create_tiles(image_path):
    """
    cuts the big image (from the path) into tiles with tile size as stated in the global variables. the images are
    squared.
    the tiles will be saved under /images_tiles/image_name/...png
    @todo verify if the tiling works with triangles images.
    """
    log.info(f"Start creating tiles for the image {image_path} ...")
    tilling_start_time = datetime.now()
    global tile_dir, loaded_model
    global tile_size
    global image_shape_x, image_shape_y
    if not exists(tile_dir):
        makedirs(tile_dir)
    # add condition to test the file size if it is 0 then just return?
    img = np.array(PILImage.create(image_path))
    image_shape_x, image_shape_y, _ = img.shape
    img_name = image_path.split("\\")[-1]
    if exists(join(tile_dir, img_name)):
        filelist = glob.glob(join(tile_dir, img_name, img_name + "*.png"))
        for f in filelist:
            remove(f)
    else:
        makedirs(join(tile_dir, img_name))
    # Cut tiles and save them
    for i in range(image_shape_x // tile_size):
        for j in range(image_shape_y // tile_size):
            img_tile = img[
                       i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size
                       ]
            Image.fromarray(img_tile).save(
                f"{join(tile_dir, img_name)}/{img_name}_000{i * (image_shape_x // tile_size) + j}.png"
            )
    log.info(f"Created tiles for image: {image_path}")
    tilling_end_time = datetime.now()
    log.info(f"Tiling time: {(tilling_end_time - tilling_start_time)}")


def get_image_tiles(nbr_tiles, img_name) -> L:
    """
    Returns a sorted list of the first `n` image tile filenames in `path`
    @todo precise here the usage of the function
    """
    global tile_dir
    files = L()
    files.extend(get_image_files(path=tile_dir, folders=img_name)[:nbr_tiles])
    return files


def save_predictions(image_path):
    """
    Saves the intermediary detected images from all tiles. Predicts all tiles of one scene and saves them to disk.
    """
    global loaded_model
    if not exists(predicted_dir):
        makedirs(predicted_dir)
    img_name = image_path.split("\\")[-1]
    # get all the tiles from the directory they belong to.
    tiles = os_sorted(get_image_tiles(len(listdir(join(tile_dir, img_name))), img_name))
    # @todo more documentation for how the prediction works
    for i in range(len(tiles)):
        pred, _, outputs = loaded_model.predict(tiles[i])
        output = torch.exp(pred[:, :]).detach().cpu().numpy()
        np.save(f"{predicted_dir}/saved_{i:02d}", output)
    outputs = os_sorted(glob.glob(f"{predicted_dir}/*.npy"))
    return len(tiles) == len(outputs)


def merge_tiles(arr, h, w):
    """
    combines all output detected tiles into the original shape of the image
    @todo add more documentation here
    """

    try:  # with color channel
        n, nrows, ncols, c = arr.shape
        return (
            arr.reshape(h // nrows, -1, nrows, ncols, c).swapaxes(1, 2).reshape(h, w, c)
        )
    except ValueError:  # without color channel
        n, nrows, ncols = arr.shape
        return arr.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)


def get_saved_predictions():
    """
    Load saved prediction mask tiles for a scene and return assembled mask
    """
    mask_tiles = os_sorted(glob.glob(f"{predicted_dir}/*.npy"))
    mask_array = list(map(np.load, mask_tiles))
    global image_shape_x, image_shape_y, tile_size
    # Remove hard coded values
    if mask_array is None or len(mask_array) == 0:
        return None
    try:
        mask_array = merge_tiles(
            np.array(mask_array),
            (image_shape_x // tile_size) * tile_size,
            (image_shape_y // tile_size) * tile_size,
        )
    except ValueError as err:
        print(f"Unexpected {err=}, {type(err)=}")

    return mask_array


def create_shp_from_mask(file, mask_array):
    """
    Transforms the image to a geo-encoded image
    todo add here the test if the image is a blank one, then return an empty shape file.
    """
    log.info(f"Start creating shape file from mask for the file {file}")
    global output_folder
    with rasterio.open(file, "r") as src:
        raster_meta = src.meta
    # create an empty shapefile and interrupt the function.
    if mask_array is None or len(mask_array) == 1:
        pred_name = file.split("\\")[-1]
        empty_schema = {"geometry": "Polygon", "properties": {"id": "int"}}
        no_crs = None
        gdf = gpd.GeoDataFrame(geometry=[])
        gdf.to_file(
            f"{output_folder}/{pred_name}_predicted.shp",
            driver="ESRI Shapefile",
            schema=empty_schema,
            crs=no_crs,
        )
        return
    mask_array = np.array(mask_array)
    shapes = rasterio.features.shapes(mask_array, transform=raster_meta["transform"])
    polygons = [
        shapely.geometry.Polygon(shape[0]["coordinates"][0]) for shape in shapes
    ]
    # Bug here with non-rectangular images.
    # Maybe the solution is to make all the images rectangular by adding white pixels
    my_list = raster_meta["crs"]
    gdf = gpd.GeoDataFrame(crs=my_list, geometry=polygons)
    gdf["area"] = gdf["geometry"].area
    # Drop shapes that are too small or too large to be a building
    gdf = gdf[(gdf["area"] > 2) & (gdf["area"] < 500000)]
    pred_name = file.split("\\")[-1]
    # in case the geo-dataframe is empty which means no settlements are detected
    if gdf.empty:
        empty_schema = {"geometry": "Polygon", "properties": {"id": "int"}}
        no_crs = None
        gdf = gpd.GeoDataFrame(geometry=[])
        gdf.to_file(
            f"{output_folder}/{pred_name}_predicted.shp",
            driver="ESRI Shapefile",
            schema=empty_schema,
            crs=no_crs,
        )
    else:
        gdf.to_file(
            f"{output_folder}/{pred_name}_predicted.shp", driver="ESRI Shapefile"
        )


#  I should check here the current stage with randomly shapes images

# start detection on the image tiles
@app.get("/startInference/")
def create_inferences(file: str):
    headers = {"Access-Control-Allow-Origin": "*"}
    process_start = datetime.now()
    global image_shape_x, image_shape_y
    inf_start = datetime.now()
    if (os.stat(file).st_size == 0):
        log.warn(f"The image {file} is empty and no tiles will be created.")
        pred_name = file.split("\\")[-1]
        empty_schema = {"geometry": "Polygon", "properties": {"id": "int"}}
        no_crs = None
        gdf = gpd.GeoDataFrame(geometry=[])
        gdf.to_file(
            f"{output_folder}/{pred_name}_predicted.shp",
            driver="ESRI Shapefile",
            schema=empty_schema,
            crs=no_crs,
        )

        content = {"message": "An empty Shape file for the empty image is stored."}
        return JSONResponse(
            content=content, status_code=status.HTTP_200_OK, headers=headers
        )

    create_tiles(file)
    saved_pred = save_predictions(file)
    inf_finish = datetime.now()
    log.info(f"Inference time: {(inf_finish - inf_start)}")
    img_name = file.split("\\")[-1]
    if not saved_pred:
        content = {
            "message": "The tiles and outputs have mismatched, please start the segmenting process again!"
        }
        shutil.rmtree(join(tile_dir, img_name))
        filelist = glob.glob(join(predicted_dir, "*.npy"))
        for f in filelist:
            os.remove(f)
        return JSONResponse(
            content=content, status_code=status.HTTP_409_CONFLICT, headers=headers
        )

    else:
        create_shp_from_mask(file, get_saved_predictions())
        content = {"message": "Output saved in the Output Folder."}

        shutil.rmtree(join(tile_dir, img_name))
        filelist = glob.glob(join(predicted_dir, "*.npy"))
        for f in filelist:
            os.remove(f)
        process_finish = datetime.now()
        log.info(f"Total Time: {(process_finish - process_start)}")
        return JSONResponse(
            content=content, status_code=status.HTTP_200_OK, headers=headers
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8005, log_level="info", reload=False)
