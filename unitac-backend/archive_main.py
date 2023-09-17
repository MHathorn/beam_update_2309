__author__ = "Chiranjit Roy"
__copyright__ = "Copyright 2022, UNITAC"
__email__ = "chiranjit.roy@modis.com"
__status__ = "Production"

# git push test

from email import header
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastai.vision.all import *
import pandas as pd
import numpy as np
import os
import glob
import shutil
from datetime import datetime
import pathlib
from natsort import os_sorted
import rasterio
import rasterio.features
import sys
# from semtorch import get_segmentation_learner

import shapely.geometry
import geopandas as gpd


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def p2c(*_): pass
def get_msk(*_): pass
def get_y_augmented(*_): pass


loadedModel = None
inputNames = []
outputFolder = ""
tileDir = "./image_tiles"
tileSize = 1000
predictedDir = "./predicted_images"
imageShapeX = None
imageShapeY = None

app = FastAPI()


origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#start up event for the whole process to check hardware checkup are proper
@app.on_event("startup")
async def startup_event():
    print("GPU available: " + str(torch.cuda.is_available()))
    print(torch.cuda.device(0))

# function to help teh frontend note when the backend is ready.. like a health check
@app.get("/ping")
def ping_pong():
    headers = {'Access-Control-Allow-Origin': '*'}
    return JSONResponse(content={'started': True}, status_code=status.HTTP_200_OK, headers=headers)


# function to exit the process clean
@app.post("/exit")
def exit_process():
    os._exit(0)



# function to get images from frontend in a loop
@app.get("/uploadImages/")
def uploadImages(folder: str):
    global inputNames
    print("Input Folder is: " + folder)
    inputNames = glob.glob(os.path.join(folder, "*.tif"))
    inputNames.extend(glob.glob(os.path.join(folder, "*.tiff")))
    headers = {'Access-Control-Allow-Origin': '*'}
    try:
        if len(inputNames) > 0:
            print("Input Images are:" + inputNames[0])
            content = {
                "selectedImages": inputNames
            }
            return JSONResponse(content=content, status_code=status.HTTP_200_OK, headers=headers)
        else:
            print("No relevant Images")
            return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, headers=headers)
    except ValueError as err:
        print(f"Unexpected {err=}, {type(err)=}")
    

# loads teh model by deafault and otherwise
@app.get("/loadModel/")
def loadModel(model: str):
    global loadedModel
    print(model)
    #model = './models/exported-model.pkl' if not model else model
    try:
        loadedModel = load_learner(model, cpu=False if torch.cuda.is_available() else True, pickle_module=pickle)
    except OSError as err:
        print(f"Unexpected {err=}, {type(err)=}")
    content = {
        "selectedModel": model
    }
    headers = {'Access-Control-Allow-Origin': '*'}
    return JSONResponse(content=content, status_code=status.HTTP_200_OK, headers=headers)

# load where teh final final shoud be saved
@app.get("/loadOutputDir/")
def loadOutput(folder: str):
    global outputFolder
    print(folder)
    outputFolder = folder

    content = {
        "selectedOutput": outputFolder
    }
    headers = {'Access-Control-Allow-Origin': '*'}
    return JSONResponse(content=content, status_code=status.HTTP_200_OK, headers=headers)

# cuts the big images into tiles with tile size as stated in the global variables
def createTiles(file):
    print("Creating tile ...")
    tile_start = datetime.now()
    global tileDir, loadedModel
    global tileSize
    global imageShapeX, imageShapeY
    if not os.path.exists(tileDir):
        os.makedirs(tileDir)
    img = np.array(PILImage.create(file))
    imageShapeX, imageShapeY, _ = img.shape
    img_name = file.split('\\')[-1]
    if os.path.exists(os.path.join(tileDir, img_name)):
        filelist = glob.glob(os.path.join(tileDir, img_name, img_name+"*.png"))
        for f in filelist:
            os.remove(f)
    else:
        os.makedirs(os.path.join(tileDir, img_name))
    # Cut tiles and save them
    for i in range(imageShapeX // tileSize):
        for j in range(imageShapeY // tileSize):
            imgTile = img[i * tileSize:(i + 1) * tileSize,
                          j * tileSize:(j + 1) * tileSize]
            Image.fromarray(imgTile).save(
                f'{os.path.join(tileDir,img_name)}/{img_name}_000{i*(imageShapeX//tileSize) + j}.png'
            )
    print(f'Created tiles for image: {file}')
    tile_end = datetime.now()
    print(f"Tiling time: {(tile_end - tile_start)}")
    
# returns a sorted list of all tiles
def getImageTiles(nTiles, img_name) -> L:
    "Returns a list of the first `n` image tile filenames in `path`"
    global tileDir
    files = L()
    files.extend(get_image_files(
        path=tileDir, folders=img_name)[:nTiles])
    return files

# saves teh intermediatry detected images from all tiles
def savePredictions(file):
    global loadedModel, codes
    "Predicts all tiles of one scene and saves them to disk"
    if not os.path.exists(predictedDir):
        os.makedirs(predictedDir)
    img_name = file.split('\\')[-1]
    tiles = os_sorted(getImageTiles(
        len(os.listdir(os.path.join(tileDir, img_name))), img_name))

    ############# Uncomment for using get_preds ########################################
    # dls = SegmentationDataLoaders.from_label_func(Path(os.path.join(tileDir, img_name)), tiles, label_func=None, bs=5, device= torch.device('cuda'))
    # dls = SegmentationDataLoaders.from_label_func(Path(os.path.join(tileDir, img_name)), tiles, bs=1, label_func=None)
    # preds = loadedModel.get_preds(dl=dls.train)
    # predicted_classes = preds[0].argmax(axis = 1)
    # return predicted_classes
    ############# Uncomment for using get_preds ########################################

    ############# Comment this section if using get_preds ####################
    for i in range(len(tiles)):
        pred, _, outputs = loadedModel.predict(tiles[i])
        output = torch.exp(pred[:, :]).detach().cpu().numpy()
        np.save(f'{predictedDir}/saved_{i:02d}', output)
    outputs = os_sorted(glob.glob(f'{predictedDir}/*.npy'))
    if (len(tiles) != len(outputs)):
        return False
    else:
        return True
    ############# Comment this section if using get_preds ####################

# combine all output detected tiles into the original shape of teh image
def unblockShape(arr, h, w):

    try:  # with color channel
        n, nrows, ncols, c = arr.shape
        return (arr.reshape(h // nrows, -1, nrows, ncols,
                            c).swapaxes(1, 2).reshape(h, w, c))
    except ValueError:  # without color channel
        n, nrows, ncols = arr.shape
        return (arr.reshape(h // nrows, -1, nrows,
                            ncols).swapaxes(1, 2).reshape(h, w))


############# Uncomment this section if using get_preds ####################
#def getSavedPreds(preds):
############# Uncomment this section if using get_preds ####################
############# Comment this section if using get_preds ####################
def getSavedPreds():
############# Comment this section if using get_preds ####################

    "Load saved prediction mask tiles for a scene and return assembled mask"
    ############# Comment this section if using get_preds ####################
    maskTiles = os_sorted(glob.glob(f'{predictedDir}/*.npy'))
    maskArrs = list(map(np.load, maskTiles))
    ############# Comment this section if using get_preds ####################
    global imageShapeX, imageShapeY, tileSize
    # Remove hard coded values
    if maskArrs is None or len(maskArrs) ==0 :
        return None
    try:
        maskArray = unblockShape(np.array(
            maskArrs), (imageShapeX // tileSize)*tileSize, (imageShapeY // tileSize)*tileSize)
    except ValueError as err:
        print(f"Unexpected {err=}, {type(err)=}")
    ############# Uncomment for using get_preds ########################################
    # maskArray = unblockShape(preds, (imageShapeX // tileSize)*tileSize, (imageShapeY // tileSize)*tileSize)
    ############# Uncomment for using get_preds ########################################
    
    return maskArray

# transform the image to a geo encoded image
def createShpFromMask(file, maskArray):
    #todo fix the issue with non-rectongular shapes and images with no settlements.
    print(file)
    global outputFolder
    with rasterio.open(file,
                       "r") as src:
        rasterMeta = src.meta
    # create an empty shapefile and interrupt the function.
    if maskArray is None or len(maskArray) == 1:
        pred_name = file.split('\\')[-1]
        empty_schema = {"geometry": "Polygon", "properties": {"id": "int"}}
        no_crs = None
        gdf = gpd.GeoDataFrame(geometry=[])
        gdf.to_file(f'{outputFolder}/{pred_name}_predicted.shp', driver='ESRI Shapefile', schema=empty_schema,
                    crs=no_crs)
        return
    shapes = rasterio.features.shapes(maskArray,
                                      transform=rasterMeta["transform"])
    polygons = [
        shapely.geometry.Polygon(shape[0]["coordinates"][0])
        for shape in shapes
    ]
    gdf = gpd.GeoDataFrame(crs=rasterMeta["crs"], geometry=polygons)
    gdf["area"] = gdf['geometry'].area
    # Drop shapes that are too small or too large to be a building
    gdf = gdf[(gdf["area"] > 2) & (gdf["area"] < 500000)]
    pred_name = file.split('\\')[-1]
    if gdf.empty:
        empty_schema = {"geometry": "Polygon", "properties": {"id": "int"}}
        no_crs = None
        gdf = gpd.GeoDataFrame(geometry=[])
        gdf.to_file(f'{outputFolder}/{pred_name}_predicted.shp', driver='ESRI Shapefile', schema=empty_schema,
                    crs=no_crs)
    else:
        gdf.to_file(f'{outputFolder}/{pred_name}_predicted.shp',
                    driver='ESRI Shapefile')

# start detection on the image tiles
@app.get("/startInference/")
def createInferences(file: str):
    headers = {'Access-Control-Allow-Origin': '*'}
    process_start = datetime.now()
    global imageShapeX, imageShapeY
    inf_start = datetime.now()
    createTiles(file)
    ############# Uncomment for using get_preds ########################################
    # savePred = True
    # preds = savePredictions(file)
    ############# Uncomment for using get_preds ########################################

    ############# Comment for using get_preds ########################################
    savePred = savePredictions(file)
    ############# Comment for using get_preds ########################################


    
    inf_finish = datetime.now()
    print(f"Inference Time: {(inf_finish - inf_start)}")
    img_name = file.split('\\')[-1]
    if savePred == False:
        content = {
            "message": "The tiles and ouputs have mismatched, please start the segmenting process again!"
        }
        shutil.rmtree(os.path.join(tileDir, img_name))
        filelist = glob.glob(os.path.join(predictedDir, '*.npy'))
        for f in filelist:
            os.remove(f)
        # shutil.rmtree(predictedDir)
        return JSONResponse(content=content, status_code=status.HTTP_409_CONFLICT, headers=headers)

    else:
        ############# Uncomment for using get_preds ########################################
        # createShpFromMask(file, getSavedPreds(preds))
        ############# Uncomment for using get_preds ########################################


        ############# Comment for using get_preds ########################################
        createShpFromMask(file, getSavedPreds())
        ############# Comment for using get_preds ########################################
        
        content = {
            "message": "Output saved in the Output Folder."
        }

        shutil.rmtree(os.path.join(tileDir, img_name))
        filelist = glob.glob(os.path.join(predictedDir, '*.npy'))
        for f in filelist:
            os.remove(f)
        # shutil.rmtree(predictedDir)
        process_finish = datetime.now()
        print(f"Total Time: {(process_finish - process_start)}")
        return JSONResponse(content=content, status_code=status.HTTP_200_OK, headers=headers)


if __name__ == '__main__':
    uvicorn.run("main:app", host='localhost', port=8005,
                log_level="info", reload=False)
