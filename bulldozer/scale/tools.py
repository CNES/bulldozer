from collections import namedtuple
from typing import Callable
import numpy as np
import rasterio
import concurrent.futures
from tqdm import tqdm
from bulldozer.scale.Shared import Shared, dictToRasterioProfile, getNumpyArrayShapeFromProfile

Tile = namedtuple('Tile', ["startX", "startY", "endX", "endY", "topM", "rightM", "bottomM", "leftM"])

def computeTiles(rasterHeight: float,
                 rasterWidth: float,
                 stableMargin: int,
                 nbProcs: int):
    """
        Given a square tile size and a stable margin,
        this method will compute a list of tiles covering
        all the given input image
    """
    tiles = []

    stripHeight: int = int(rasterHeight / nbProcs)
    nbTilesY : int = int(rasterHeight / stripHeight)
    nbTilesY = nbTilesY if rasterHeight % stripHeight == 0 else nbTilesY + 1
    
    for ty in range(nbTilesY):
        
        # Loop over those numbers and create all the tiles     
        # Determine the stable and unstable boundaries of the tile
        startX = 0
        startY = ty * stripHeight
        endX = int(rasterWidth) - 1
        endY = min((ty+1)* stripHeight - 1, int(rasterHeight) - 1)
        topM = stableMargin if startY - stableMargin >= 0 else startY
        leftM = 0
        bottomM = stableMargin if endY + stableMargin <= rasterHeight - 1 else int(rasterHeight) - 1 - endY
        rightM = 0
        
        tiles.append(Tile(startX=startX,
                          startY=startY,
                          endX=endX,
                          endY=endY,
                          topM = topM,
                          rightM = rightM,
                          bottomM = bottomM,
                          leftM = leftM))
        
    return tiles

def runNImgToImgAlgo(algoComputer: Callable,
                    algoParams: dict,
                    inputImagePaths: list,
                    tile: Tile):
    
    # Convert the tile to a rasterio window
    col_off = tile.startX - tile.leftM
    row_off = tile.startY - tile.topM
    width = tile.endX + tile.rightM - col_off + 1
    height = tile.endY + tile.bottomM - row_off + 1
    window = rasterio.windows.Window(col_off = col_off,
                                     row_off = row_off,
                                     width = width,
                                     height = height)
    

    inputBuffers = []
    for imgPath in inputImagePaths:
        if Shared.is_shared_memory_path(imgPath) :
            sh = Shared()
            sh.open(imgPath)
            array = sh.getArray()
            extract=array[row_off:row_off+height, col_off:col_off+width].copy()
            sh.close()
            inputBuffers.append(extract)
            
        else:
            inputBuffers.append(rasterio.open(imgPath).read(window=window, indexes=1))
    

    # Image to image filter, the user callable function must return
    # an output buffer
    outputBuffer = algoComputer(inputBuffers, algoParams)

    # The output window describes the stable area
    outputWindow = rasterio.windows.Window(col_off = tile.startX,
                                            row_off = tile.startY,
                                            width   = tile.endX - tile.startX + 1,
                                            height  = tile.endY - tile.startY + 1)
        
    # Extract the stable area
    stableStartX = int(tile.leftM)
    stableStartY = int(tile.topM)
    stableEndX = int(stableStartX + tile.endX - tile.startX + 1)
    stableEndY = int(stableStartY + tile.endY - tile.startY + 1)

    outputBuffer = outputBuffer[stableStartY:stableEndY, stableStartX:stableEndX]


    # send to master for writing
    return outputBuffer, outputWindow

def scaleRunDebug(inputImagePaths: list,
                  outputImagePath: str,
                  algoComputer: Callable,
                  algoParams: dict,
                  generateOutputProfileComputer: Callable,
                  nbWorkers: int,
                  maxMemory: float,
                  stableMargin: int,
                  inMemory: bool = True) -> np.ndarray:
    
    """
        Memory aware multiprocessing execution
    """
    
    with rasterio.open(inputImagePaths[0], "r") as inputImgDataset:

        # Generate the tiles
        tiles = computeTiles(rasterHeight = inputImgDataset.height,
                            rasterWidth = inputImgDataset.width,
                            stableMargin = stableMargin,
                            nbProcs = nbWorkers,
                            maxMemory=maxMemory)

        # Generate the output profile
        outputProfile = generateOutputProfileComputer(inputImgDataset.profile)

        wholeOutputArray = np.zeros((outputProfile["height"], 
                                        outputProfile["width"]), dtype=outputProfile["dtype"])

        for tile in tqdm(tiles, desc=algoParams['desc']):

            outputImgBuffer, outputWindow = runNImgToImgAlgo(algoComputer, algoParams, inputImagePaths, tile)
            wholeOutputArray[outputWindow.row_off: outputWindow.row_off + outputWindow.height, outputWindow.col_off: outputWindow.col_off + outputWindow.width] = outputImgBuffer[:]
        
        return wholeOutputArray

def scaleRun(inputImagePaths: list,
             outputImagePath: str,
             algoComputer: Callable,
             algoParams: dict,
             generateOutputProfileComputer: Callable,
             nbWorkers: int,
             stableMargin: int,
             inMemory: bool = True) -> np.ndarray:
    
    """
        Memory aware multiprocessing execution
    """

        
    if Shared.is_shared_memory_path(inputImagePaths[0]) :
        sh = Shared()
        sh.open(inputImagePaths[0])
        inputProfile = sh.metadata['profile']
        shape = getNumpyArrayShapeFromProfile(inputProfile)
        dtype = inputProfile['dtype']
        inputProfile = dictToRasterioProfile(inputProfile)
    
    else : 
        with rasterio.open(inputImagePaths[0], "r") as inputImgDataset:
            shape = (inputImgDataset.height, inputImgDataset.width)
            if inputImgDataset.count > 1:
                shape = (inputImgDataset.count, inputImgDataset.height, inputImgDataset.width)
            dtype = inputImgDataset.dtypes[0]
            inputProfile = inputImgDataset.profile.copy()
    

    # Generate the tiles
    tiles = computeTiles(rasterHeight = inputProfile['height'],
                        rasterWidth = inputProfile['width'],
                        stableMargin = stableMargin,
                        nbProcs = nbWorkers)


    if inMemory:
        wholeOutputArray = np.zeros(shape, dtype=dtype)
        with concurrent.futures.ProcessPoolExecutor(max_workers = nbWorkers) as executor:

            futures = {executor.submit(runNImgToImgAlgo,
                                       algoComputer,
                                       algoParams,
                                       inputImagePaths,
                                       tile) for tile in tiles}
                
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=algoParams['desc']):

                outputImgBuffer, outputWindow = future.result()
                wholeOutputArray[outputWindow.row_off: outputWindow.row_off + outputWindow.height, outputWindow.col_off: outputWindow.col_off + outputWindow.width] = outputImgBuffer[:]
        return wholeOutputArray
               
        
    else:
        # Generate the output profile
        outputProfile = generateOutputProfileComputer(inputProfile)

        with rasterio.open(outputImagePath, "w", **outputProfile) as outputImgDataset:


            with concurrent.futures.ProcessPoolExecutor(max_workers = nbWorkers) as executor:

                futures = {executor.submit(runNImgToImgAlgo,
                                           algoComputer,
                                           algoParams,
                                           inputImagePaths,
                                           tile) for tile in tiles}
                    
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=algoParams['desc']):

                    outputImgBuffer, outputWindow = future.result()

                    outputImgDataset.write(outputImgBuffer, window=outputWindow, indexes=1)
            
        return None

