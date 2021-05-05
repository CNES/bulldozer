# Copyright 2021 PIERRE LASSALLE
# All rights reserved

import logging
import logging.config
import platform
import psutil
import os
import getpass
import numpy as np
from danesfield_DTM import DTMEstimator
import rasterio
from rasterio.fill import fillnodata
import scipy.ndimage as ndimage
from tqdm import tqdm
from config_parser import ConfigParser
from datetime import datetime
from sys import stdout
from git import Repo
from shutil import copy

logging.config.fileConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/logging.ini"), disable_existing_loggers=False)
logger = logging.getLogger(__name__)

def init_logger():
    """
            This method initiates the log file in order to store the environment state
    """
    info={}
    try:
        # Git info
        try :
            repo = Repo(search_parent_directories=True)
            info['commit_sha'] = repo.head.object.hexsha
            info['branch'] = repo.active_branch
        except Exception as e:
            info['commit_sha'] = "No git repo found ({})".format(e)
            info['branch'] = "No git repo found ({})".format(e)
            
        # Node info
        info['user']=getpass.getuser()
        info['node']=platform.node()
        info['processor']=platform.processor()
        info['ram']=str(round(psutil.virtual_memory().total / (1024 **3)))+" GB"

        # OS info
        info['system']=platform.system()
        info['release']=platform.release()
        info['os_version']=platform.version()
        
        init = ("\n"+"#"*17+"\n#   BULLDOZER   #\n"+"#"*17+"\n# <Git info>\n#\t- branch: {}\n#\t- commit SHA: {}"
                "\n#\n# <Node info>\n#\t - user: {}\n#\t - node: {}\n#\t - processor: {}\n#\t - RAM: {}"
                "\n#\n# <OS info>\n#\t - system: {}\n#\t - release: {}\n#\t - version: {}\n"
                +"#"*17).format(info['branch'], info['commit_sha'], info['user'], info['node'], 
                                info['processor'], info['ram'], info['system'], info['release'], info['os_version'])
        logger.debug(init)
    except Exception as e:
        logger.warning("Init error: " + e)
        
def checkInputArray(bufferPath: str, inputBuffer: np.ndarray, outputPath: str, flushFlag: bool):
    
    dataset = None
    buffer : np.ndarray = None

    if flushFlag:
        assert(bufferPath is not None and outputPath is not None)
        dataset = rasterio.open(bufferPath)
        if inputBuffer is None:
            buffer = dataset.read(1)
        else:
            buffer = inputBuffer
    else:
        if inputBuffer is not None:
            buffer = inputBuffer
        else:
            assert(bufferPath is not None)
            dataset = rasterio.open(bufferPath)
            buffer = dataset.read(1)

    return dataset, buffer

def saveToRaster(inputModelPath: str, bufferToSave: np.ndarray, outputPath: str, noDataVal: float = -32768):

    inputModelDataset = rasterio.open(inputModelPath)

    outputDataset = rasterio.open(outputPath,
                                  "w",
                                  driver="GTiff",
                                  height=inputModelDataset.height,
                                  width=inputModelDataset.width,
                                  count=1,
                                  nodata=noDataVal,
                                  dtype=rasterio.dtypes.float32,
                                  crs=inputModelDataset.read_crs(),
                                  transform=inputModelDataset.transform)
    outputDataset.write(bufferToSave, 1)

def handleOutputBuffer(bufferPath: str, bufferToWrite: np.ndarray, outputPath: str, flushFlag: bool):
    if flushFlag:
        assert(bufferPath is not None and outputPath is not None)
        saveToRaster(bufferPath, bufferToWrite, outputPath)

class Bulldozer:

    def __init__(self):
        """ """
    
    def preprocess_FillCarsNoData(self, 
                                  dsmPathToFill: str=None, 
                                  dsmBufferToFill: np.ndarray = None, 
                                  outputFilledDsmPath: str=None,
                                  flushFlag:bool = True,
                                  maxSearchDistance=100.0,
                                  smoothingIterations=0,
                                  noDataValue=-32768.0,
                                  newValue=np.nan):
        """
            This methods uses raterio library to fill cars no data value with interpolation.
            This pre-processing is advised for using Bulldozer DTM extraction to minimize artefact results.

            @params:
                dsmBufferToFill (np.ndarray): array containing Dsm values with shape (height, width)

                maxSearchDistance (float): The maximum number of pixels to search in all directions to find values
                                           to interpolate from.
                
                smoothingIterations (int): The number of 3x3 smoothing filter passes to run.
                
                noDataValue (float) : the value of no data to fill in the DSM.
                
                newValue (float) : The value that replaces the noData values. If nan (default value), the method
                                     computes an interpolation for each noData point.
        """
        dsmDataset, dsmBuffer = checkInputArray(dsmPathToFill, dsmBufferToFill, outputFilledDsmPath, flushFlag)

        self.nodatamask = np.ma.masked_equal(dsmBuffer, noDataValue)
        
        if(np.isnan(newValue)):
            # if newValue is not set, than use the rasterio interpolation to fill the noDataValues
            dsmBuffer = fillnodata(dsmBuffer, mask=self.nodatamask, max_search_distance=maxSearchDistance,
                                   smoothing_iterations=smoothingIterations)
        else : 
            # otherwise replace noDataValues by newDataValues
            #REPLACE BY dsmBuffer[self.nodatamask.mask] = newValue?
            dsmBuffer[dsmBuffer == noDataValue] = newValue

        handleOutputBuffer(dsmPathToFill, dsmBuffer, outputFilledDsmPath, flushFlag)

    
    def preprocess_DetectDisturbedAreasAndFill(self,
                                               dsmPath: str=None,
                                               dsmBuffer: np.ndarray = None,
                                               outputCorrectedDsmPath: str = None,
                                               flushFlag: bool = True,
                                               slopeThreshold: float = 2.0,
                                               disturbedThreshold: int = 3,
                                               disturbedInfluenceDistance: float = 5.0,
                                               dsmResolution: float = 0.5,
                                               noDataValue: float = -32768):
        """
            This method detects disturbed areas in the DSM and use rasterio library to fill them by interpolation
            This pre-processing is advised when the area of interest contains homogeneous textures where the
            the CARS correlator can give chaotic results.

            @params:
                slopeThreshold (float): if the slope is greater than this threshold then we consider it as disturbed variation
                disturbedThreshold (int): If the number of successive disturbed pixels along a row is lower than this threshold then
                                          this sequence of pixels is considered as a disturbed area.
                disturbedInfluenceDistance (float): if the distance between 2 lists of disturbed cols is lower than this threshold expressed in meters 
                                                    then they are merged.
                dsmResolution (float): resolution of the input DSM in meters.
        """

        dsmDataset, dsmBuffer = checkInputArray(dsmPath, dsmBuffer, outputCorrectedDsmPath, flushFlag)

        # disturbedAreasZDL will contain row lists and each list will contains k_row lists of disturbed columns.
        disturbedAreasZDL = []

        # Loop over each row
        print("Step: Detecting local disturbed areas")
        for row in tqdm(range(dsmBuffer.shape[0])):
            disturbedAreasZDL.append([])
            col = 1
            nbDisturbedAreas = 0
            while col < dsmBuffer.shape[1]:
                slope = np.abs(dsmBuffer[row, col] - dsmBuffer[row, col-1])
                if slope >= slopeThreshold:
                    nbDisturbedAreas += 1
                    disturbedAreasZDL[row].append([])
                    disturbedAreasZDL[row][nbDisturbedAreas-1].append(col-1)
                    disturbedAreasZDL[row][nbDisturbedAreas-1].append(col)
                    col += 1
                    while col < dsmBuffer.shape[1]:
                        slope = np.abs(dsmBuffer[row, col] - dsmBuffer[row, col-1])
                        if slope >= slopeThreshold:
                            disturbedAreasZDL[row][nbDisturbedAreas-1].append(col)
                            col += 1
                        else:
                            col += 1
                            break
                else:
                    col += 1
        
        # mergedDisturbedAreas will contain row lists which will contain p_row lists with p_row <= k_row. The p_row lists
        # contain one or more disturbed cols from the k_row lists 
        mergedDisturbedAreas = []
        row = 0
        # Retrieve the distance influence in pixel unit
        distThreshold = disturbedInfluenceDistance / dsmResolution

        # Keep track of the maximum distance of a merged disturbed area
        maxDistance = 0

        print("Step: Merging local disturbed areas")
        for rowList in tqdm(disturbedAreasZDL):
            nbAreas = len(rowList)
            nbMerged = 0
            j = 0
            mergedDisturbedAreas.append([])
            while j < nbAreas:
                if len(rowList[j]) >= disturbedThreshold:
                    # We analyse if we found a group of disturbed areas
                    k = j + 1
                    nbConnected = 0
                    while k < nbAreas:
                        if rowList[k][0] - rowList[k-1][len(rowList[k-1])-1] <= distThreshold:
                            nbConnected += 1
                            k += 1
                        else:
                            break
                    
                    startCol = rowList[j][0]
                    endCol = rowList[j+nbConnected][len(rowList[j+nbConnected])-1]
                    maxDistance = max(maxDistance, endCol - startCol + 1)
                    mergedDisturbedAreas[row].append([])
                    for c in range(startCol, endCol + 1):
                        mergedDisturbedAreas[row][nbMerged].append(c)
                    nbMerged += 1
                    j += nbConnected + 1
                else:
                    j += 1

            row += 1
        
        # Fill with no-data value the col that belongs to merged disturbed areas
        print("Step: fill disturbed areas with no data")
        for row in tqdm(range(dsmBuffer.shape[0])):
            if len(mergedDisturbedAreas[row]) > 0:
                for disturbedArea in mergedDisturbedAreas[row]:
                    for col in disturbedArea:
                        dsmBuffer[row][col] = noDataValue
        
        print("Step: fill nodata by interpolation")
        # Use rasterio fill nodata interpolate
        self.disturbedAreaMask = np.ma.masked_equal(dsmBuffer, noDataValue)
        dsmBuffer = fillnodata(dsmBuffer, mask=self.disturbedAreaMask, max_search_distance=maxDistance)
        handleOutputBuffer(dsmPath, dsmBuffer, outputCorrectedDsmPath, flushFlag)

    
    def main_DtmExtractionWithCNESStrategy(self,
                                           dsmPath: str = None,
                                           dsmBuffer: np.ndarray = None,
                                           dtmPath: str=None,
                                           flushFlag: bool = True,
                                           maxObjectWidth: float = 100.0,
                                           dsmResolution: float = 0.5,
                                           numInnerIterationsStep1: int = 10,
                                           numOuterIterationsStep2: int = 100,
                                           numInnerIterationsStep2: int = 10,
                                           noDataValue: float = -32768):
        """
            Add a notion of max width for an object which is not a ground.
            From this max width results a dezoom resolution where we initialize the DTM at the dezoomed DSM
            At this level, we only apply ninner iterations to produce a dezoomed initial DTM
            This DTM is then used as input in the classical DrapCloth algorithm.
            The goal of this strategy is to ensure the drap not to separate from hills
        """

        dsmDataset, dsmBuffer = checkInputArray(dsmPath, dsmBuffer, dtmPath, flushFlag)

        nbObjectPixels = maxObjectWidth / dsmResolution

        # Determine the closest power of 2 of nbObjectPixels
        power = 0
        while 2**power < nbObjectPixels:
            power+=1
        
        if abs(2**(power-1) - nbObjectPixels) <  abs(2**(power) - nbObjectPixels):
            power -= 1

        logger.info("The dezoom factor is " + str(2**power) + " pixels")
        drapClothHandler =  DTMEstimator()

        logger.info("Building the pyramid...")
        dsmPyramids = []
        dsmPyramids.append(np.copy(dsmBuffer))
        for j in tqdm(range(1,power)):
            dsmPyramids.append(drapClothHandler.downsample(dsmPyramids[j-1]))

        pyramidSize = len(dsmPyramids)
        logger.info("Length of the pyramid " + str(pyramidSize))


        # Dtm is initialized at the most dezoomed dsm
        dtm = np.copy(dsmPyramids[pyramidSize-1])

        logger.info("Step 1: prevent unhook from the hills")
        # For the step, we only apply nInnerIterationsStep1.
        # This step allows to not unhook from the hills.
        for i in tqdm(range(numInnerIterationsStep1)):
            dtm = ndimage.uniform_filter(dtm, size=3)
        
        # Init classical parameters of drap cloth
        minv = np.min(dsmBuffer)
        maxv = np.max(dsmBuffer)
        step = (maxv - minv) / numOuterIterationsStep2
        max_level = power - 1
        level = max_level
        num_iter = numOuterIterationsStep2

        logger.info("Step 2: classical drap cloth filtering...")
        for j in tqdm(range(0, power-1)):

            logger.info("Process level " + str(j))
            logger.info("Dtm upsampling...")
            # DTM upsampling
            dtmNext = np.copy(dsmPyramids[pyramidSize-2-j])
            drapClothHandler.upsample(dtm, dtmNext)
            dtm = dtmNext

            logger.info("Filtering step...")
            for i in range(num_iter):
                dtm += step
                for i in range(numInnerIterationsStep2):
                    # handle DSM intersections, snap back to below DSM
                    np.minimum(dtm, dsmPyramids[pyramidSize-2-j], out=dtm)
                    # apply spring tension forces (blur the DTM)
                    dtm = ndimage.uniform_filter(dtm, size=3)

                # Final check intersection
                np.minimum(dtm, dsmPyramids[pyramidSize-2-j], out=dtm)

            step = step / (2 * 2 ** (max_level - level))
            # Decrease the number of iterations as well
            num_iter = max(1, int(numOuterIterationsStep2 / (2 ** (max_level - level))))
            level-=1

        #TODO merge nodatamask in the previous step to reduce memory print + add option to keep the interpolation
        dtm[self.nodatamask.mask] = noDataValue
        dtm[self.disturbedAreaMask.mask] = noDataValue
        handleOutputBuffer(dsmPath, dtm, dtmPath, flushFlag)

    
    def main_DtmExtractionWithThalesStrategy(self,
                                             dsmPath: str = None,
                                             dsmBuffer: np.ndarray = None,
                                             dtmPath: str=None,
                                             flushFlag: bool = True):
        """
            The strategy proposed by Thales is different but very interesting.
            A input cut resolution allows to determine the maximum dezoom. At this dezoom, with a given input step, the
            DTM is first initialized as the DSM and we apply nouter iterations nested with ninner iterations.
            Then we go back along the pyramid where at each level we just apply ninner iterations with intersection chech. No more gravity
            is applied.
        """
        
    
    def postprocess_RemoveSharpSinks(self,
                                    dtmPath: str = None,
                                    dsmBuffer: np.ndarray = None,
                                    outputCorrectedDtmPath: str = None,
                                    flushFlag: bool = True):
        """
            The extraction of DTM from photogrametric DSM can result in some sharp sinks (dark areas, local height artefacts).
            This methods uses a median filter / or sinks detection and remove them by interpolation.
        """



if __name__ == "__main__":
    init_logger()
    starting_time = datetime.now()
    logger.info("Starting time: " + starting_time.strftime("%Y-%m-%d %H:%M:%S"))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/configuration_template.yaml")
    parser = ConfigParser(verbose=True)

    cfg = parser.read(path)
    bulldozer = Bulldozer()

    bulldozer.preprocess_FillCarsNoData(dsmPathToFill=cfg['dsmPath'], 
                                        outputFilledDsmPath=cfg['outputFillDsmPath'],
                                        maxSearchDistance=cfg['maxSearchDistance'],
                                        smoothingIterations=cfg['smoothingIterations'],
                                        noDataValue=cfg['nodata'],
                                        newValue=float(cfg['new_value']))
    bulldozer.preprocess_DetectDisturbedAreasAndFill(dsmPath=cfg['outputFillDsmPath'],
                                        outputCorrectedDsmPath=cfg['outputCorrectedDsmPath'],
                                        slopeThreshold=cfg['slopeThreshold'],
                                        disturbedThreshold=cfg['disturbedThreshold'],
                                        disturbedInfluenceDistance=cfg['disturbedInfluenceDistance'],
                                        dsmResolution=cfg['dsmResolution'],
                                        noDataValue=cfg['nodata'])
                                        
    bulldozer.main_DtmExtractionWithCNESStrategy(dsmPath=cfg['outputCorrectedDsmPath'], 
                                        dtmPath=cfg['outputDtmPath'],
                                        maxObjectWidth=cfg['maxObjectWidth'],
                                        dsmResolution=cfg['dsmResolution'],
                                        numInnerIterationsStep1=cfg['numInnerIterationsStep1'],
                                        numOuterIterationsStep2=cfg['numOuterIterationsStep2'],
                                        numInnerIterationsStep2=cfg['numInnerIterationsStep2'],
                                        noDataValue=cfg['nodata'])
    logger.info("Ending time: {} (Runtime: {})".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.now()-starting_time))
    if cfg['keepLog']:
        try:
            copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs/bulldozer_logfile.log"),
                     os.path.dirname(os.path.abspath(cfg['outputDtmPath'])))
        except Exception as e:
            logger.warning("Error while writting the logfile: " + str(e), exc_info=True)
