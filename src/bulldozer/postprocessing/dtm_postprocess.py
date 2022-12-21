#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of Bulldozer
# (see https://github.com/CNES/bulldozer).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module is used to postprocess the DTM in order to improve its quality. It required a DTM generated from Bulldozer.
"""
import os
import time
import logging
import rasterio
from rasterio.fill import fillnodata
import numpy as np
import scipy.ndimage as ndimage
import bulldozer.springforce as sf
from rasterio.fill import fillnodata
from bulldozer.utils.helper import write_dataset, Runtime
from bulldozer.utils.logging_helper import BulldozerLogger
from bulldozer.scale.tools import scaleRun

def generateOutputProfileForPitsDetection(inputProfile: rasterio.DatasetReader.profile):
    """
        Only the dtype change 
    """
    outputProfile = inputProfile
    outputProfile["nodata"] = None
    outputProfile["dtype"] = np.ubyte
    return outputProfile

def detectPitsComputer(inputBuffers: list, params: dict) -> np.ndarray:
    """
    """

    dtm = inputBuffers[0]
    pits_mask = np.zeros(dtm.shape, dtype=np.ubyte)


    # Generates the low frenquency DTM
    bfilters = sf.PyBulldozerFilters()
    dtm_LF = ndimage.uniform_filter(dtm, size = params["filter_size"])
    # Retrieves the high frenquencies in the input DTM
    dtm_HF = dtm - dtm_LF

    # Tags the pits
    pits_mask[dtm_HF < 0.] = 1
    
    return pits_mask

@Runtime
def build_pits_mask(dtm_path : np.ndarray,
                    pits_mask_path: str,
                    nb_max_workers : int = 1) -> None:
    """
    This method detects pits in the input DTM.
    Those pits are generated during the DTM extraction by remaining artefacts.

    Args:
        dtm_path: path to the DTM extracted with bulldozer.
        nb_max_workers: number of availables workers (multiprocessing requirement).

    Returns:
        mask flagging the pits in the input DTM.
    """
    BulldozerLogger.log("Pits mask generation: Start", logging.DEBUG)

    with rasterio.open(dtm_path, 'r') as dataset:

        resolution = dataset.profile['transform'][0]
        pitsParams = {
            "filter_size": 35.5/resolution,
            "desc": "Pits detection"
        }

        scaleRun(inputImagePaths = [dtm_path],
                 outputImagePath = pits_mask_path,
                 algoComputer = detectPitsComputer,
                 algoParams = pitsParams,
                 generateOutputProfileComputer = generateOutputProfileForPitsDetection,
                 nbWorkers = nb_max_workers,
                 stableMargin = pitsParams["filter_size"],
                 inMemory=False)

        return None

def generateOuputProfileForFillPits(inputProfile: rasterio.DatasetReader.profile):
    """
        Only the dtype change 
    """
    return inputProfile

def fillPitsComputer(inputBuffers: list, params: dict) -> np.ndarray:
    """
    """

    # Generates the low frenquency DTM
    dtm = inputBuffers[0]
    pits_mask = inputBuffers[1]

    dtm_LF = ndimage.uniform_filter(dtm, size = params["filter_size"])
    #dtm[pits_mask] = dtm_LF
    dtm = np.where( (pits_mask) & (dtm != params["nodata"]), dtm_LF, dtm)

    return dtm

@Runtime
def fill_pits(raw_dtm_path : str,
              pits_mask_path : str, 
              out_dtm_path : str, 
              nb_max_workers : int,
              nodata: float) -> None:
    """
    This method fills the pits of the input raw DTM and writes the result in out_dtm_path.

    Args:
        raw_dtm_path: path to the input raw DTM.
        pits_mask: corresponding pits mask.
        out_dtm_path: path to the output filled DTM. If None, overrides the raw_dtm_path raster.
        nb_max_workers: number of availables workers (multiprocessing requirement).
    """

    with rasterio.open(raw_dtm_path, 'r') as dataset:

        resolution = dataset.profile['transform'][0]

        pitsParams = {
            "filter_size": 35.5/resolution,
            "desc": "Pits filling",
            "nodata": nodata
        }

        scaleRun(inputImagePaths = [raw_dtm_path, pits_mask_path], 
                 outputImagePath = out_dtm_path, 
                 algoComputer = fillPitsComputer, 
                 algoParams = pitsParams, 
                 generateOutputProfileComputer = generateOuputProfileForFillPits, 
                 nbWorkers = nb_max_workers, 
                 stableMargin = pitsParams["filter_size"],
                 inMemory=False)


def generateOuputProfileForBuildDhm(inputProfile: rasterio.DatasetReader.profile):
    """
        Only the dtype change 
    """
    return inputProfile

def buildDhmComputer(inputBuffers: list, params: dict) -> np.ndarray:
    """
    """
    # Generates the low frenquency DTM
    dtm = inputBuffers[0]
    dsm = inputBuffers[1]
    dhm = dsm - dtm
    dhm[dtm == params["nodata"]] == params["nodata"]
    return dhm

@Runtime
def build_dhm(dsm_path : str, 
              dtm_path : str, 
              dhm_path : str, 
              nb_max_workers : int = 1,
              nodata : float = None) -> None:
    """
    This method generates a Digital Height Model DHM (DSM-DTM) in the provided directory.
    The DSM and DTM must have the same metadata (shape, type, etc.).

    Args:
        dsm_path: path to the input DSM.
        dtm_path: path to the input DTM.
        output_dir: path to the output DHM file.
        nb_max_workers: number of availables workers (multiprocessing requirement).
    """
    BulldozerLogger.log("DHM generation: Start", logging.INFO)

    dhmParams = {
        "nodata": nodata,
        "desc": "Build Dhm"
    }

    scaleRun(inputImagePaths = [dtm_path, dsm_path], 
             outputImagePath = dhm_path, 
             algoComputer = buildDhmComputer, 
             algoParams = dhmParams, 
             generateOutputProfileComputer =  generateOuputProfileForBuildDhm, 
             nbWorkers = nb_max_workers, 
             stableMargin = 0,
             inMemory = False)

def compute_dhm(dsm_path : str, 
                dtm_path : str, 
                dhm_path : str, 
                window : rasterio.windows.Window,
                nodata : float = None) -> (np.ndarray, rasterio.windows.Window)  :
    """
    This method computes the Digital Height Model DHM (DSM-DTM) for a given window.

    Args:
        dsm_path: path to the input DSM.
        dtm_path: path to the input DTM.
        dhm_path: path to the output DHM.
        window: coordinates of the concerned window.

    Returns:
        the DHM values and the corresponding window.
    """
    with rasterio.open(dsm_path, 'r') as dsm_dataset:
        dsm_strip = dsm_dataset.read(1, window=window).astype(np.float32)
        with rasterio.open(dtm_path, 'r') as dtm_dataset:
            dtm_strip = dtm_dataset.read(1, window=window).astype(np.float32)
            dhm_strip = dsm_strip - dtm_strip
            dhm_strip[dsm_strip == nodata] == nodata
            return(dhm_strip, window)



def fillDtmComputer(inputBuffers: list, params: dict) -> np.ndarray:
    """
    """
    # Fill DTM
    dtm = inputBuffers[0]
    mask = np.where(dtm == params["nodata"], 0, 1).astype(np.ubyte)
    filled_dtm = fillnodata(dtm, mask=mask, max_search_distance=params["search_distance"])
    return filled_dtm

@Runtime
def buildIntermediateFilledDtm(dtm_path: str,
                               filled_dtm_path: str,
                               nodata: float,
                               nb_max_workers: int):

    BulldozerLogger().log("Fill intermediate DTM", logging.INFO)

    fillParams = {
        "nodata": nodata,
        "desc": "Fill DTM",
        "search_distance": 100
    }

    scaleRun(inputImagePaths = [dtm_path], 
             outputImagePath = filled_dtm_path, 
             algoComputer = fillDtmComputer, 
             algoParams = fillParams, 
             generateOutputProfileComputer = generateOuputProfileForBuildDhm, 
             nbWorkers = nb_max_workers, 
             stableMargin = fillParams["search_distance"],
             inMemory = False)

@Runtime
def postprocess_pipeline(raw_dtm_path : str, 
                         output_dir : str,
                         nb_max_workers : int = 1,
                         quality_mask_path : str = None, 
                         generate_dhm : bool = False,
                         dsm_path : str = None,
                         nodata : float = None) -> None:
    """
    Bulldozer postprocess pipeline. It removes remaining pits.
    It also generates the DHM and reproject the products if the options are set.

    Args:
        raw_dtm_path: path to the DTM generated with bulldozer (raw).
        output_dir: path to the output directory.
        nb_max_workers: number of availables workers (multiprocessing requirement).
        quality_mask_path: path to the quality mask associated with the DTM.
        generate_dhm: option that indicates if bulldozer has to generate the DHM (DSM-DTM).
        dsm_path: path to the input DSM. This argument is required for the DHM generation.
        output_CRS: if a CRS (different from the input DSM) is provided, reproject the DTM to the new CRS.
    """
    BulldozerLogger.log("Starting postprocess", logging.INFO)

    # Fill DTM nodata for future pits detection
    # To apply uniform filter you need to fill nodata value of the raw dtm
    # then detect pits on the filled dtm and use raw dtm to put back nodata value
    filled_dtm_path : str = os.path.join(output_dir, "filled_dtm.tif")
    buildIntermediateFilledDtm(dtm_path = raw_dtm_path,
                               filled_dtm_path = filled_dtm_path,
                               nodata = nodata,
                               nb_max_workers = nb_max_workers)

    # Detects the pits in the DTM (due to correlation issue)
    pits_mask_path: str = os.path.join(output_dir, 'pits.tif')
    build_pits_mask(filled_dtm_path, pits_mask_path, nb_max_workers)

    dtm_path = os.path.join(output_dir, 'DTM.tif')
    # Fills the detected pits
    #fill_pits(raw_dtm_path, pits_mask_path, dtm_path, nb_max_workers, nodata)
    fill_pits(filled_dtm_path, pits_mask_path, dtm_path, nb_max_workers, nodata)
            
    with rasterio.open(dtm_path, 'r') as dtm_dataset:
        # Updates the quality mask if it's provided
        if quality_mask_path:
            # Add a new band for the pits mask
            with rasterio.open(quality_mask_path, 'r') as q_mask_dataset:
                with rasterio.open(pits_mask_path, "r") as pits_mask_dataset:
                    pits_mask = pits_mask_dataset.read(indexes=1)
                    q_mask_profile = q_mask_dataset.profile
                    q_mask_profile['count'] = q_mask_profile['count'] + 1
                    write_dataset(quality_mask_path, pits_mask, q_mask_profile, band=q_mask_profile['count'])
                    os.remove(pits_mask_path)
        else:
            BulldozerLogger.log("No quality mask provided", logging.WARNING)
        # Generates the DHM (DSM - DTM) if the option is activated
        if generate_dhm and dsm_path:
            dhm_path = os.path.join(output_dir, "DHM.tif")
            build_dhm(dsm_path, dtm_path, dhm_path, nb_max_workers, nodata)