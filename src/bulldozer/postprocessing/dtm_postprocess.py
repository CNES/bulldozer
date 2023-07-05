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
import numpy as np
import scipy.ndimage as ndimage
import bulldozer.springforce as sf
from rasterio.fill import fillnodata
from bulldozer.scale.tools import scaleRun
from bulldozer.scale.Shared import Shared
from rasterio.warp import reproject, Resampling
from bulldozer.utils.logging_helper import BulldozerLogger
from bulldozer.utils.helper import write_dataset, Runtime, retrieve_raster_resolution, Pyramid, write_tiles, downsample_profile, retrieve_nodata

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

def checkIntersectionComputer(inputBuffers : list, params : dict) -> np.ndarray:
    """
    """
    dtm = inputBuffers[0]
    dsm = inputBuffers[1]
    # Check intersection
    dsm_valid = dsm != params["dsm_nodata"]
    dtm_valid = dtm != params["dtm_nodata"]
    valid = np.logical_or(dsm_valid, dtm_valid)
    np.minimum(dtm, dsm, out=dtm, where=valid)
    return dtm

@Runtime
def checkIntersection(dtm_path : str,
                      dsm_path : str, 
                      out_dtm_path : str, 
                      nb_max_workers : int,
                      nodata: float) -> None:
    """
    This method will check that the new DTM is under the raw DSM and writes the result in out_dtm_path.

    Args:
        dtm_path: path to the input DTM.
        dsm_path: path to the input raw DSM.
        out_dtm_path: path to the output filled DTM. If None, overrides the dtm_path raster.
        nb_max_workers: number of availables workers (multiprocessing requirement).
    """

    with rasterio.open(dtm_path, 'r') as dataset:

        resolution = dataset.profile['transform'][0]
        dsm_nodata = retrieve_nodata(dsm_path, nodata)
        intersectionParams = {
            "desc": "Intersection check",
            "dtm_nodata": nodata,
            "dsm_nodata": dsm_nodata
        }
        
        # We use the same profile as pits 
        scaleRun(inputImagePaths = [dtm_path, dsm_path], 
                 outputImagePath = out_dtm_path, 
                 algoComputer = checkIntersectionComputer, 
                 algoParams = intersectionParams, 
                 generateOutputProfileComputer = generateOuputProfileForFillPits, 
                 nbWorkers = nb_max_workers,
                 stableMargin = 0,
                 inMemory=False)


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
              out_dtm_path : str = None, 
              nb_max_workers : int = 1,
              nodata: float = None) -> None:
    """
    This method fills the pits of the input raw DTM and writes the result in out_dtm_path.

    Args:
        raw_dtm_path: path to the input raw DTM.
        pits_mask: corresponding pits mask.
        out_dtm_path: path to the output filled DTM. If None, overrides the raw_dtm_path raster.
        nb_max_workers: number of availables workers (multiprocessing requirement).
    """
    if out_dtm_path is None:
        out_dtm_path = raw_dtm_path

    if nodata is None:
        nodata = retrieve_nodata(raw_dtm_path) 

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
    dhm = np.where(dsm == params['dsm_nodata'], params['dtm_nodata'], dhm)
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
    BulldozerLogger.log("DHM generation: Start", logging.DEBUG)

    dsm_nodata = retrieve_nodata(dsm_path, nodata)

    dhmParams = {
        "dtm_nodata": nodata,
        "dsm_nodata": dsm_nodata,
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
def adaptToTargetResolution(raw_dtm_path: str,
                            dsm_path: str,
                            output_dir: str,
                            quality_mask_path: str):

    BulldozerLogger().log("Adapt to user resolution ", logging.INFO)

    with rasterio.open(raw_dtm_path) as raw_dtm_dataset:
        if "minLevel" in raw_dtm_dataset.tags():
            decimated_level: int = int(raw_dtm_dataset.tags()["minLevel"])
            if decimated_level > 0:
                with rasterio.open(dsm_path) as dsm_dataset:
                    # Need to decimate the dsm
                    dsm_pyramid = Pyramid(raster_path = dsm_path)
                    decimated_dsm = dsm_pyramid.getArrayAtLevel(level=decimated_level)
                    # Flush the decimated dsm to disk
                    decimated_dsm_path = os.path.join(output_dir, "decimated_dsm.tif")
                    write_tiles(tile_buffer = decimated_dsm, 
                                tile_path = decimated_dsm_path, 
                                original_profile = downsample_profile(dsm_dataset.profile, 2**decimated_level))

                if quality_mask_path:
                    # Need to reproject quality the mask
                    with rasterio.open(quality_mask_path) as full_mask_dataset:

                        dest_profile = full_mask_dataset.profile.copy()
                        dest_profile.update({
                        'transform': raw_dtm_dataset.transform,
                        'width': raw_dtm_dataset.width,
                        'height': raw_dtm_dataset.height
                        })

                        decimated_quality_mask_path = os.path.join(output_dir, "decimated_quality_mask.tif")

                        # apply with rasterio
                        with rasterio.open(decimated_quality_mask_path, 'w', **dest_profile) as dst:

                            reproject(
                                source=rasterio.band(full_mask_dataset, 1),
                                destination=rasterio.band(dst, 1),
                                dst_transform=raw_dtm_dataset.transform,
                                src_transform=full_mask_dataset.transform,
                                resampling= Resampling.min,
                                src_nodata=full_mask_dataset.nodata,
                                dst_nodata=full_mask_dataset.nodata,
                            )
                        
                        os.rename(src=decimated_quality_mask_path, 
                                dst=quality_mask_path)

                return decimated_dsm_path
            else:
                return dsm_path
        else:
            return dsm_path

@Runtime
def postprocess_pipeline(raw_dtm_path : str, 
                         output_dir : str,
                         nb_max_workers : int = 1,
                         quality_mask_path : str = None, 
                         generate_dhm : bool = False,
                         dsm_path : str = None,
                         check_intersection: bool = False,
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
    BulldozerLogger.log("Starting postprocess", logging.DEBUG)

    # We need to retrieve the resolution factor from the dtm
    # in the case the user gives an output resolution greater than the full
    # resolution
    if dsm_path is not None:
        dsm_path: str = adaptToTargetResolution(raw_dtm_path = raw_dtm_path,
                                                dsm_path = dsm_path,
                                                output_dir = output_dir,
                                                quality_mask_path = quality_mask_path)


    # Fill DTM nodata for future pits detection
    # To apply uniform filter you need to fill nodata value of the raw dtm
    # then detect pits on the filled dtm and use raw dtm to put back nodata value
    filled_dtm_path : str = os.path.join(output_dir, "filled_dtm.tif")
    buildIntermediateFilledDtm(dtm_path = raw_dtm_path,
                               filled_dtm_path = filled_dtm_path,
                               nodata = nodata,
                               nb_max_workers = nb_max_workers
                               )

    # Detects the pits in the DTM (due to correlation issue)
    pits_mask_path: str = os.path.join(output_dir, 'pits.tif')
    build_pits_mask(filled_dtm_path, pits_mask_path, nb_max_workers)

    dtm_path = os.path.join(output_dir, 'DTM.tif')
    # Fills the detected pits
    #fill_pits(raw_dtm_path, pits_mask_path, dtm_path, nb_max_workers, nodata)
    fill_pits(filled_dtm_path, pits_mask_path, dtm_path, nb_max_workers, nodata)


    if dsm_path and check_intersection:
        # check the interstion after the filling
        checkIntersection(dtm_path = filled_dtm_path,
                        dsm_path = dsm_path, 
                        out_dtm_path = dtm_path, 
                        nb_max_workers = nb_max_workers,
                        nodata = nodata)

    os.remove(filled_dtm_path)

    with rasterio.open(dtm_path, 'r') as dtm_dataset:
        # Updates the quality mask if it's provided
        if quality_mask_path:
            # Add a new band for the pits mask
            concat_mask = np.zeros((2, dtm_dataset.height, dtm_dataset.width), dtype=np.uint8)
            concat_profile = None
            with rasterio.open(quality_mask_path, 'r') as q_mask_dataset:
                with rasterio.open(pits_mask_path, "r") as pits_mask_dataset:
                    concat_mask[1,:,:] = pits_mask_dataset.read(indexes=1)
                    concat_mask[0,:,:] = q_mask_dataset.read(indexes=1)
                    concat_profile = q_mask_dataset.profile
                    os.remove(pits_mask_path)
            concat_profile['count'] = 2
            with rasterio.open(quality_mask_path, 'w', **concat_profile) as q_mask_dataset:
                q_mask_dataset.write(concat_mask)
        else:
            BulldozerLogger.log("No quality mask provided", logging.WARNING)
        # Generates the DHM (DSM - DTM) if the option is activated
        if generate_dhm and dsm_path:
            dhm_path = os.path.join(output_dir, "DHM.tif")
            build_dhm(dsm_path, dtm_path, dhm_path, nb_max_workers, nodata)
