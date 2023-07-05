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
    This module is used to preprocess the DSM in order to improve the DTM computation.
"""

import rasterio
from rasterio.fill import fillnodata
import os
import numpy as np
import logging
import bulldozer.bordernodata as bn
import bulldozer.disturbedareas as da
from bulldozer.utils.helper import write_dataset
from bulldozer.utils.logging_helper import BulldozerLogger
from bulldozer.scale.tools import scaleRun
from bulldozer.scale.Shared import Shared
from bulldozer.utils.helper import Runtime, retrieve_nodata, DefaultValues


def generate_identical_profile(input_profile: rasterio.DatasetReader.profile) -> dict:
    return input_profile

def generate_output_profile_for_mask(input_profile: rasterio.DatasetReader.profile) -> dict:
    output_profile = input_profile
    output_profile['dtype'] = np.ubyte
    return output_profile

def border_nodata_computer( inputBuffers: list,
                            params: dict) -> np.ndarray:
    """ 
    This method computes the border nodata mask in a given window of the input DSM.

    Args:
        inputBuffers: contain just one DSM buffer.
        params:  dictionnary containing:
            nodata value: DSM potentially custom nodata 
            doTranspose: boolean flag to computer either horizontally or vertically the border no data.
    Returns:
        mask flagging the border nodata areas
    """
    dsm = inputBuffers[0]
    nodata = params['nodata']

    if np.isnan(nodata):
        dsm = np.nan_to_num(dsm, False, nan=DefaultValues['NODATA'])
        nodata = DefaultValues['NODATA']
    
    # We're using our C++ implementation to perform this computation
    border_nodata = bn.PyBorderNodata()

    if params["doTranspose"]:
        return border_nodata.build_border_nodata_mask(dsm, nodata).astype(np.ubyte)
    else:
        # Vertical border nodata detection case (axis = 1)
        dsm = dsm.T
        border_nodata_mask = border_nodata.build_border_nodata_mask(dsm, nodata).astype(np.ubyte)
        return border_nodata_mask.T
    
@Runtime
def build_border_nodata_mask(dsm_path : str, 
                             nb_max_workers : int = 1,
                             nodata : float = None) -> np.ndarray:
    """
    This method builds a mask corresponding to the border nodata values.
    Those areas correpond to the nodata points on the edges if the DSM is skewed.

    Args:
        dsm_path: path to the input DSM.
        nb_max_workers: number of availables workers (multiprocessing).
        nodata: nodata value of the input DSM. If None, retrieve this value from the input DSM metadata.

    Returns:
        border nodata boolean masks.
    """

    borderNoDataParams: dict = {
        'doTranspose': False,
        'nodata': nodata,
        'desc': "Build Horizontal Border NoData Mask"
    }

    horizontal_border_nodata = scaleRun(inputImagePaths = [dsm_path], 
                                        outputImagePath = None, 
                                        algoComputer = border_nodata_computer, 
                                        algoParams = borderNoDataParams, 
                                        generateOutputProfileComputer = generate_output_profile_for_mask, 
                                        nbWorkers = nb_max_workers, 
                                        stableMargin = 0, 
                                        inMemory=True)
    
    
    with Shared.make_shared_from_numpy(horizontal_border_nodata) as shared_horizontal_border_nodata:
        horizontal_border_nodata = None
        
        borderNoDataParams['doTranspose'] = True
        borderNoDataParams['desc'] = "Build Vertical Border NoData Mask"
        vertical_border_nodata = scaleRun(inputImagePaths = [dsm_path], 
                                            outputImagePath = None, 
                                            algoComputer = border_nodata_computer, 
                                            algoParams = borderNoDataParams, 
                                            generateOutputProfileComputer = generate_output_profile_for_mask, 
                                            nbWorkers = nb_max_workers, 
                                            stableMargin = 0, 
                                            inMemory=True)
        
        # Merges the two masks
        mask =  np.logical_and(shared_horizontal_border_nodata.getArray(), vertical_border_nodata)
        vertical_border_nodata = None
    
    
    return mask


def disturbedAreasComputer(inputBuffers: list, params: dict) -> np.ndarray:
    """
        This method computes the disturbance in a DSM window through the horizontal axis.
        It returns the corresping disturbance mask.

        Args:
            inputBuffers: contain just one DSM buffer.
            params:  dictionnary containing:
                nodata value: DSM potentially custom nodata 
                slope_treshold: if the slope is greater than this threshold then we consider it as disturbed variation.
                is_four_connexity: number of evaluated axis.
        Returns:
            mask flagging the disturbed areas. 
    """

    nodata = params["nodata"]

    if np.isnan(nodata):
        dsm = np.nan_to_num(dsm, False, nan=DefaultValues['NODATA'])
        nodata = DefaultValues['NODATA']

    disturbed_areas = da.PyDisturbedAreas(params["is_four_connexity"])
    disturbance_mask = disturbed_areas.build_disturbance_mask(inputBuffers[0], 
                                                              params["slope_threshold"],
                                                              params["nodata"]).astype(np.ubyte)
    return disturbance_mask


def build_disturbance_mask(dsm_path: str,
                           nb_max_workers : int = 1,
                           slope_threshold: float = 2.0,
                           is_four_connexity : bool = True,
                           nodata: float = None) -> np.array:
    """
    This method builds a mask corresponding to the disturbed areas in a given DSM.
    Most of those areas correspond to water or correlation issues during the DSM generation (obstruction, etc.).

    Args:
        dsm_path: path to the input DSM.
        nb_max_workers: number of availables workers (multiprocessing).
        slope_treshold: if the slope is greater than this threshold then we consider it as disturbed variation.
        is_four_connexity: number of evaluated axis. 
                        Vertical and horizontal if true else vertical, horizontal and diagonals.

    Returns:
        masks containing the disturbed areas.
    """
    if nodata is None:
        nodata = retrieve_nodata(dsm_path)

    #TODO assert not none c++ param raise Value Error (missing value for parameters)
    disturbanceParams = {
        "is_four_connexity": is_four_connexity,
        "slope_threshold": slope_threshold,
        "nodata": nodata,
        "desc": "Build Disturbance Mask"
    }

    disturbance_mask = scaleRun(inputImagePaths = [dsm_path], 
                                outputImagePath = None,
                                algoComputer = disturbedAreasComputer, 
                                algoParams = disturbanceParams, 
                                generateOutputProfileComputer = generate_output_profile_for_mask, 
                                nbWorkers = nb_max_workers,
                                stableMargin = 1,
                                inMemory=True)
    return disturbance_mask != 0
    
def fillNoDataComputer(inputBuffers: list,
                       params: dict) -> np.ndarray:
    
    return fillnodata(inputBuffers[0], mask=inputBuffers[1], max_search_distance=100.0, smoothing_iterations=0)


def multiProcsFillNoData(inputImagePaths: list,
                         outputPath: str,
                         nb_max_workers: int = 1) -> None:
    
    fillNoDataParams = {
        "desc": "Filling DSM nodata..."
    }

    filledDSM = scaleRun(inputImagePaths = inputImagePaths, 
                         outputImagePath = outputPath, 
                         algoComputer = fillNoDataComputer, 
                         algoParams = fillNoDataParams, 
                         generateOutputProfileComputer = generate_identical_profile, 
                         nbWorkers = nb_max_workers, 
                         stableMargin = 100,
                         inMemory=False)

    
def write_quality_mask(border_nodata_mask: np.ndarray,
                       inner_nodata_mask : np.ndarray, 
                       disturbed_area_mask : np.ndarray,
                       quality_mask_path: str,
                       profile : rasterio.profiles.Profile) -> None:
    """
    This method merges the nodata masks generated during the DSM preprocessing into a single quality mask.
    There is a priority order: inner_nodata > disturbance
    (e.g. if a pixel is tagged as disturbed and inner_nodata, the output value will correspond to inner_nodata).

    Args:
        inner_nodata_mask: nodata areas in the input DSM.
        disturbed_area_mask: areas flagged as nodata due to their aspect (mainly correlation issue).
        output_dir: bulldozer output directory. The quality mask will be written in this folder.
        profile: DSM profile (TIF metadata).
    """     
    quality_mask_profile = profile.copy()

    # Metadata update
    quality_mask_profile['dtype'] = np.uint8
    quality_mask_profile['count'] = 3
    # We don't except nodata value in this mask
    quality_mask_profile['nodata'] = None
    buffer = np.stack([inner_nodata_mask, disturbed_area_mask, border_nodata_mask], axis=0)
    with rasterio.open(quality_mask_path, 'w', nbits=1, **quality_mask_profile) as dst_dataset:
        dst_dataset.write(buffer)
        dst_dataset.close()

def preprocess_pipeline(dsm_path : str, 
                        output_dir : str,
                        nb_max_workers : int = 1,
                        nodata : float = None,
                        slope_threshold : float = 2.0, 
                        is_four_connexity : bool = True,
                        minValidHeight: float = None) -> None:
    """
    This method merges the nodata masks generated during the DSM preprocessing into a single quality mask.
    There is a priority order: inner_nodata > disturbance
    (e.g. if a pixel is tagged as disturbed and inner_nodata, the output value will correspond to inner_nodata).

    Args:
        dsm_path: path to the input DSM.
        output_dir: bulldozer output directory. The quality mask will be written in this folder.
        nb_max_workers: number of availables workers (for multiprocessing purpose).
        nodata: nodata value of the input DSM. If None, retrieve this value from the input DSM metadata.
        slope_treshold: if the slope is greater than this threshold then we consider it as disturbed variation.
        is_four_connexity: number of evaluated axis. 
                        Vertical and horizontal if true else vertical, horizontal and diagonals.
        minValidHeight: DSM minimum valid elevation. All the points lower this threshold will be consider as nodata.             
    """ 

    BulldozerLogger.log("Starting preprocess", logging.DEBUG)
    outputFilledDsmPath = os.path.join(output_dir, 'preprocessed_DSM.tif')

    # The value is already retrieved before calling the preprocess method
    # If it is none, it is set automatically to -32768
    if nodata is None:
        nodata = retrieve_nodata(dsm_path)

    with rasterio.open(dsm_path) as dsm_dataset:
        preprocessedDsmProfile = dsm_dataset.profile
        preprocessedDsmProfile['nodata'] = nodata
        

    # Read the buffer in shared memory
    with Shared.make_shared_from_rasterio(dsm_path) as shared_dsm:
        dsm = shared_dsm.getArray()
        dsm_memory_path = shared_dsm.get_memory_path()
        
        # Get the maximum height value
        maxValidHeight: float = np.amax(dsm)
        
        # handle the case where there are dynamic nodata values (MicMac DSM for example)
        if minValidHeight is not None:
            BulldozerLogger.log("Min valid height set by the user" + str(minValidHeight), logging.INFO)
            dsm[:] = np.where( dsm < minValidHeight, nodata, dsm)[:]
        else:
            BulldozerLogger.log("Min valid height is not set by the user, this may be dangerous if there are aberrant low height values !", logging.DEBUG)

        # Retrieves the disturbed area mask (mainly correlation issues: occlusion, water, etc.)
        BulldozerLogger.log("Compute disturbance mask", logging.DEBUG)
        disturbed_area_mask = build_disturbance_mask(dsm_memory_path, nb_max_workers, slope_threshold, is_four_connexity, nodata)
        BulldozerLogger.log("disturbance mask: Done", logging.INFO)


        with Shared.make_shared_from_numpy(disturbed_area_mask) as shared_disturbed_area_mask:

            # Can be unset since it became a shared resource
            disturbed_area_mask = None
            
            # Generates inner nodata mask
            BulldozerLogger.log("Starting inner_nodata_mask and border_nodata_mask building", logging.DEBUG)
            border_nodata_mask = build_border_nodata_mask(dsm_memory_path, nb_max_workers, nodata)
            inner_nodata_mask = np.logical_and(np.logical_not(border_nodata_mask), dsm == nodata)
            BulldozerLogger.log("inner_nodata_mask and border_nodata_mask generation: Done", logging.INFO)
        
            # Replace border nodata by maximum valid height
            dsm[border_nodata_mask] = maxValidHeight

            # Merges and writes the quality mask
            quality_mask_path = os.path.join(output_dir, "quality_mask.tif")
            write_quality_mask(border_nodata_mask, inner_nodata_mask, shared_disturbed_area_mask.getArray(), quality_mask_path, preprocessedDsmProfile)

            # Merge inner nodata and disturbed areas into a mask indicating all the pixels to interpolate by idw
            pixelsToInterpolate = np.invert(np.logical_or(inner_nodata_mask, shared_disturbed_area_mask.getArray()))
            
            border_nodata_mask = None
            inner_nodata_mask = None
            
            with Shared.make_shared_from_numpy(pixelsToInterpolate) as shared_pixels_to_interpolate_mask:
    
                pixelsToInterpolate = None

                pixels_to_interpolate_mem_path = shared_pixels_to_interpolate_mask.get_memory_path()

                multiProcsFillNoData( inputImagePaths= [dsm_memory_path, pixels_to_interpolate_mem_path], 
                                      outputPath = outputFilledDsmPath,
                                      nb_max_workers= nb_max_workers)

    # The input dsm for dtm extraction must be filled !
    BulldozerLogger.log("Preprocess done", logging.INFO)
    return outputFilledDsmPath, quality_mask_path
