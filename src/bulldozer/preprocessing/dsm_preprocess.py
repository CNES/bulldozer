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

import sys
import rasterio
import os
import concurrent.futures
import numpy as np
import logging
import scipy.ndimage as ndimage
import bulldozer.disturbedareas as da
from rasterio.windows import Window
from rasterio.fill import fillnodata
from bulldozer.utils.helper import write_dataset
from tqdm import tqdm
from os import remove
from bulldozer.utils.logging_helper import BulldozerLogger
from bulldozer.scale.tools import computeTiles, scaleRun

# No data value constant used in bulldozer
NO_DATA_VALUE = -32768

def build_inner_nodata_mask(dsm : np.ndarray) -> np.ndarray:
    """
    This method builds a mask corresponding to inner nodata values in a given DSM.
    (mainly correlation issues in the DSM)

    Args:
        dsm: array containing DSM values.

    Returns:
        boolean mask corresponding to the inner nodata areas.
    """
    
    # Get the global nodata mask
    nodata_area = (dsm == NO_DATA_VALUE)

    # Connect the groups of nodata elements into labels (non-zero values)
    labeled_array, _ = ndimage.label(nodata_area)

    # Get the label IDs corresponding to nodata regions touching the edges
    border_region_ident = np.unique(
                            np.concatenate((np.unique(labeled_array[0,:]),
                            np.unique(labeled_array[-1,:]),
                            np.unique(labeled_array[:,0]),
                            np.unique(labeled_array[:,-1])), axis = 0))
    # Remove ID = 0 which correspond to background (not nodata areas)
    border_region_ident = border_region_ident[border_region_ident != 0]

    # Retrieve all the border nodata areas and create the corresponding mask 
    border_nodata_mask = np.isin(labeled_array,border_region_ident)

    # Retrieve all the nodata areas in the input DSM that aren't border areas and create the corresponding mask 
    inner_nodata_mask = np.logical_and(nodata_area == True, np.isin(labeled_array,border_region_ident) != True)

    return [border_nodata_mask, inner_nodata_mask]


def compute_disturbance(dsm_path : rasterio.DatasetReader,
                        window : rasterio.windows.Window,
                        slope_treshold : float,
                        is_four_connexity : bool) -> (np.ndarray, rasterio.windows.Window) :
    """
    This method computes the disturbance in a DSM window through the horizontal axis.
    It returns the corresping disturbance mask.

    Args:
        dsm_path: path to the input DSM.
        window: coordinates of the concerned window.
        slope_treshold: if the slope is greater than this threshold then we consider it as disturbed variation.
        is_four_connexity: number of evaluated axis. 
                        Vertical and horizontal if true else vertical, horizontal and diagonals.

    Returns:
        mask flagging the disturbed area and its associated window location in the input DSM.
    """
    #logger.debug("Starting disturbed area analysis. Window strip: {}".format(window))
    with rasterio.open(dsm_path, 'r') as dataset:
        dsm_strip = dataset.read(1, window=window).astype(np.float32)
        disturbed_areas = da.PyDisturbedAreas(is_four_connexity)
        disturbance_mask = disturbed_areas.build_disturbance_mask(dsm_strip, slope_treshold, NO_DATA_VALUE).astype(np.ubyte)
        #logger.debug("Disturbance mask computation: Done (Window strip: {}".format(window))
        return disturbance_mask, window


def generateOuputProfileForDisturbedAreas(inputProfile: rasterio.DatasetReader.profile):
    """
        Only the dtype change 
    """
    outputProfile = inputProfile
    outputProfile["dtype"] = np.ubyte
    return outputProfile

def disturbedAreasComputer(inputBuffers: list, params: dict) -> np.ndarray:
    """
    """
    disturbed_areas = da.PyDisturbedAreas(params["is_four_connexity"])
    disturbance_mask = disturbed_areas.build_disturbance_mask(inputBuffers[0], 
                                                              params["slope_threshold"],
                                                              params["nodata"]).astype(np.ubyte)
    return disturbance_mask


def build_disturbance_mask(dsm_path: str,
                           nb_max_workers : int,
                           slope_treshold: float = 2.0,
                           is_four_connexity : bool = True,
                           nodata: float = NO_DATA_VALUE) -> np.array:
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

    disturbanceParams = {
        "is_four_connexity": is_four_connexity,
        "slope_threshold": slope_treshold,
        "nodata": nodata,
        "desc": "Build Disturbance Mask"
    }

    disturbance_mask = scaleRun(inputImagePaths = [dsm_path], 
                                outputImagePath = None,
                                algoComputer= disturbedAreasComputer, 
                                algoParams = disturbanceParams, 
                                generateOutputProfileComputer =generateOuputProfileForDisturbedAreas, 
                                nbWorkers = nb_max_workers, 
                                stableMargin = 1,
                                inMemory=True)

    # # Determine the number of strip and their height
    # with rasterio.open(dsm_path, 'r') as dataset:
    #     strip_height = dataset.height // nb_max_workers
    #     strips = [[i*strip_height-1, (i+1)*strip_height] for i in range(nb_max_workers)]
    #     # Borders handling
    #     strips[0][0] = 0
    #     strips[-1][1] = dataset.height - 1

    #     # Output binary mask initialization
    #     disturbance_mask = np.zeros((dataset.height, dataset.width), dtype = np.ubyte)

    #     # Launching parallel processing: each worker computes the disturbance mask for a given DSM strip
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=nb_max_workers) as executor :
    #         futures = {executor.submit(compute_disturbance, dsm_path, Window(0,strip[0],dataset.width,strip[1]-strip[0]+1), 
    #         slope_treshold, is_four_connexity) for strip in strips}
    #         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Build Disturbance Mask") :
    #             mask, window = future.result()
    #             window_shape = window.flatten()
    #             start_row = window_shape[1]
    #             end_row = start_row + window_shape[3] - 1
    #             start_row_mask = 0
    #             end_row_mask = mask.shape[0] - 1 
    #             if start_row > 0:
    #                 start_row_mask = 1
    #                 start_row = start_row + 1
                
    #             if end_row < dataset.height - 1:
    #                 end_row_mask = end_row_mask - 1
    #                 end_row = end_row - 1

    #             disturbance_mask[start_row:end_row+1,:] = mask[start_row_mask:end_row_mask+1, :]
    return disturbance_mask != 0
    
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
    quality_mask = np.zeros(np.shape(inner_nodata_mask), dtype=np.uint8)

    # Metadata update
    profile['dtype'] = np.uint8
    profile['count'] = 1
    # We don't except nodata value in this mask
    profile['nodata'] = None
    quality_mask[disturbed_area_mask] = 2
    quality_mask[inner_nodata_mask] = 1
    quality_mask[border_nodata_mask] = 3
    write_dataset(quality_mask_path, quality_mask, profile)



def preprocess_pipeline(dsm_path : str, 
                        output_dir : str,
                        nb_max_workers : int,
                        nodata : float = None,
                        slope_treshold : float = 2.0, 
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
    """ 

    BulldozerLogger.log("Starting preprocess", logging.INFO)

    # The value is already retrieved before calling the preprocess method
    # If it is none, it is set automatically to -32768
    if nodata is None:
        BulldozerLogger.log("No data value is set to " + str(NO_DATA_VALUE), logging.INFO)
        nodata = NO_DATA_VALUE

    with rasterio.open(dsm_path) as dsm_dataset:
        
        # Read the buffer in memory
        dsm = dsm_dataset.read(1)
        
        # handle the case where there are dynamic nodata values (MicMac DSM for example)
        if minValidHeight:
            BulldozerLogger.log("Min valid height set by the user" + str(minValidHeight), logging.INFO)
            dsm = np.where( dsm < minValidHeight, nodata, dsm)

        preprocessedDsmProfile = dsm_dataset.profile
        preprocessedDsmProfile['nodata'] = nodata
        
        # Generates inner nodata mask
        BulldozerLogger.log("Starting inner_nodata_mask building", logging.INFO)
        border_nodata_mask, inner_nodata_mask = build_inner_nodata_mask(dsm)
        dsm[border_nodata_mask] = np.max(dsm)
        BulldozerLogger.log("inner_nodata_mask generation: Done", logging.INFO)
                
        # Retrieves the disturbed area mask (mainly correlation issues: occlusion, water, etc.)
        BulldozerLogger.log("Compute disturbance mask", logging.INFO)
        disturbed_area_mask = build_disturbance_mask(dsm_path, nb_max_workers, slope_treshold, is_four_connexity, nodata)
        BulldozerLogger.log("disturbance mask: Done", logging.INFO)
        
        # Merges and writes the quality mask
        quality_mask_path = os.path.join(output_dir, "quality_mask.tif")
        write_quality_mask(border_nodata_mask, inner_nodata_mask, disturbed_area_mask, quality_mask_path, dsm_dataset.profile)

    #     bulldoLogger.info("Filled no data values")
    #     # Generates filled DSM if the user provides a valid filled_dsm_path
    #     if create_filled_dsm:
    #         filled_dsm = fillnodata(dsm, mask=np.invert(inner_nodata_mask))
    #         filled_dsm = fillnodata(filled_dsm, mask=np.invert(disturbed_area_mask))

    #         filled_dsm_path = os.path.join(output_dir, 'filled_DSM.tif')

    #         # Generates the filled DSM file (DSM without inner nodata nor disturbed areas)
    #         write_dataset(filled_dsm_path, filled_dsm, filledDSMProfile)
    #     bulldoLogger.info("Filled no data values: Done")
        
        dsm[disturbed_area_mask] = nodata


        # Creates the preprocessed DSM. This DSM is only intended for bulldozer DTM extraction function.
        BulldozerLogger.log("Write preprocessed dsm", logging.INFO)
        preprocessed_dsm_path = os.path.join(output_dir, 'preprocessed_DSM.tif')
        write_dataset(preprocessed_dsm_path, dsm, preprocessedDsmProfile)

        dsm_dataset.close()
        BulldozerLogger.log("Preprocess done", logging.INFO)

        return preprocessed_dsm_path, quality_mask_path