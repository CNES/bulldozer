# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

"""
    This module is used to preprocess the DSM in order to improve the DTM computation.
"""

import sys
import rasterio
import os
import concurrent.futures
import numpy as np
import scipy.ndimage as ndimage
import bulldozer.disturbedareas as da
from rasterio.windows import Window
from rasterio.fill import fillnodata
from bulldozer.utils.helper import write_dataset
from tqdm import tqdm
from os import remove
from bulldozer.utils.logging_helper import BulldozerLogger

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


def build_disturbance_mask(dsm_path: str,
                            nb_max_workers : int,
                            slope_treshold: float = 2.0,
                            is_four_connexity : bool = True,
                            sequential: bool = False) -> np.array:
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
    # Determine the number of strip and their height
    with rasterio.open(dsm_path, 'r') as dataset:
        strip_height = dataset.height // nb_max_workers
        strips = [[i*strip_height-1, (i+1)*strip_height] for i in range(nb_max_workers)]
        # Borders handling
        strips[0][0] = 0
        strips[-1][1] = dataset.height - 1

        # Output binary mask initialization
        disturbance_mask = np.zeros((dataset.height, dataset.width), dtype = np.ubyte)

        if sequential:
            
            for strip in tqdm(strips, desc="Sequential preprocessing..."):
                mask, window = compute_disturbance(dsm_path, 
                                                   Window(0,strip[0],dataset.width,strip[1]-strip[0]+1), 
                                                   slope_treshold, 
                                                   is_four_connexity)
                window_shape = window.flatten()
                start_row = window_shape[1]
                end_row = start_row + window_shape[3] - 1
                start_row_mask = 0
                end_row_mask = mask.shape[0] - 1 
                if start_row > 0:
                    start_row_mask = 1
                    start_row = start_row + 1
                
                if end_row < dataset.height - 1:
                    end_row_mask = end_row_mask - 1
                    end_row = end_row - 1

                disturbance_mask[start_row:end_row+1,:] = mask[start_row_mask:end_row_mask+1, :]

        else:
            # Launching parallel processing: each worker computes the disturbance mask for a given DSM strip
            with concurrent.futures.ProcessPoolExecutor(max_workers=nb_max_workers) as executor :
                futures = {executor.submit(compute_disturbance, dsm_path, Window(0,strip[0],dataset.width,strip[1]-strip[0]+1), 
                slope_treshold, is_four_connexity) for strip in strips}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Build Disturbance Mask") :
                    mask, window = future.result()
                    window_shape = window.flatten()
                    start_row = window_shape[1]
                    end_row = start_row + window_shape[3] - 1
                    start_row_mask = 0
                    end_row_mask = mask.shape[0] - 1 
                    if start_row > 0:
                        start_row_mask = 1
                        start_row = start_row + 1
                    
                    if end_row < dataset.height - 1:
                        end_row_mask = end_row_mask - 1
                        end_row = end_row - 1

                    disturbance_mask[start_row:end_row+1,:] = mask[start_row_mask:end_row_mask+1, :]

        return disturbance_mask != 0
    
def write_quality_mask(border_nodata_mask: np.ndarray,
                       inner_nodata_mask : np.ndarray, 
                       disturbed_area_mask : np.ndarray,
                       output_dir : str,
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
    quality_mask_path = os.path.join(output_dir, "quality_mask.tif")

    # Metadata update
    profile['dtype'] = np.uint8
    profile['count'] = 1
    # We don't except nodata value in this mask
    profile['nodata'] = 0
    quality_mask[disturbed_area_mask] = 2
    quality_mask[inner_nodata_mask] = 1
    quality_mask[border_nodata_mask] = 3
    write_dataset(quality_mask_path, quality_mask, profile)



def run(dsm_path : str, 
        output_dir : str,
        nb_max_workers : int,
        create_filled_dsm : bool = False,
        nodata : float = None,
        slope_treshold : float = 2.0, 
        is_four_connexity : bool = True,
        minValidHeight: float = None,
        sequential: bool = False) -> None:
    """
    This method merges the nodata masks generated during the DSM preprocessing into a single quality mask.
    There is a priority order: inner_nodata > disturbance
    (e.g. if a pixel is tagged as disturbed and inner_nodata, the output value will correspond to inner_nodata).

    Args:
        dsm_path: path to the input DSM.
        output_dir: bulldozer output directory. The quality mask will be written in this folder.
        nb_max_workers: number of availables workers (for multiprocessing purpose).
        nodata: nodata value of the input DSM. If None, retrieve this value from the input DSM metadata.
        create_filled_dsm: flag to indicate if bulldozer has to generate a filled DSM (without nodata or disturbed areas).
        slope_treshold: if the slope is greater than this threshold then we consider it as disturbed variation.
        is_four_connexity: number of evaluated axis. 
                        Vertical and horizontal if true else vertical, horizontal and diagonals.
    """ 



    bulldoLogger = BulldozerLogger.getInstance(loggerFilePath=os.path.join(output_dir, "trace.log"))
    bulldoLogger.info("Starting preprocess")

    with rasterio.open(dsm_path) as dsm_dataset:
        if not nodata:
            # If nodata is not specified in the config file, retrieve the value from the dsm metadata
            nodata = dsm_dataset.nodata
            if nodata is None:
                # We assume that is nodata is not given then nodata = NODATA_VALUE
                nodata = NO_DATA_VALUE
        
        dsm = dsm_dataset.read(1)

        # Converts no data
        if np.isnan(nodata):
            dsm = np.nan_to_num(dsm, False, nan=NO_DATA_VALUE)
        else:
            if nodata != NO_DATA_VALUE:
                dsm = np.where(dsm == nodata, NO_DATA_VALUE, dsm)
        
        # handle the case where there are dynamic nodata values (MicMac DSM for example)
        if minValidHeight is not None:
            dsm = np.where( dsm < minValidHeight, NO_DATA_VALUE, dsm)

        filledDSMProfile = dsm_dataset.profile
        filledDSMProfile['nodata'] = NO_DATA_VALUE
        
        # Generates inner nodata mask
        bulldoLogger.info("Starting inner_nodata_mask building")
        border_nodata_mask, inner_nodata_mask = build_inner_nodata_mask(dsm)
        dsm[border_nodata_mask] = np.max(dsm)
        bulldoLogger.info("inner_nodata_mask generation: Done")
                
        # Retrieves the disturbed area mask (mainly correlation issues: occlusion, water, etc.)
        bulldoLogger.info("Compute disturbance mask")
        disturbed_area_mask = build_disturbance_mask(dsm_path, nb_max_workers, slope_treshold, is_four_connexity, sequential)
        bulldoLogger.info("disturbance mask: Done")
        
        # Merges and writes the quality mask
        write_quality_mask(border_nodata_mask, inner_nodata_mask, disturbed_area_mask, output_dir, dsm_dataset.profile)

        bulldoLogger.info("Filled no data values")
        # Generates filled DSM if the user provides a valid filled_dsm_path
        if create_filled_dsm:
            filled_dsm = fillnodata(dsm, mask=np.invert(inner_nodata_mask))
            filled_dsm = fillnodata(filled_dsm, mask=np.invert(disturbed_area_mask))

            filled_dsm_path = os.path.join(output_dir, 'filled_DSM.tif')

            # Generates the filled DSM file (DSM without inner nodata nor disturbed areas)
            write_dataset(filled_dsm_path, filled_dsm, filledDSMProfile)
        bulldoLogger.info("Filled no data values: Done")
        
        dsm[disturbed_area_mask] = nodata


        # Creates the preprocessed DSM. This DSM is only intended for bulldozer DTM extraction function.
        preprocessed_dsm_path = os.path.join(output_dir, 'preprocessed_DSM.tif')
        write_dataset(preprocessed_dsm_path, dsm, filledDSMProfile)

        dsm_dataset.close()
        bulldoLogger.info("preprocess: Done")
