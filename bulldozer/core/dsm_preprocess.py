# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

"""
    This module is used to preprocess the DSM in order to improve the DTM computation.
"""

import sys
import logging
import rasterio
import os.path
import concurrent.futures
import numpy as np
import scipy.ndimage as ndimage
import DisturbedAreas as da
from rasterio.windows import Window
from rasterio.fill import fillnodata
from bulldozer.utils.helper import write_dataset
from tqdm import tqdm
from os import remove

logger = logging.getLogger(__name__)

def build_nodata_mask(dsm : np.ndarray, nodata_value : float) -> np.ndarray:
    """
    This method builds a mask corresponding to inner nodata values in a given DSM.
    (mainly correlation issues in the DSM)

    Args:
        dsm: array containing DSM values.
        nodata_value: nodata value.

    Returns:
        boolean mask corresponding to the inner nodata areas.
    """
    logger.debug("Starting noDataMask building")
    # Get the global nodata mask
    if np.isnan(nodata_value):
        nodata_value = -32768
        dsm = np.nan_to_num(dsm, False, nodata_value)
    nodata_area = (dsm == nodata_value)

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


    # Retrieve all the nodata areas in the input DSM that aren't border areas and create the corresponding mask 
    inner_nodata_mask = np.logical_and(nodata_area == True, np.isin(labeled_array,border_region_ident) != True)
    logger.debug("inner_nodata_mask generation: Done")

    return inner_nodata_mask


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
        disturbed_treshold: if the number of successive disturbed pixels along a row is lower than this threshold 
                            then this sequence of pixels is considered as a disturbed area.
        disturbed_influence_distance: if the distance between 2 lists of disturbed cols is lower than this threshold 
                                        expressed in meters then they are merged.
        dsm_resolution: input DSM resolution (in meters).

    Returns:
        mask flagging the disturbed area and its associated window location in the input DSM.
    """
    logger.debug("Starting disturbed area computation")
    with rasterio.open(dsm_path, 'r') as dataset:
        dsm_strip = dataset.read(1, window=window).astype(np.float32)
        disturbed_areas = da.PyDisturbedAreas(is_four_connexity)
        disturbance_mask = disturbed_areas.build_disturbance_mask(dsm_strip, slope_treshold).astype(np.ubyte)
        logger.debug("Disturbance mask computation: Done")
        return disturbance_mask, window

###########
#TODO WIP #
###########
def build_disturbance_mask(dsm_path: str,
                        outputMaskPath : str,
                        nb_max_workers : int,
                        disturbance_nodata : np.uint8,
                        slope_treshold: float = 2.0,
                        is_four_connexity : bool = True):
    """
    This method builds a mask corresponding to the disturbed areas in a given DSM.
    Most of those areas correspond to water or correlation issues during the DSM generation (obstruction, etc.).

    Args:
        dsm_path: path to the input DSM.
        window: coordinates of the concerned window.
        slope_treshold: if the slope is greater than this threshold then we consider it as disturbed variation.
        disturbed_treshold: if the number of successive disturbed pixels along a row is lower than this threshold 
                            then this sequence of pixels is considered as a disturbed area.
        disturbed_influence_distance: if the distance between 2 lists of disturbed cols is lower than this threshold 
                                        expressed in meters then they are merged.

    Returns:
        masks containing the inner and border nodata areas.
    """
    # Determine the number of strip and their height
    with rasterio.open(dsm_path, 'r') as dataset:
        strip_height = dataset.height // nb_max_workers
        strips = [[i*strip_height-1, (i+1)*strip_height] for i in range(nb_max_workers)]
        # border handling
        strip[0][0] = 0
        strips[-1][1] = dataset.height - 1

        disturbance_mask = np.zeros((dataset.height, dataset.width), dtype = np.ubyte)
        with concurrent.futures.ProcessPoolExecutor(max_workers=nb_max_workers) as executor :
            futures = {executor.submit(compute_disturbance, dsm_path, Window(0,strip[0],dataset.width,strip[1]-strip[0]+1), 
            slope_treshold, is_four_connexity) for strip in strips}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Build Disturbance Mask") :
                mask, window = future.result()
                window_shape = window.flatten()
                start_row = window_shape[1]
                end_row = window.shape[2]
                start_row_mask = 0
                end_row_mask = mask.shape[0] - 1 
                if start_row > 0:
                    start_row_mask = 1
                    start_row = start_row - 1
                
                if end_row < dataset.height - 1:
                    end_row_mask = mask.shape[0]

                disturbance_mask[start_row:end_row,:] = mask[start_row_mask:end_row_mask+1, :]
        
        return disturbance_mask       
    
def write_quality_mask(inner_nodata_mask : np.ndarray, 
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
    quality_mask_path = output_dir + "quality_mask.tif"

    # Metadata update
    profile['dtype'] = np.uint8
    profile['count'] = 1
    # We don't except nodata value in this mask
    profile['nodata'] = 0
    quality_mask[disturbed_area_mask] = 2
    quality_mask[inner_nodata_mask] = 1
    write_dataset(quality_mask_path, quality_mask, profile)


###########
#TODO WIP #
###########
def preprocess(dsm_path : str, 
                output_dir : str,
                nodata : float,
                nb_max_workers : int,
                slope_treshold : float, 
                is_four_connexity : bool,
                filled_dsm_path : str = None) -> None:
    logger.debug("Starting preprocess")
    with rasterio.open(dsm_path) as dsm_dataset:
        if not nodata:
            # If nodata is not specified in the config file, retrieve the value from the dsm metadata
            nodata = dsm_dataset.nodata
        
        dsm = dsm_dataset.read(1)

        # Converts no data
        nodata_value = -32768
        # Get the global nodata mask
        if np.isnan(nodata):
            dsm = np.nan_to_num(dsm, False, nodata_value)
        else:
            dsm = np.where(dsm == nodata, nodata_value, dsm)

        dsm_dataset.profile['nodata'] = nodata_value
        
        # Generates border and inner nodata mask
        inner_nodata_mask = build_nodata_mask(dsm, nodata)
                
         # Retrieves the disturbed area mask (mainly correlation issues: occlusion, water, etc.)
        disturbed_area_mask = build_disturbance_mask(dsm_path, nb_max_workers, slope_treshold, is_four_connexity)
        
        # Merges and writes the quality mask
        write_quality_mask(inner_nodata_mask, disturbed_area_mask, output_dir, dsm_dataset.profile)

        # Generates filled DSM if the user provides a valid filled_dsm_path
        if filled_dsm_path is not None:
            filled_dsm = fillnodata(dsm, mask=np.invert(inner_nodata_mask))
            filled_dsm = fillnodata(filled_dsm, mask=np.invert(disturbed_area_mask))

            # Overwrites the filled DSM file with the disturbance areas filled
            write_dataset(filled_dsm_path, filled_dsm, dsm_dataset.profile)
        
        # Creates the preprocessed DSM (without disturbed areas). This DSM is only intended for bulldozer DTM extraction function.
        dsm[disturbed_area_mask] = nodata

        # We set a very high value for those nodata points
        preprocessed_dsm_path = output_dir + 'preprocessed_DSM.tif'
        write_dataset(preprocessed_dsm_path, dsm, dsm_dataset.profile)

        dsm_dataset.close()
        logger.info("preprocess: Done")


if __name__ == "__main__":
    # assert(len(sys.argv)>=X)#X = nb arguments obligatoires +1 car il y a sys.argv[0] qui vaut le nom de la fonction
    # argv[1] should be the path to the input DSM
    input_dsm_path = sys.argv[1]

    # input file format check
    if not (input_dsm_path.endswith('.tiff') or input_dsm_path.endswith('.tif')) :
        logger.exception('\'dsm_path\' argument should be a path to a TIF file (here: {})'.format(input_dsm_path))
        raise ValueError()
    # input file existence check
    if not os.path.isfile(input_dsm_path):
        logger.exception('The input DSM file \'{}\' doesn\'t exist'.format(input_dsm_path))
        raise FileNotFoundError()
    
    output_dir = sys.argv[2]

    # preprocess(input_dsm_path, output_dir,XXX)

    # If this module is called in standalone, the user doesn't required the preprocessed DSM. We remove it
    preprocessed_dsm_path = output_dir + 'preprocessed_DSM.tif' 
    try:
        remove(preprocessed_dsm_path)
    except OSError:
        logger.warning("Error occured during the preprocessed DSM file deletion. Path: {}".format(preprocessed_dsm_path))
