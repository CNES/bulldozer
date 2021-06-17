# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

"""
    This module is used to postprocess the DTM in order to improve its quality. It required a DTM generated from Bulldozer.
"""
import logging
import rasterio
import numpy as np
import scipy.ndimage as ndimage
from bulldozer.utils.helper import write_dataset

logger = logging.getLogger(__name__)

def build_sinks_mask(dtm: np.ndarray):
    """
    The extraction of DTM from photogrametric DSM can result in some sharp sinks (dark areas, local height artefacts).
    This methods uses a median filter / or sinks detection and remove them by interpolation.
    """
    sharp_sinks_mask = np.zeros(np.shape(dtm), dtype=bool)
    
    # Sharpsinks are high frenquency objects and local minima
    # First, we extract the high frenquency DTM removing low frequency
    dtm_LF = ndimage.uniform_filter(dtm, size = 71)
    dtm_HF = dtm - dtm_LF
    
    # Then we detect minima
    sharp_sinks = dtm_HF < 0.
         
    dtm[sharp_sinks] = dtm_LF[sharp_sinks]
    sharp_sinks_mask[sharp_sinks] = 1
    
    return dtm, sharp_sinks_mask

def restoreNoDataOnEdge(dtm: np.ndarray, border_no_data_mask: np.ndarray, no_data_value: float) -> np.ndarray:
    """
    XXX
    """
    restored_dtm = np.copy(dtm)
    restored_dtm[border_no_data_mask] = no_data_value
    
    return restored_dtm

def merge_nodata_masks(border_nodata_mask: np.array, 
                    inner_nodata_mask : np.array,
                    disturbance_mask : np.array,
                    sink_mask : np.array):
    """
      #TODO
    border_mask_value: mask value referring to the border nodata areas.
        inner_mask_value: mask value referring to the inner nodata areas.
    """
    try:
        # Init the quality mask based on one of the input mask shape (all the input mask should have the same shape)
        quality_mask = np.zeros(border_nodata_mask.shape)
        # Iterates throught rows and columns and assigns the corresponding value
        for row in range(border_nodata_mask.shape[0]):
            for col in range(border_nodata_mask.shape[1]):
                quality_mask[row,col] = compute_nodata_value(border_nodata_mask[row,col], inner_nodata_mask[row,col], disturbance_mask[row,col], sink_mask[row,col])
        # Returns the merged nodata mask
        return quality_mask
    except IndexError:
        logger.exception("Input nodata masks shape mismatch:\n- border nodata mask shape:{}\n- inner nodata mask shape:{}\n- disturbance mask shape:{}\n- sink mask shape:{}".format(border_nodata_mask.shape, inner_nodata_mask.shape, disturbance_mask.shape, sink_mask.shape))
        raise IndexError


def compute_nodata_value(border_nodata : bool, inner_nodata : bool, disturbance : bool, sink : bool) -> np.uint8:
    """
    This method assigns a nodata value for a given pixel.
    According to the quality masks values for this pixel, it returns a value corresponding to the nodata source.
    There is a priority order: border_nodata > inner_nodata > disturbance > sink (e.g. if a pixel is tagged as disturbed and border_nodata, the output value will correspond to border_nodata).

    Args:
        border_nodata: value of the border nodata mask of the analyzed pixel.
        inner_nodata: value of the inner nodata mask of the analyzed pixel.
        disturbance: value of the disturbance mask of the analyzed pixel.
        sink: value of the sink mask of the analyzed pixel.

    Returns:
        integer value (0: not a nodata point/1: border nodata/2: inner nodata/3: disturbed area/4: filled sink).
    """
    if border_nodata:
        # Nodata value referring to the border nodata areas mask
        return 1
    elif inner_nodata:
        # Nodata value of inner nodata mask (mainly correlation issues during the DSM making)
        return 2
    elif disturbance:
        # Nodata value referring to the disturbed area (water, etc.)
        return 3
    elif sink:
        # Nodata value referring to the sharp sinks observed after the dtm computation
        return 4
    else:
        # Not a nodata pixel 
        return 0

def postprocess(dtm, output_dir, border_nodata_mask, inner_nodata_mask, disturbance_mask, sink_mask):
    """
    Remove sharpsinks and restore NO_DATA values on edges
    """
    dtm, sink_mask = build_sinks_mask(dtm)
    quality_mask = merge_nodata_masks(border_nodata_mask, inner_nodata_mask, disturbance_mask, sink_mask)
    # Check if the output CRS or resolution is different from the input. If it's different, 
    # if (output_CRS and output_CRS!=input_CRS) or (output_res and input_resolution!=out_resolution):
    
    # Writes quality mask
    quality_path = output_dir + "quality_mask.tif"
    try:
        write_dataset(quality_path, quality_mask, dtm.profile)
    except (FileNotFoundError, rasterio.RasterioIOError) as err:
        logger.error('Invalid quality mask path provided ({})\nError: {}'.format(quality_path, err))

    
    #self.dtm = core.restoreNoDataOnEdge(self.dtm, self.border_no_data_mask, self.no_data_value)
