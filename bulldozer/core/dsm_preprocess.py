# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

"""
    This module is used to preprocess the DSM in order to improve the DTM computation.
"""

import logging
import numpy as np
import scipy.ndimage as ndimage

logger = logging.getLogger(__name__)

def build_nodata_mask(dsm : np.ndarray, no_data_value : float, border_mask_value : int, inner_mask_value : int) -> (np.ndarray, np.ndarray):
    """
    This methods build a mask corresponding to nodata values in a given DSM.
    The mask splits the nodata values into two categories:
        - border nodata values
        - inner nodata values (mainly correlation issues in the DSM)
    because both types of nodata areas will no be similarly corrected.

    Args:
        dsm: array containing DSM values.
        no_data_value: nodata value.
        border_mask_value: mask value referring to the border nodata areas.
        inner_mask_value: mask value referring to the inner nodata areas.

    Returns:
        masks containing the inner and border nodata areas.
    """
    logger.debug("Starting noDataMask building")
    # Get the global nodata mask
    no_data_area = (dsm == no_data_value)

    # Connect the groups of nodata elements into labels (non-zero values)
    labeled_array, _ = ndimage.label(no_data_area)

    # Get the label IDs corresponding to nodata regions touching the edges
    border_region_ident = np.unique(
                            np.concatenate((np.unique(labeled_array[0,:]),
                            np.unique(labeled_array[-1,:]),
                            np.unique(labeled_array[:,0]),
                            np.unique(labeled_array[:,-1])), axis = 0))
    # Remove ID = 0 which correspond to background (not nodata areas)
    border_region_ident = border_region_ident[border_region_ident != 0]

    # Retrieve all the border nodata areas and create the corresponding mask 
    border_no_data_mask = np.where(np.isin(labeled_array,border_region_ident), border_mask_value, 0)

    # Retrieve all the nodata areas in the input DSM that aren't border areas and create the corresponding mask 
    inner_no_data_mask = np.where(np.logical_and(
        no_data_area == True, 
        np.isin(labeled_array,border_region_ident) != True), 
        inner_mask_value,0)
    
    return border_no_data_mask, inner_no_data_mask

def preprocess():
    pass
    # # Combine all masks (inner + border no_data, disturbance)
    # mask = np.logical_or(self.border_no_data_mask, self.inner_no_data_mask)
    # mask = np.logical_or(mask, self.disturbance_mask)
    # if fill_method == 0:
    #     # fillnodata will fill data where mask == 0
    #     self.dsm = fillnodata(self.dsm, mask=np.invert(mask))
    # elif fill_method == 1:
    #     self.dsm[mask] = 9000

