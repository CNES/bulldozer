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

def build_sinks_mask(dtm : np.ndarray, resolution : float) -> (np.ndarray, np.ndarray):
    """
    This method detects sinks in the input DTM.
    Those sinks are generated during the DTM extraction by remaining artefacts.

    Args:
        dtm: DTM extracted with bulldozer.
        resolution: DTM resolution (in meters).

    Returns:
        low frequency DTM and the mask flagging the sinks area in the input DTM.
    """
    sharp_sinks_mask = np.zeros(np.shape(dtm), dtype=bool)
    # Generates the low frenquency DTM
    # TODO Release 2 : remove the magic number for the filter size
    dtm_LF = ndimage.uniform_filter(dtm, size = round(35.5/resolution))

    # Retrieves the high frenquencies in the input DTM
    dtm_HF = dtm - dtm_LF
    
    # Tags the sinks
    sharp_sinks_mask[dtm_HF < 0.] = 1
    
    return dtm_LF, sharp_sinks_mask

def postprocess(dtm_path : str, 
                output_dir : str,
                quality_mask_path : str, 
                dhm : bool = False,
                dsm_path : str = None) -> None:
    """
    This method removes sharpsinks and reprojects/dezoom the output DTM if required.
    It also generates the DHM if the option is activated.

    Args:
        dtm_path: path to the DTM generated with bulldozer.
        output_dir : path to the output directory.
        quality_mask_path: path to the quality mask associated with the DTM.
        dhm : option that indicates if bulldozer has to generate the DHM (DSM-DTM).
        dsm_path : path to the input DSM. This argument is required for the DHM generation.

    """
    with rasterio.open(dtm_path) as dtm_dataset:
        # Read the result DTM from the DTM extraction
        dtm = dtm_dataset.read(1)
        # Generates the sinks mask and retrieves the low frequency DTM
        dtm_LF, sinks_mask = build_sinks_mask(dtm, dtm_dataset)
        # Interpolates the sinks in the initial DTM with the elevation of the low frequency DTM
        dtm[sinks_mask] = dtm_LF[sinks_mask]
        # Overrides the old DTM
        write_dataset(dtm_path, dtm, dtm_dataset.profile)

        # Updates the output quality mask
        with rasterio.open(quality_mask_path) as q_mask_dataset:
            # Retrieves the quality masks generated during the DSM preprocess
            quality_mask = q_mask_dataset.read(1)
            border_nodata = (quality_mask == 1)
            inner_nodata = (quality_mask == 2)
            disturbed_areas = (quality_mask == 3)
            
            # Keeps the following priority order : border_nodata(1) > inner_nodata(2) > disturbance(3) > sink(4)
            quality_mask[sinks_mask] = 4
            quality_mask[disturbed_areas] = 3
            quality_mask[inner_nodata] = 2
            quality_mask[border_nodata] = 1
            # Overrides the previous quality mask by adding the sinks_masks
            write_dataset(quality_mask_path, quality_mask, q_mask_dataset.profile)
                
        # Generates the DHM (DSM - DTM) if the option is activated
        if dhm:
            with rasterio.open(dsm_path) as dsm_dataset:
                dsm = dsm_dataset.read(1)
                write_dataset(output_dir, dsm - dtm, dtm_dataset.profile)
        #TODO Release2 : add reprojection and dezoom option
        # Check if the output CRS or resolution is different from the input. If it's different, 
        # if (output_CRS and output_CRS!=input_CRS) or (output_res and input_resolution!=out_resolution):
        

