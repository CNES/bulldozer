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

class PostProcess(object):
    """
        Bulldozer post processing set of functions.
    """

    def __init__(self) -> None:
        pass

    def build_sinks_mask(dtm : np.ndarray) -> (np.ndarray, np.ndarray):
        """
        This method detects sinks in the input DTM.
        Those sinks are generated during the DTM extraction by remaining artefacts.

        Args:
            dtm: DTM extracted with bulldozer.

        Returns:
            low frequency DTM and the mask flagging the sinks area in the input DTM.
        """
        logger.debug("Starting sinks mask building") 
        sharp_sinks_mask = np.zeros(np.shape(dtm), dtype=bool)
        # Generates the low frenquency DTM
        # TODO Release 2 : remove the magic number for the filter size (at least size = round(35.5/resolution))
        dtm_LF = ndimage.uniform_filter(dtm, size = 70)

        # Retrieves the high frenquencies in the input DTM
        dtm_HF = dtm - dtm_LF
        
        # Tags the sinks
        sharp_sinks_mask[dtm_HF < 0.] = 1

        logger.info("Sinks mask generation: Done")
        return dtm_LF, sharp_sinks_mask


    def build_dhm(dtm : np.ndarray, dsm_path : str, output_dir : str) -> None:
        """
        This method generates a DHM (DTM-DSM) in the directory provided.

        Args:
            dtm : DTM in raster format.
            dsm_path : path to the input DSM.
            output_dir : path to the output directory.
        """
        logger.debug("Starting DHM generation") 
        with rasterio.open(dsm_path) as dsm_dataset:
            dsm = dsm_dataset.read(1)
            write_dataset(output_dir + 'DHM.tif', dsm - filled_dtm, dtm_dataset.profile)
        logger.info("DHM generation: Done")
        return dtm_LF, sharp_sinks_mask


    def run(dtm_path : str, 
            output_dir : str,
            quality_mask_path : str = None, 
            dhm : bool = False,
            dsm_path : str = None) -> None:
        """
        Bulldozer postprocess pipeline. It removes sharpsinks.
        It also generates the DHM if the option is activated.

        Args:
            dtm_path : path to the DTM generated with bulldozer.
            output_dir : path to the output directory.
            quality_mask_path : path to the quality mask associated with the DTM.
            dhm : option that indicates if bulldozer has to generate the DHM (DSM-DTM).
            dsm_path : path to the input DSM. This argument is required for the DHM generation.

        """
        logger.debug("Starting postprocess")
        with rasterio.open(dtm_path) as dtm_dataset:
            # Read the result DTM from the DTM extraction
            dtm = dtm_dataset.read(1)
            with rasterio.open(quality_mask_path) as q_mask_dataset:
                quality_mask = q_mask_dataset.read(1)
                filled_dtm = fillnodata(dtm, mask=np.invert(quality_mask > 0))

                # Generates the sinks mask and retrieves the low frequency DTM
                dtm_LF, sinks_mask = build_sinks_mask(filled_dtm)
                # Interpolates the sinks in the initial DTM with the elevation of the low frequency DTM
                dtm[sinks_mask] = dtm_LF[sinks_mask]
                # Overrides the old DTM if run funct is called throught the bulldozer pipeline
                write_dataset(output_dir + 'DTM.tif', filled_dtm, dtm_dataset.profile)

                # Updates the output quality mask
                # Retrieves the quality masks generated during the DSM preprocess
                inner_nodata = (quality_mask == 1)
                disturbed_areas = (quality_mask == 2)
                
                # Keeps the following priority order : inner_nodata(1) > disturbance(2) > sink(3)
                quality_mask[sinks_mask] = 3
                quality_mask[disturbed_areas] = 2
                quality_mask[inner_nodata] = 1
                # Overrides the previous quality mask by adding the sinks_masks
                write_dataset(quality_mask_path, quality_mask, q_mask_dataset.profile)
                    
            # Generates the DHM (DSM - DTM) if the option is activated
            if dhm and dsm_path:
                build_dhm(filled_dtm, dsm_path, output_dir)

            #TODO Release2 : add reprojection and dezoom option
            # Check if the output CRS or resolution is different from the input. If it's different, 
            # if (output_CRS and output_CRS!=input_CRS) or (output_res and input_resolution!=out_resolution):
            logger.info("Postprocess : Done")