from typing import List
import logging
from copy import copy

import numpy as np
import scipy.ndimage as ndimage

from bulldozer.utils.bulldozer_logger import BulldozerLogger
import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe


def fill_pits_filter(inputBuffers: list,
                     input_profiles: list,
                     params: dict) -> List[np.ndarray]:
    """
    Perform pits removal and create pits detection mask.

    :param inputBuffers: DTM buffer, border no data
    :return: a List composed of the processed dtm without pits and the pits mask
    """
    dtm = inputBuffers[0][0, :, :]
    pits_mask = np.zeros(dtm.shape, dtype=np.ubyte)

    # Generates the low frequency DTM
    # bfilters = sf.PyBulldozerFilters()
    border_mask_expanded = ndimage.binary_dilation(inputBuffers[1][0, :, :], structure=ndimage.generate_binary_structure(2, 2), iterations=round(params["filter_size"]))
    
    dtm_LF = ndimage.uniform_filter(dtm, size=params["filter_size"])

    # Retrieves the high frequencies in the input DTM
    dtm_HF = dtm - dtm_LF

    # Tags the pits
    pits_mask[dtm_HF < 0.] = 1
    pits_mask[border_mask_expanded] = 0

    # fill pits
    #dtm[pits_mask] = dtm_LF
    dtm = np.where( (pits_mask) & (dtm != params["nodata"]) & (dtm != np.nan), dtm_LF, dtm)

    return [dtm, pits_mask]


def fill_pits_profile(input_profiles: list,
                      params: dict) -> dict:
    """
    Defines filter outputs profiles
    """
    msk_profile = copy(input_profiles[0])
    msk_profile['dtype'] = np.uint8
    msk_profile['nodata'] = None
    return [input_profiles[0], msk_profile]


def run(dtm_key: str,
        eomanager: eom.EOContextManager):
    """
    Performs the pit removal process using EOScale.

    :param dtm_key: the dtm to process key in the eo manager
    :return : The processed dtm and the pits mask keys
    """
    resolution = eomanager.get_profile(dtm_key)['transform'][0]
    filter_size = 35.5 / resolution

    fill_pits_parameters: dict = {
        "filter_size": filter_size,
        "nodata": eomanager.get_profile(dtm_key)['nodata']
    }

    [filled_dtm_key, pits_mask_key] = \
        eoexe.n_images_to_m_images_filter(inputs=[dtm_key],
                                          image_filter=fill_pits_filter,
                                          filter_parameters=fill_pits_parameters,
                                          generate_output_profiles=fill_pits_profile,
                                          context_manager=eomanager,
                                          stable_margin=int(filter_size/2),
                                          filter_desc="Pits removal processing...")

    eomanager.release(key=dtm_key)
    dtm_key = filled_dtm_key

    return dtm_key, pits_mask_key
