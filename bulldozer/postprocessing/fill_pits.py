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

    :param inputBuffers: DTM buffer
    :return: a List composed of the processed dtm without pits and the pits mask
    """
    dtm = inputBuffers[0][0, :, :]
    pits_mask = np.zeros(dtm.shape, dtype=np.ubyte)

    border_mask = inputBuffers[1][0, :, :]

    dtm[border_mask==1] = params["nodata"]

    no_data_mask = np.where(border_mask==1, 0, 1)
    no_data_mask_new = np.zeros(border_mask.shape, dtype=np.ubyte)

    # Retrieves the first and last rows and columns with valid data
    valid_rows = np.any(no_data_mask, axis=1)
    valid_cols = np.any(no_data_mask, axis=0)

    first_valid_row = np.argmax(valid_rows)
    last_valid_row = len(valid_rows) - 1 - np.argmax(valid_rows[::-1])

    first_valid_col = np.argmax(valid_cols)
    last_valid_col = len(valid_cols) - 1 - np.argmax(valid_cols[::-1])

    # Computes the first and last rows and columns to fill for the uniform filter
    first_fill_col = first_valid_col - round(params["filter_size"])-1
    last_fill_col = last_valid_col + round(params["filter_size"])+1
    first_fill_row = first_valid_row - round(params["filter_size"])-1
    last_fill_row = last_valid_row + round(params["filter_size"])+1

    if first_fill_col>=0:
        no_data_mask[:, :first_fill_col] = 1
        no_data_mask_new[:, :first_fill_col] = 1

    if last_fill_col<=np.shape(no_data_mask)[0]:
        no_data_mask[:, last_fill_col:] = 1
        no_data_mask_new[:, last_fill_col:] = 1

    if first_fill_row>=0:
        no_data_mask[:first_fill_row, :] = 1
        no_data_mask_new[:first_fill_row, :] = 1

    if last_fill_row<=np.shape(no_data_mask)[1]:
        no_data_mask[last_fill_row:, :] = 1
        no_data_mask_new[last_fill_row:, :] = 1

    # Fill dtm for the uniform filter
    dtm = fillnodata(dtm, no_data_mask, 250)

    dtm_LF = ndimage.uniform_filter(dtm, size=params["filter_size"])

    # Retrieves the high frequencies in the input DTM
    dtm_HF = dtm - dtm_LF

    # Tags the pits
    pits_mask[dtm_HF < 0.] = 1
    pits_mask[border_mask==1] = 0

    # fill pits
    dtm = np.where( (pits_mask) & (dtm != params["nodata"]), dtm_LF, dtm)

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
        border_nodata_key: str,
        eomanager: eom.EOContextManager):
    """
    Performs the pit removal process using EOScale.

    :param dtm_key: the dtm to process key in the eo manager
    :param border_nodata_key: Border no data
    :return : The processed dtm and the pits mask keys
    """
    resolution = eomanager.get_profile(dtm_key)['transform'][0]
    filter_size = 35.5 / resolution

    fill_pits_parameters: dict = {
        "filter_size": filter_size,
        "nodata": eomanager.get_profile(dtm_key)['nodata']
    }

    [filled_dtm_key, pits_mask_key] = \
        eoexe.n_images_to_m_images_filter(inputs=[dtm_key, border_nodata_key],
                                          image_filter=fill_pits_filter,
                                          filter_parameters=fill_pits_parameters,
                                          generate_output_profiles=fill_pits_profile,
                                          context_manager=eomanager,
                                          stable_margin=int(filter_size/2),
                                          filter_desc="Pits removal processing...")

    eomanager.release(key=dtm_key)
    dtm_key = filled_dtm_key

    return dtm_key, pits_mask_key
