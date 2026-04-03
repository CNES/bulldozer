#!/usr/bin/env python
#
# Copyright (c) 2022-2026 Centre National d'Etudes Spatiales (CNES).
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
This module is used to fill the remaining pits in the generated DTM.
"""

import logging
import shutil

import numpy as np
from rasterio.fill import fillnodata
from scipy import ndimage

from bulldozer.eomultiprocessing.bulldozer_executor import mp_n_to_m_images
from bulldozer.eomultiprocessing.bulldozer_manager import BulldozerContextManager
from bulldozer.eomultiprocessing.utils import write
from bulldozer.utils.bulldozer_logger import BulldozerLogger, Runtime
from bulldozer.utils.helper import ubyte_profile_1bit


def fill_pits_filter(
    dtm: np.ndarray, border_mask: np.ndarray, filter_size: float, search_distance: int, nodata: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform pits removal and creates pits detection mask.

    Args:
        dtm: the DTM to process.
        border_mask: Border no data.
        filter_size: size of uniformer filter.
        search_distance: max serch distance for fillnodata.
        nodata: DSM nodata value.

    Returns:
        the processed dtm without pits and the pits mask
    """
    pits_mask = np.zeros(dtm.shape, dtype=np.ubyte)

    dtm[:] = fillnodata(dtm, mask=np.logical_not(border_mask), max_search_distance=search_distance)

    dtm_lf = ndimage.uniform_filter(dtm, size=filter_size)

    # Tags the pits
    pits_mask[dtm < dtm_lf] = 1
    pits_mask[border_mask == 1] = 0

    # Fill pits
    dtm[:] = np.where(pits_mask, dtm_lf, dtm)

    # Clean DTM
    dtm[border_mask == 1] = nodata

    return dtm, pits_mask


# TODO - rename function
@Runtime
def run(
    dtm_key: str | np.ndarray,
    border_nodata_mask_key: str | np.ndarray,
    dtm_profile: dict,
    nodata: float,
    manager: BulldozerContextManager,
) -> str | np.ndarray:
    """
    Performs the pit removal process using EOScale.

    Args:
        dtm_key: the DTM to process (numpy array or path to file).
        border_nodata_mask_key: Border no data (numpy array or path to file).
        dtm_profile: profile of the input DTM.
        nodata: DSM nodata value.
        manager: bulldozer context manager.

    Returns:
        The processed dtm
    """
    fill_pits_profile = ubyte_profile_1bit(dtm_profile)

    BulldozerLogger.log("Pits removal processing...", logging.INFO)

    filled_dtm_key: str | np.ndarray
    pits_mask_key: str | np.ndarray

    pits_mask_filename = "filled_pits.tif"
    fill_pits_parameters = {
        "filter_size": 35.5 / dtm_profile["transform"][0],
        "search_distance": 100,
        "nodata": nodata,
    }

    [filled_dtm_key, pits_mask_key] = mp_n_to_m_images(
        inputs=[dtm_key, border_nodata_mask_key],
        image_height=dtm_profile["height"],
        image_width=dtm_profile["width"],
        output_profiles=[dtm_profile, fill_pits_profile],
        output_keys=["filled_dtm.tif", pits_mask_filename],
        func=fill_pits_filter,
        func_parameters=fill_pits_parameters,
        context_manager=manager,
        stable_margin=int(fill_pits_parameters["filter_size"] / 2),
    )

    pits_mask_path = manager.get_path(pits_mask_filename, "mask")
    if isinstance(pits_mask_key, np.ndarray):
        write(pits_mask_key, pits_mask_path, fill_pits_profile)
    else:  # already saved in tmp folder
        shutil.move(pits_mask_key, pits_mask_path)

    return filled_dtm_key
