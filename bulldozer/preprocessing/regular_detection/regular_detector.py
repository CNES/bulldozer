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
This module is used to extract the regular areas in the provided DSM.
"""

import logging

import numpy as np
from scipy.ndimage import binary_opening

from bulldozer.eomultiprocessing.bulldozer_executor import mp_n_to_m_images
from bulldozer.eomultiprocessing.bulldozer_manager import BulldozerContextManager
from bulldozer.eomultiprocessing.utils import read
from bulldozer.preprocessing import regular  # type: ignore
from bulldozer.utils.bulldozer_logger import BulldozerLogger, Runtime
from bulldozer.utils.helper import ubyte_profile_1bit


def regular_mask_filter(dsm: np.ndarray, regular_slope: float, nodata: float) -> np.ndarray:
    """
    This method is used in the main `detect_regular_areas`.
    It calls the Cython method to extract regular areas.

    Args:
        dsm: input DSM.
        regular_slope: maximum slope of a regular area.
        nodata: DSM nodata value (if nan, the nodata is set to -32768).

    Returns:
        regular areas mask.
    """
    reg_filter = regular.PyRegularAreas()
    reg_mask = reg_filter.build_regular_mask(dsm, regular_slope, nodata)

    return reg_mask


@Runtime
def detect_regular_areas(
    dsm_key: str | np.ndarray,
    dsm_profile: dict,
    regular_slope: float,
    nodata: float,
    max_object_size: int,
    manager: BulldozerContextManager,
    reg_filtering_iter: int | None = None,
) -> str | np.ndarray:
    """
    This method returns the binary mask flagging regular areas location in the provided DSM.

    Args:
        dsm_key: input DSM (numpy array or path to file).
        dsm_profile: profile of the input DSM.
        regular_slope: maximum slope of a regular area.
        nodata: DSM nodata value (if nan, the nodata is set to -32768).
        max_object_size: foreground max object size (in meter).
        reg_filtering_iter:  number of regular mask filtering iterations.
        manager: bulldozer context manager.

    Returns:
        the regular areas mask.
    """
    regular_mask_profile = ubyte_profile_1bit(dsm_profile)

    BulldozerLogger.log("Raw regular mask processing...", logging.INFO)

    regular_mask_key: str | np.ndarray

    [regular_mask_key] = mp_n_to_m_images(
        inputs=[dsm_key],
        image_height=dsm_profile["height"],
        image_width=dsm_profile["width"],
        output_profiles=[regular_mask_profile],
        output_keys=["raw_regular_mask.tif"],
        func=regular_mask_filter,
        func_parameters={"regular_slope": regular_slope, "nodata": nodata},
        stable_margin=1,
        context_manager=manager,
        debug=True,
    )

    BulldozerLogger.log("Regular mask processing...", logging.INFO)

    bin_regular_mask: np.ndarray
    bin_regular_mask_filename = "regular_mask.tif"

    bin_regular_mask = (
        read(regular_mask_key).astype(bool) if isinstance(regular_mask_key, str) else regular_mask_key.astype(bool)
    )
    del regular_mask_key

    if reg_filtering_iter is not None:
        nb_iterations = reg_filtering_iter
    else:
        nb_iterations = int(np.max([1, max_object_size / 4]))
        BulldozerLogger.log(
            f"'reg_filtering_iter' is not set or less than 1. Used default computed value: {nb_iterations}",
            logging.DEBUG,
        )

    # This condition allows the user to deactivate the filtering
    # (iterations=0 in binary_opening ends up to filtering until nothing change)
    if nb_iterations >= 1:
        binary_opening(bin_regular_mask, iterations=nb_iterations, output=bin_regular_mask)

    regular_mask_path = manager.write_tif(bin_regular_mask, bin_regular_mask_filename, regular_mask_profile, key="mask")

    if not manager.in_memory:
        return regular_mask_path

    return bin_regular_mask
