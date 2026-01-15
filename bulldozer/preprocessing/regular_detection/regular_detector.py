#!/usr/bin/env python
# coding: utf8
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
import shutil
from typing import Union

import numpy as np
import rasterio
from scipy.ndimage import binary_opening

from bulldozer.multiprocessing.bulldozer_executor import mp_n_to_m_images
from bulldozer.multiprocessing.bulldozer_manager import BulldozerContextManager
from bulldozer.multiprocessing.utils import write
from bulldozer.preprocessing import regular  # type: ignore
from bulldozer.utils.bulldozer_logger import BulldozerLogger, Runtime
from bulldozer.utils.helper import ubyte_profile


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

    return reg_mask.astype(np.ubyte)


@Runtime
def detect_regular_areas(
    dsm_key: Union[str, np.ndarray],
    dsm_profile: dict,
    regular_slope: float,
    nodata: float,
    max_object_size: int,
    manager: BulldozerContextManager,
    reg_filtering_iter: Union[int, None] = None,
) -> Union[str, np.ndarray]:
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
    regular_mask_profile = ubyte_profile(dsm_profile)

    BulldozerLogger.log("Raw regular mask processing...", logging.INFO)

    regular_mask_key: Union[str, np.ndarray]
    raw_regular_mask_filename = "raw_regular_mask.tif"
    regular_parameters = {"regular_slope": regular_slope, "nodata": nodata}

    if manager.pool is None:
        # no multiprocessing
        if isinstance(dsm_key, str):
            raise ValueError("Without multiprocessing the input dsm must be a numpy array.")
        regular_mask_key = regular_mask_filter(dsm_key, **regular_parameters)
    else:
        # multiprocessing
        [regular_mask_key] = mp_n_to_m_images(
            inputs=[dsm_key],
            image_height=dsm_profile["height"],
            image_width=dsm_profile["width"],
            output_profiles=[regular_mask_profile],
            output_keys=[raw_regular_mask_filename],
            func=regular_mask_filter,
            func_parameters=regular_parameters,
            stable_margin=1,
            context_manager=manager,
            binary=True,
        )

    if manager.dev_mode:
        raw_regular_path = manager.get_path(raw_regular_mask_filename, "dev")
        if isinstance(regular_mask_key, np.ndarray):
            write(regular_mask_key, raw_regular_path, regular_mask_profile, binary=True)
        else:  # already saved in tmp folder
            shutil.move(regular_mask_key, raw_regular_path)
            regular_mask_key = raw_regular_path

    BulldozerLogger.log("Regular mask processing...", logging.INFO)

    bin_regular_mask: np.ndarray
    bin_regular_mask_filename = "regular_mask.tif"

    if isinstance(regular_mask_key, np.ndarray):
        bin_regular_mask = regular_mask_key.astype(bool)
    else:
        with rasterio.open(regular_mask_key) as src:
            bin_regular_mask = src.read(1).astype(bool)
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

    if not manager.in_memory:
        key = "dev" if manager.dev_mode else "tmp"
        regular_mask_path = manager.get_path(bin_regular_mask_filename, key=key)
        write(bin_regular_mask, regular_mask_path, regular_mask_profile, binary=True)
        return regular_mask_path

    # else in memory
    if manager.dev_mode:
        regular_mask_path = manager.get_path(bin_regular_mask_filename, key="dev")
        write(bin_regular_mask, regular_mask_path, regular_mask_profile, binary=True)

    return bin_regular_mask
