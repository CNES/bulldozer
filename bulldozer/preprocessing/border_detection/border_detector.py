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
This module is used to detect border and inner nodata in the input DSM.
"""

import logging
from typing import Tuple, Union

import numpy as np
import rasterio
from scipy.ndimage import binary_fill_holes

from bulldozer.multiprocessing.bulldozer_executor import mp_n_to_m_images
from bulldozer.multiprocessing.bulldozer_manager import BulldozerContextManager
from bulldozer.multiprocessing.utils import write
from bulldozer.preprocessing import border  # type: ignore
from bulldozer.utils.bulldozer_logger import BulldozerLogger, Runtime
from bulldozer.utils.helper import ubyte_profile


def border_nodata_filter(dsm: np.ndarray, nodata: float, do_transpose: bool = False) -> np.ndarray:
    """
        This method is used in the main `detect_border_nodata` method.
        It calls the Cython method to extract border nodata along a vertical or horizontal axis.

    Args:
        dsm: input DSM.
        nodata: DSM nodata value (if nan, the nodata is set to default value: -32768.0).
        do_transpose: axis for the detection (True: vertical or False: horizontal).

    Returns:
        border nodata mask along specified axis.
    """
    border_nodata = border.PyBorderNodata()

    dsm = dsm.reshape((1, dsm.shape[0], dsm.shape[1]))

    if do_transpose:
        # Vertical border nodata detection case
        border_nodata_mask = border_nodata.build_border_nodata_mask(dsm.T, nodata, True).astype(np.ubyte).T
    else:
        # Horizontal border nodata detection case
        border_nodata_mask = border_nodata.build_border_nodata_mask(dsm, nodata, False).astype(np.ubyte)

    return border_nodata_mask


@Runtime
def detect_border_nodata(
    dsm_key: Union[str, np.ndarray],
    dsm_profile: dict,
    nodata: float,
    manager: BulldozerContextManager,
) -> Tuple[Union[str, np.ndarray], Union[str, np.ndarray]]:
    """
    This method returns the binary masks flagging the border and inner nodata.
    The border nodata corresponds to the nodata points on the edges if the DSM is skewed and the inner nodata
    corresponds to the other nodata points.

    Args:
        dsm_key: input DSM (numpy array or path to file).
        dsm_profile: profile of the input DSM.
        nodata: DSM nodata value (if nan, the nodata is set to default value: -32768.0).
        manager: bulldozer context manager.

    Returns:
        border and inner nodata masks.
    """
    nodata_mask_profile = ubyte_profile(dsm_profile)

    # Horizontal border nodata detection
    BulldozerLogger.log("Horizontal nodata mask processing...", logging.INFO)
    hor_border_nodata_mask_key: Union[str, np.ndarray]
    hor_border_nodata_mask_filename = "hor_border_nodata_mask.tif"
    if manager.pool is None:
        # no multiprocessing
        if isinstance(dsm_key, str):
            raise ValueError("Without multiprocessing the input DSM must be a numpy array.")
        hor_border_nodata_mask_key = border_nodata_filter(dsm_key, nodata)
    else:
        # multiprocessing
        [hor_border_nodata_mask_key] = mp_n_to_m_images(
            inputs=[dsm_key],
            image_height=dsm_profile["height"],
            image_width=dsm_profile["width"],
            output_profiles=[nodata_mask_profile],
            output_keys=[hor_border_nodata_mask_filename],
            func=border_nodata_filter,
            func_parameters={"nodata": nodata, "do_transpose": False},
            context_manager=manager,
            stable_margin=0,
            tile_mode=False,
        )

    # Vertical border nodata detection
    BulldozerLogger.log("Vertical nodata mask processing...", logging.INFO)
    ver_border_nodata_mask_key: Union[str, np.ndarray]
    ver_border_nodata_mask_filename = "ver_border_nodata_mask.tif"
    if manager.pool is None:
        # no multiprocessing
        if isinstance(dsm_key, str):
            raise ValueError("Without multiprocessing the input DSM must be a numpy array.")
        ver_border_nodata_mask_key = border_nodata_filter(dsm_key, nodata, do_transpose=True)
    else:
        # multiprocessing
        [ver_border_nodata_mask_key] = mp_n_to_m_images(
            inputs=[dsm_key],
            image_height=dsm_profile["height"],
            image_width=dsm_profile["width"],
            output_profiles=[nodata_mask_profile],
            output_keys=[ver_border_nodata_mask_filename],
            func=border_nodata_filter,
            func_parameters={"nodata": nodata, "do_transpose": True},
            context_manager=manager,
            stable_margin=0,
            tile_mode=False,
            strip_along_lines=True,
        )

    if manager.in_memory:
        border_nodata_mask = np.logical_and(hor_border_nodata_mask_key, ver_border_nodata_mask_key)
    else:
        with rasterio.open(hor_border_nodata_mask_key) as hor_mask:
            with rasterio.open(ver_border_nodata_mask_key) as ver_mask:
                border_nodata_mask = np.logical_and(hor_mask.read(1), ver_mask.read(1))
    del hor_border_nodata_mask_key, ver_border_nodata_mask_key

    # Filling the holes inside the border nodata mask
    border_nodata_mask = np.where(border_nodata_mask == 0, 1, 0).astype(np.uint8)
    binary_fill_holes(border_nodata_mask, output=border_nodata_mask)
    border_nodata_mask = np.where(border_nodata_mask == 0, 1, 0)

    border_nodata_mask_path = manager.get_path("border_nodata.tif", key="mask")
    write(border_nodata_mask, border_nodata_mask_path, nodata_mask_profile, binary=True)

    # Inner nodata detection
    BulldozerLogger.log("Build Inner NoData Mask", logging.INFO)
    if isinstance(dsm_key, str):
        with rasterio.open(dsm_key) as src:
            dsm = src.read(1)
    else:
        dsm = dsm_key

    inner_nodata_mask_path = manager.get_path("inner_nodata.tif", "mask")
    inner_nodata_mask = np.logical_and(np.logical_not(border_nodata_mask), dsm == nodata)
    write(inner_nodata_mask, inner_nodata_mask_path, nodata_mask_profile, binary=True)

    if not manager.in_memory:
        return border_nodata_mask_path, inner_nodata_mask_path

    return border_nodata_mask, inner_nodata_mask
