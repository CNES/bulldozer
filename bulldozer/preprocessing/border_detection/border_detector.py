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
This module is used to detect border and inner nodata in the input DSM.
"""

import numpy as np
from scipy.ndimage import binary_fill_holes

from bulldozer.eomultiprocessing.bulldozer_executor import mp_n_to_m_images
from bulldozer.eomultiprocessing.bulldozer_manager import BulldozerContextManager
from bulldozer.eomultiprocessing.utils import read
from bulldozer.preprocessing import border  # type: ignore
from bulldozer.utils.bulldozer_logger import Runtime, logger
from bulldozer.utils.helper import ubyte_profile_1bit


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

    if do_transpose:
        # Vertical border nodata detection case
        border_nodata_mask = border_nodata.build_border_nodata_mask(dsm.T, nodata).T
    else:
        # Horizontal border nodata detection case
        border_nodata_mask = border_nodata.build_border_nodata_mask(dsm, nodata)

    return border_nodata_mask


@Runtime
def detect_border_nodata(
    dsm_key: str | np.ndarray,
    dsm_profile: dict,
    nodata: float,
    manager: BulldozerContextManager,
) -> tuple[str | np.ndarray, str]:
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
    nodata_mask_profile = ubyte_profile_1bit(dsm_profile)

    # Horizontal border nodata detection
    logger.info("Horizontal nodata mask processing...")
    hor_border_nodata_mask_key: str | np.ndarray

    [hor_border_nodata_mask_key] = mp_n_to_m_images(
        inputs=[dsm_key],
        image_height=dsm_profile["height"],
        image_width=dsm_profile["width"],
        output_profiles=[nodata_mask_profile],
        output_keys=["hor_border_nodata_mask.tif"],
        func=border_nodata_filter,
        func_parameters={"nodata": nodata, "do_transpose": False},
        context_manager=manager,
        stable_margin=0,
        tile_mode=False,
    )

    # Vertical border nodata detection
    logger.info("Vertical nodata mask processing...")
    ver_border_nodata_mask_key: str | np.ndarray

    [ver_border_nodata_mask_key] = mp_n_to_m_images(
        inputs=[dsm_key],
        image_height=dsm_profile["height"],
        image_width=dsm_profile["width"],
        output_profiles=[nodata_mask_profile],
        output_keys=["ver_border_nodata_mask.tif"],
        func=border_nodata_filter,
        func_parameters={"nodata": nodata, "do_transpose": True},
        context_manager=manager,
        stable_margin=0,
        tile_mode=False,
        strip_along_lines=True,
    )

    hor_border_nodata_mask = (
        read(hor_border_nodata_mask_key) if isinstance(hor_border_nodata_mask_key, str) else hor_border_nodata_mask_key
    )
    ver_border_nodata_mask = (
        read(ver_border_nodata_mask_key) if isinstance(ver_border_nodata_mask_key, str) else ver_border_nodata_mask_key
    )

    border_nodata_mask = np.logical_and(hor_border_nodata_mask, ver_border_nodata_mask)
    del hor_border_nodata_mask_key, ver_border_nodata_mask_key, hor_border_nodata_mask, ver_border_nodata_mask

    # Filling the holes inside the border nodata mask
    border_nodata_mask[:] = ~border_nodata_mask
    binary_fill_holes(border_nodata_mask, output=border_nodata_mask)
    border_nodata_mask[:] = ~border_nodata_mask

    border_nodata_mask_path = manager.write_tif(
        border_nodata_mask, "border_nodata.tif", nodata_mask_profile, key="mask"
    )

    # Inner nodata detection
    logger.info("Build Inner NoData Mask")
    dsm = read(dsm_key) if isinstance(dsm_key, str) else dsm_key

    inner_nodata_mask = np.logical_and(np.logical_not(border_nodata_mask), dsm == nodata)
    inner_nodata_mask_path = manager.write_tif(inner_nodata_mask, "inner_nodata.tif", nodata_mask_profile, key="mask")
    del inner_nodata_mask

    if not manager.in_memory:
        return border_nodata_mask_path, inner_nodata_mask_path

    return border_nodata_mask, inner_nodata_mask_path
