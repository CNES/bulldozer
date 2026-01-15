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
This module is used to detect ground anchors points before the main DTM extraction step.
"""

import logging
import shutil
from typing import Union

import numpy as np
import rasterio

from bulldozer.multiprocessing.bulldozer_executor import mp_n_to_m_images
from bulldozer.multiprocessing.bulldozer_manager import BulldozerContextManager
from bulldozer.multiprocessing.utils import write
from bulldozer.utils.bulldozer_logger import BulldozerLogger, Runtime
from bulldozer.utils.helper import ubyte_profile


def ground_anchors_filter(
    inter_dtm: np.ndarray,
    dsm: np.ndarray,
    regular_mask: np.ndarray,
    dsm_z_accuracy: float,
) -> np.ndarray:
    """
        This method is used in the main `detect_ground_anchors`.

    Args:
        inter_dtm: first estimation of the final DTM.
        dsm: input DSM.
        regular_mask: regular areas of the input DSM.
        dsm_z_accuracy: DSM altimetric resolution (in meter).

    Returns:
        ground anchors mask.
    """
    ground_anchors_mask = np.where(
        np.logical_and(np.absolute(inter_dtm - dsm) <= dsm_z_accuracy, regular_mask),
        1,
        0,
    ).astype(np.uint8)

    return ground_anchors_mask


@Runtime
def detect_ground_anchors(
    intermediate_dtm_key: Union[str, np.ndarray],
    dsm_key: Union[str, np.ndarray],
    regular_mask_key: Union[str, np.ndarray],
    dsm_profile: dict,
    dsm_z_accuracy: float,
    manager: BulldozerContextManager,
    ground_mask_path: Union[str, None] = None,
) -> Union[str, np.ndarray]:
    """
    This method returns the binary mask flagging pre-detected ground areas location in the provided DSM.

    Args:
        intermediate_dtm_key: first estimation of the final DTM (numpy array or path to file).
        dsm_key: input DSM (numpy array or path to file).
        regular_mask_key: regular areas of the input DSM (numpy array or path to file).
        ground_mask_path: ground mask path
        dsm_profile: profile of the input DSM.
        dsm_z_accuracy: DSM altimetric resolution (in meter).
        manager: bulldozer context manager.

    Returns:
        the regular areas mask.
    """
    ground_anchors_profile = ubyte_profile(dsm_profile)

    BulldozerLogger.log("Ground anchors mask processing...", logging.INFO)
    ground_anchors_mask_key: Union[str, np.ndarray]
    ground_anchors_mask_filename = "ground_anchors_mask.tif"
    ground_anchors_parameters = {"dsm_z_accuracy": dsm_z_accuracy}
    if manager.pool is None:
        # no multiprocessing
        if isinstance(intermediate_dtm_key, str) or isinstance(dsm_key, str) or isinstance(regular_mask_key, str):
            raise ValueError("Without multiprocessing the inputs must be numpy arrays.")
        ground_anchors_mask_key = ground_anchors_filter(
            intermediate_dtm_key, dsm_key, regular_mask_key, **ground_anchors_parameters
        )
    else:
        # multiprocessing
        [ground_anchors_mask_key] = mp_n_to_m_images(
            inputs=[intermediate_dtm_key, dsm_key, regular_mask_key],
            image_height=dsm_profile["height"],
            image_width=dsm_profile["width"],
            output_profiles=[ground_anchors_profile],
            output_keys=[ground_anchors_mask_filename],
            func=ground_anchors_filter,
            func_parameters=ground_anchors_parameters,
            context_manager=manager,
            stable_margin=0,
        )

    if manager.dev_mode:
        ground_anchors_mask_path = manager.get_path(ground_anchors_mask_filename, "dev")
        if isinstance(ground_anchors_mask_key, np.ndarray):
            write(ground_anchors_mask_key, ground_anchors_mask_path, ground_anchors_profile)
        else:  # already saved in tmp folder
            shutil.move(ground_anchors_mask_key, ground_anchors_mask_path)
            ground_anchors_mask_key = ground_anchors_mask_path

    # Union of detected ground anchors mask with provided ground_mask
    if ground_mask_path is not None:
        BulldozerLogger.log("Ground mask processing...", logging.INFO)

        if isinstance(ground_anchors_mask_key, str):
            with rasterio.open(ground_anchors_mask_key) as src:
                ground_anchors_mask = src.read(1)
        else:
            ground_anchors_mask = ground_anchors_mask_key

        with rasterio.open(ground_mask_path) as src:
            ground_mask = src.read(1)

        np.logical_or(ground_anchors_mask, ground_mask, out=ground_anchors_mask)

        anchorage_mask_with_ground_filename = "anchorage_mask_with_ground.tif"
        if manager.in_memory:
            ground_anchors_mask_key = ground_anchors_mask
            if manager.dev_mode:
                anchorage_mask_with_ground_path = manager.get_path(anchorage_mask_with_ground_filename, key="dev")
                write(ground_anchors_mask, anchorage_mask_with_ground_path, ground_anchors_profile)
        else:
            key = "dev" if manager.dev_mode else "tmp"
            anchorage_mask_with_ground_path = manager.get_path(anchorage_mask_with_ground_filename, key=key)
            write(ground_anchors_mask, anchorage_mask_with_ground_path, ground_anchors_profile)
            ground_anchors_mask_key = anchorage_mask_with_ground_path

    return ground_anchors_mask_key
