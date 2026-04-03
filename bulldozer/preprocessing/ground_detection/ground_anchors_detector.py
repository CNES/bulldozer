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
This module is used to detect ground anchors points before the main DTM extraction step.
"""

import numpy as np

from bulldozer.eomultiprocessing.bulldozer_executor import mp_n_to_m_images
from bulldozer.eomultiprocessing.bulldozer_manager import BulldozerContextManager
from bulldozer.eomultiprocessing.utils import read
from bulldozer.utils.bulldozer_logger import Runtime, logger
from bulldozer.utils.helper import ubyte_profile_1bit


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

    ground_anchors_mask = np.logical_and(
        np.absolute(inter_dtm - dsm) <= dsm_z_accuracy,
        regular_mask,
    ).astype(np.uint8)

    return ground_anchors_mask


@Runtime
def detect_ground_anchors(
    intermediate_dtm_key: str | np.ndarray,
    dsm_key: str | np.ndarray,
    regular_mask_key: str | np.ndarray,
    dsm_profile: dict,
    dsm_z_accuracy: float,
    manager: BulldozerContextManager,
    ground_mask_path: str | None = None,
) -> str | np.ndarray:
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
    ground_anchors_profile = ubyte_profile_1bit(dsm_profile)

    logger.info("Ground anchors mask processing...")

    ground_anchors_mask_key: str | np.ndarray
    [ground_anchors_mask_key] = mp_n_to_m_images(
        inputs=[intermediate_dtm_key, dsm_key, regular_mask_key],
        image_height=dsm_profile["height"],
        image_width=dsm_profile["width"],
        output_profiles=[ground_anchors_profile],
        output_keys=["ground_anchors_mask.tif"],
        func=ground_anchors_filter,
        func_parameters={"dsm_z_accuracy": dsm_z_accuracy},
        context_manager=manager,
        stable_margin=0,
        debug=True,
    )

    # Union of detected ground anchors mask with provided ground_mask
    if ground_mask_path is not None:
        logger.info("Ground mask processing...")

        ground_anchors_mask = (
            read(ground_anchors_mask_key) if isinstance(ground_anchors_mask_key, str) else ground_anchors_mask_key
        )
        ground_mask = read(ground_mask_path)

        ground_anchors_mask[:] = np.logical_or(ground_anchors_mask, ground_mask)

        anchorage_mask_with_ground_filename = "anchorage_mask_with_ground.tif"

        if not manager.in_memory:
            anchorage_mask_with_ground_path = manager.write_tif(
                ground_anchors_mask, anchorage_mask_with_ground_filename, ground_anchors_profile
            )
            ground_anchors_mask_key = anchorage_mask_with_ground_path
        else:
            if manager.dev_mode:
                manager.write_tif(ground_anchors_mask, anchorage_mask_with_ground_filename, ground_anchors_profile)
            ground_anchors_mask_key = ground_anchors_mask

    return ground_anchors_mask_key
