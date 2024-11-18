#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022-2025 Centre National d'Etudes Spatiales (CNES).
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
import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe
from bulldozer.utils.bulldozer_logger import Runtime


def ground_anchors_profile(input_profiles: list,
                           params: dict) -> dict:
    """
        This method is used in the main `detect_ground_anchors` 
        method to provide the output mask profile (binary profile).

        Args:
            input_profiles: input profile.
            params: extra parameters.
        
        Returns:
            updated profile.
    """
    return input_profiles[2]


def ground_anchors_filter(input_buffers: list,
                          input_profiles: list, 
                          filter_parameters: dict) -> np.ndarray:
    """ 
        This method is used in the main `detect_ground_anchors`.

    Args:
        input_buffers: input DSM.
        input_profiles: DSM profile.
        filter_parameters: dictionary containing the DTM altimetric uncertainty.
    
    Returns:
        ground anchors mask.
    """
    inter_dtm = input_buffers[0][0, :, :]
    dsm = input_buffers[1][0, :, :]
    regular_mask = input_buffers[2][0, :, :]
    ground_anchors_mask = np.where(np.logical_and(np.absolute(inter_dtm-dsm) <= filter_parameters["dsm_z_accuracy"],regular_mask), 1, 0).astype(np.uint8)
    return ground_anchors_mask


@Runtime
def detect_ground_anchors(intermediate_dtm_key: str,
                          dsm_key: str,
                          regular_mask_key: str,
                          dsm_z_accuracy: float,
                          eomanager: eom.EOContextManager) -> dict:
    """
        This method returns the binary mask flagging pre-detected ground areas location in the provided DSM.

        Args:
            intermediate_dtm_key: first estimation of the final DTM.
            dsm_key: input DSM.
            regular_mask_key: regular areas of the input DSM.
            dsm_z_accuracy: DSM altimetric resolution (in meter).
            eomanager: eoscale context manager.

        Returns:
            the regular areas mask.
    """
    ground_anchors_parameters: dict = {
        "dsm_z_accuracy": dsm_z_accuracy
    }
    
    [ground_anchors_mask_key] = eoexe.n_images_to_m_images_filter(inputs=[intermediate_dtm_key, dsm_key, regular_mask_key],
                                                              image_filter=ground_anchors_filter,
                                                              filter_parameters=ground_anchors_parameters,
                                                              generate_output_profiles=ground_anchors_profile,
                                                              context_manager=eomanager,
                                                              stable_margin=0,
                                                              filter_desc="Ground anchors mask processing...")

    return {
        "ground_anchors_mask_key": ground_anchors_mask_key
    }
