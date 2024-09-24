#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022 Centre National d'Etudes Spatiales (CNES).
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
    This module groups different generic methods used in Bulldozer.
"""
import time
import os
import rasterio
import logging
import numpy as np
from git import Repo
from rasterio import Affine
from bulldozer.utils.bulldozer_logger import BulldozerLogger

# - dsm_path: str (required)
# - output_dir: str (required)
# - nb_max_workers: int (optionnal, 8 by default)
# - dsm_z_precision: float (optional, 1.0 by default)
# - fill_search_radius: int (optional, 100 by default)
# - max_ground_slope: float (optional, 20.0 % by default)
# - min_object_spatial_frequency: float (optional, 0.0625 m^-1 by default)
# - cloth_tension_force: int (optional, 3 by default)
# - prevent_unhook_iter: int (optional, 10 by default)
# - num_outer_iter: int (optional, 100 by default)
# - num_inner_iter: int (optional, 10 by default)
# - output_resolution: float (optional, null by default)
# - generate_dhm: bool (optional, True by default)
# - developer_mode : bool (optional, False by default)
# - dtm_max_error: float (optional, 2 meters)
# - pre_anchor_points_activation : bool (optional, False by default)
# - post_anchor_points_activation : bool (optional, False by default)
# - reverse_drape_cloth_activation : bool (optional, False by default)


# This dict store all the Bulldozer default parameters
DefaultValues = {
    # Basic parameters
    'GENERATE_DHM': True,
    'MIN_OBJECT_SPATIAL_FREQUENCY': 0.0625,
    'NB_MAX_WORKERS': 8,
    # Advanced settings
    'DSM_Z_PRECISION': 1.0,
    'DEVELOPER_MODE': False,
    'FILL_SEARCH_RADIUS': 100,
    'MAX_GROUND_SLOPE': 20.0,
    'DTM_MAX_ERROR': None,
    'PRE_ANCHOR_POINTS_ACTIVATION': False,
    'POST_ANCHOR_POINTS_ACTIVATION': False,
    'REVERSE_DRAPE_CLOTH_ACTIVATION': False,
    # Bulldozer core settings
    'CLOTH_TENSION_FORCE': 3,
    'PREVENT_UNHOOK_ITER': 10,
    'NUM_OUTER_ITER': 50,
    'NUM_INNER_ITER': 10,
}


def npAsContiguousArray(arr: np.array) -> np.array:
    """
    This method checks that the input array is contiguous. 
    If not, returns the contiguous version of the input numpy array.

    Args:
        arr: input array.

    Returns:
        contiguous array usable in C++.
    """
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr


def downsample_profile(profile, factor: float):
    
    transform= profile['transform']

    newprofile = profile.copy()
    dst_transform = Affine.translation(transform[2], transform[5]) * Affine.scale(transform[0]*factor, transform[4]*factor)
    
    newprofile.update({
        'transform': dst_transform,
    })
    
    return newprofile


class Runtime:
    """
    This class is used as decorator to monitor the runtime .
    """
    
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        func_start = time.perf_counter()
        result = self.function(*args, **kwargs)
        func_end = time.perf_counter()
        # %(module)s - %(funcName)s
        BulldozerLogger.log("{}-{}: Done (Runtime: {}s)".format(self.function.__module__, self.function.__name__, round(func_end-func_start,2)), logging.INFO)
        return result