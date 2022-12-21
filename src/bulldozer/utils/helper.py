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
import platform
import psutil
import time
import os
import getpass
import rasterio
import logging
import numpy as np
from git import Repo
from git.exc import InvalidGitRepositoryError
from bulldozer.utils.logging_helper import BulldozerLogger


def write_dataset(buffer_path : str, 
                  buffer : np.ndarray, 
                  profile : rasterio.profiles.Profile,
                  window : rasterio.windows.Window = None,
                  band : int = 1) -> None:
    """
        This method allows to write a TIFF file based on the input buffer.

        Args:
            buffer_path: path to the output file.
            buffer: dataset to write.
            profile: destination dataset profile (driver, crs, transform, etc.).
            band: index of the target band.
    """
    profile['driver'] = 'GTiff'
    with rasterio.open(buffer_path, 'w', **profile) as dst_dataset:
        dst_dataset.write(buffer, band)
        if window:
            dst_dataset.write(buffer, band, window)
        dst_dataset.close()

def npAsContiguousArray(arr : np.array) -> np.array:
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

def retrieve_nodata(dsm_path : str, cfg_nodata : float = None) -> float:
    """
    This method return the nodata value corresponding to the input DSM.
    The user can overrides the value existing in the metadata by providing a cfg_nodata value.

    Args:
        dsm_path: input DSM path.
        cfg_nodata: optional nodata value (overrides the value in the metadata).

    Returns:
        nodata value used in Bulldozer.
    """
    if cfg_nodata :
        return cfg_nodata
    
    # If nodata is not specified in the config file, retrieve the value from the DSM metadata
    with rasterio.open(dsm_path) as dsm_dataset:
        nodata = dsm_dataset.nodata
        return nodata
    
    # By default, if no value is set for nodata, return None
    return None
        
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
        BulldozerLogger.log("{}: Done (Runtime: {}s)".format(self.function.__name__, round(func_end-func_start,2)), logging.INFO)
        return result