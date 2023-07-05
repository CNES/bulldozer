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
from bulldozer.utils.logging_helper import BulldozerLogger

# This dict store all the Bulldozer default parameters
DefaultValues = {
    # Basic parameters
    'GENERATE_DHM' : True,
    'MAX_OBJECT_WIDTH' : 16,
    # Advanced settings
    'OUTPUT_RESOLUTION' : None,
    'NODATA' : -32768.0,
    'MIN_VALID_HEIGHT' : None,
    'NB_MAX_WORKERS' : None,
    'CHECK_INTERSECTION' : False,
    'DEVELOPPER_MODE' : False,
    # Bulldozer core settings
    'FOUR_CONNEXITY' : True,
    'UNIFORM_FILTER_SIZE': 1,
    'PREVENT_UNHOOK_ITER' : 10,
    'NUM_OUTER_ITER' : 50,
    'NUM_INNER_ITER' : 10,
    'MP_TILE_SIZE' : 1500,
    'SLOPE_THRESHOLD' : 2,
    'KEEP_INTER_DTM' : False
}

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

def retrieve_nodata(dsm_path : str, cfg_nodata : str = None) -> float:
    """
    This method return the nodata value corresponding to the input DSM.
    The user can overrides the value existing in the metadata by providing a cfg_nodata value.

    Args:
        dsm_path: input DSM path.
        cfg_nodata: optional nodata value (overrides the value in the metadata).

    Returns:
        nodata value used in Bulldozer.
    """
    if cfg_nodata is not None :
        return float(cfg_nodata)
    
    # If nodata is not specified in the config file, retrieve the value from the DSM metadata
    with rasterio.open(dsm_path) as dsm_dataset:
        nodata = dsm_dataset.nodata
        if nodata is not None :
            return nodata
    
    BulldozerLogger.log("No data value is set to " + str(DefaultValues.NODATA.value), logging.INFO)
    # By default, if no value is set for nodata, return -32768.0
    return DefaultValues.NODATA.value

def downsample_profile(profile, factor : float) :
    
    transform= profile['transform']

    newprofile = profile.copy()
    dst_transform = Affine.translation(transform[2], transform[5]) * Affine.scale(transform[0]*factor, transform[4]*factor)
    
    newprofile.update({
    'transform': dst_transform,
    })
    
    return newprofile

def retrieve_raster_resolution(raster_dataset: rasterio.DatasetReader) -> float:
    """ """
    # We assume that resolution is the same wrt to both image axis
    res_x: float =  raster_dataset.transform[0]
    res_y: float = raster_dataset.transform[4]
    if abs(res_x) != abs(res_y):
        raise ValueError("Raster GSD must be the same wrt to the rows and columns.")
    return abs(res_x)

def write_tiles(tile_buffer: np.ndarray, 
                tile_path: str,
                original_profile: dict,
                tagLevel: int = None) -> None:
    """
    """
    tile_profile = original_profile
    tile_profile["count"] = 1
    tile_profile["width"] = tile_buffer.shape[1]
    tile_profile["height"] = tile_buffer.shape[0]
    tile_profile["dtype"] = np.float32
    with rasterio.open(tile_path, 'w', **tile_profile) as dst:
        if tagLevel is not None:
            dst.update_tags(minLevel = tagLevel)
        dst.write(tile_buffer, 1)

class Pyramid(object):

    def __init__(self, raster_path: str):
        self.raster_path = raster_path

        with rasterio.open(self.raster_path) as rasterDataset:
            self.initial_shape = (rasterDataset.height, rasterDataset.width)

    def shape(self, level:int = 0 ) -> tuple:
        """
            Compute downsampled shape
            @params:
                in_shape: shape to downsample
                level: pyramide level (0 = full resolution)
            @return:
                tuple containing the downsampled shape
        """
        
        factor=2**level

        height = self.initial_shape[0] // factor 
        width  = self.initial_shape[1] // factor
        
        if(self.initial_shape[0] % factor != 0):
            height = height + 1

        if(self.initial_shape[1] % factor != 0):
            width = width + 1
        
        return (height, width)
    
    def getArrayAtLevel(self, level: int = 0):
        """
            Compute the decimated shape and then
            use rasterio decimated method to get
            the output array
            https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.BufferedDatasetWriter.read
        """

        with rasterio.open(self.raster_path) as raster_dataset:
            if level == 0:
                    return raster_dataset.read(indexes=1)
            else:
                decimated_shape: tuple = self.shape(level = level)
                return raster_dataset.read(out_shape=decimated_shape, indexes=1)
        
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