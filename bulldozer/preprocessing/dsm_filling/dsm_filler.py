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
This module is used to prefill the input DSM before the DTM extraction.
"""
import numpy as np
import rasterio
import os

from rasterio import Affine
from scipy.ndimage import zoom

import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe
import bulldozer.preprocessing.fill as fill
from bulldozer.utils.bulldozer_logger import Runtime


def filled_dsm_profile(input_profiles: list,
                       params: dict) -> dict:
    """
        This method is used to provide the output filled dsm profile.
        
        Args:
            input_profiles: input profile.
            params: extra parameters.

        Returns:
            updated profile.
    """
    return [input_profiles[0]]


def fill_dsm_method(input_buffers: list,
                    input_profiles: list, 
                    filter_parameters: dict) -> np.ndarray:
    """
        This method is used in the main `fill_dsm_process`.
        It calls the Cython method to fill a DMS.

        Args:
            input_buffers: input DSM, input regular mask.
            input_profiles: DSM profile.
            filter_parameters: filter parameters.
        
        Returns:
            regular areas mask.
    """
    fill_process = fill.PyFill()
    # the input_buffers[0] corresponds to the input DSM raster
    dsm = fill_process.iterative_filling(input_buffers[0][0, :, :],
                                         input_buffers[1][0, :, :],
                                         nodata_value=filter_parameters["nodata"],
                                         nb_it=filter_parameters["it"])
    
    return [dsm.astype(np.float32)]


def downsample_profile(profile, factor : float) :

    transform = profile['transform']

    newprofile = profile.copy()
    dst_transform = Affine.translation(transform[2], transform[5]) * Affine.scale(transform[0]*factor, transform[4]*factor)

    newprofile.update({
        'transform': dst_transform,
    })
    
    return newprofile


# def next_power_of_2(x: int) -> int:
#     """
#     This function returns the smallest power of 2 that is greater than or equal to a given non-negative integer x.

#     Args:
#         x : non negative integer.

#     Returns:
#         the corresponding power index power (2**index >= x).
#     """
#     return 0 if x == 0 else (1 << (x-1).bit_length()).bit_length() - 1


# def get_max_pyramid_level(max_object_size_pixels: float) -> int :
#     """ 
#         Given the max size of an object on the ground,
#         this method computes the max level of the pyramid
#         for drape cloth algorithm
#     """
#     power = next_power_of_2(int(max_object_size_pixels))
    
#     # Take the closest power to the max object size
#     if abs(2**(power-1) - max_object_size_pixels) < abs(2**power - max_object_size_pixels):
#         power -= 1

#     return power


@Runtime
def fill_dsm(dsm_key: str,
             regular_key: str,
             border_nodata_key: str,
             unfilled_dsm_mask_key :str,
             nodata: float,
             max_object_size: int,
             eomanager: eom.EOContextManager,
             dev_mode: bool,
             dev_dir: str = '') -> dict:
    """
    This fills the nodata of the input DSM for the following dtm extraction step.

    Args:
        dsm_key: input DSM.
        regular_key: regular mask.
        border_nodata_key: border nodata mask.
        nodata: DSM nodata value (if nan, the nodata is set to -32768).
        eomanager: eoscale context manager.
        dev_mode: if True, dev mode activated
        dev_dir: path to save dev files

    Returns:
        the filled DSM.
    """

    regular_parameters: dict = {
        "nodata": nodata,
        "it": max_object_size
    }
    
    dsm_profile = eomanager.get_profile(key=dsm_key)
    
    dsm_resolution: float = dsm_profile["transform"][0]

    # Determine max object size in pixels
    # max_object_size_pixels = max_object_size / dsm_re-solution

    # Determine the dezoom factor wrt to max size of an object
    # on the ground.
    # nb_levels = get_max_pyramid_level(max_object_size_pixels/2) + 1
    nb_levels = 100
    
    filled_dsm = eomanager.get_array(key=dsm_key)[0]
    regular = eomanager.get_array(key=regular_key)[0]
    
    # We are also filling the irregular areas
    filled_dsm[regular==0] = nodata
    
    if dev_mode:
            filled_dsm_with_regular_path: str = os.path.join(dev_dir, "filled_dsm_with_regular.tif")
            eomanager.write(key=dsm_key, img_path=filled_dsm_with_regular_path)
    
    # First iterative filling for small no data areas
    [dsm_key] = eoexe.n_images_to_m_images_filter(inputs=[dsm_key, border_nodata_key],
                                                  image_filter=fill_dsm_method,
                                                  filter_parameters=regular_parameters,
                                                  generate_output_profiles=filled_dsm_profile,
                                                  context_manager=eomanager,
                                                  stable_margin=regular_parameters['it'],
                                                  filter_desc="Iterative filling DSM - 1st pass") 
    
    if dev_mode:
            filled_dsm_1stpass_path: str = os.path.join(dev_dir, "filled_dsm_1stpass.tif")
            eomanager.write(key=dsm_key, img_path=filled_dsm_1stpass_path)
    
    filled_dsm = eomanager.get_array(key=dsm_key)[0]
    border_nodata = eomanager.get_array(key=border_nodata_key)[0]
    
    # Identifying the remaining inner no data areas
    remaining_nodata = (filled_dsm == nodata) & (border_nodata == 0)
    
    # Putting nan values instead of no data for the sampling function
    filled_dsm[filled_dsm == nodata] = np.nan  

    # if nodata areas are still in the DSM
    has_nodata = np.any(remaining_nodata)
    
    if has_nodata:

        # Downsampling the DSM to fill the bigger nodata areas
        filled_dsm_downsampled = zoom(filled_dsm, 1/nb_levels, order=1, mode='nearest') 
        filled_dsm_downsampled = np.where(np.isnan(filled_dsm_downsampled), nodata, filled_dsm_downsampled)
        
        # Creating new profile for downsampled data
        downsampled_profile = downsample_profile(profile=eomanager.get_profile(key=dsm_key), factor=nb_levels)
        downsampled_profile.update(width=np.shape(filled_dsm_downsampled)[1], height=np.shape(filled_dsm_downsampled)[0])
        downsampled_filled_dsm_key = eomanager.create_image(downsampled_profile)
        
        filled_dsm_downsample = eomanager.get_array(key=downsampled_filled_dsm_key)[0]
        filled_dsm_downsample[:] = filled_dsm_downsampled
        
        if dev_mode:
            filled_dsm_downsample_path: str = os.path.join(dev_dir, "filled_dsm_downsample.tif")
            eomanager.write(key=downsampled_filled_dsm_key, img_path=filled_dsm_downsample_path)

        # Updating the profile for the bordernodata mask
        downsampled_profile['dtype'] = np.uint8
        downsampled_profile['nodata'] = None

        # Downsampling the bordernodata mask
        downsampled_border_nodata_key = eomanager.create_image(downsampled_profile)
        border_nodata_downsample = eomanager.get_array(key=downsampled_border_nodata_key)[0]
        border_nodata_downsampled = zoom(border_nodata, 1/nb_levels, order=1, mode='nearest')
        border_nodata_downsample[:] = border_nodata_downsampled   
        
        if dev_mode:
            border_nodata_downsample_path: str = os.path.join(dev_dir, "border_nodata_downsample.tif")
            eomanager.write(key=downsampled_border_nodata_key, img_path=border_nodata_downsample_path, binary=True)
        
        regular_parameters: dict = {
        "nodata": nodata,
        "it": 1000
        }
        
        # Second iterative filling for the remaining no data areas
        [downsampled_filled_dsm_key] = eoexe.n_images_to_m_images_filter(inputs=[downsampled_filled_dsm_key, downsampled_border_nodata_key],
                                                                         image_filter=fill_dsm_method,
                                                                         filter_parameters=regular_parameters,
                                                                         generate_output_profiles=filled_dsm_profile,
                                                                         context_manager=eomanager,
                                                                         stable_margin=regular_parameters['it'],
                                                                         filter_desc="Iterative filling DSM - 2nd pass") 
                    
        filled_dsm_downsample = eomanager.get_array(key=downsampled_filled_dsm_key)[0]
        
        # Putting nan values instead of no data for the sampling function
        filled_dsm_downsample[filled_dsm_downsample == nodata] = np.nan
        
        if dev_mode:
            filled_dsm_downsample_2ndpass_path: str = os.path.join(dev_dir, "filled_dsm_downsample_2ndpass.tif")
            eomanager.write(key=downsampled_filled_dsm_key, img_path=filled_dsm_downsample_2ndpass_path)
        
        # Merging the second filling with the first one
        scale_y = filled_dsm.shape[0] / filled_dsm_downsample.shape[0]
        scale_x = filled_dsm.shape[1] / filled_dsm_downsample.shape[1]
        filled_dsm_resample = zoom(filled_dsm_downsample, (scale_y, scale_x), order=1, mode='nearest')

        filled_dsm[:] = np.where(remaining_nodata == 1, filled_dsm_resample, filled_dsm)
        
        eomanager.release(key=downsampled_filled_dsm_key)
        eomanager.release(key=downsampled_border_nodata_key)


    
    #TODO - HOTFIX to remove
    unfilled_dsm_mask = eomanager.get_array(key=unfilled_dsm_mask_key)[0]
    unfilled_dsm_mask[np.isnan(filled_dsm)] = 1
    filled_dsm[np.isnan(filled_dsm)] = 9999

    return {
        "filled_dsm" : dsm_key,
        "unfilled_dsm_mask_key" : unfilled_dsm_mask_key
    }