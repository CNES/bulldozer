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
import logging
import os
import math

from rasterio import Affine
from scipy.ndimage import zoom, binary_erosion

import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe
import bulldozer.preprocessing.fill as fill
from bulldozer.utils.bulldozer_logger import Runtime, BulldozerLogger


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
    
    if filter_parameters["use_bordernodata_mask"]==True:
        # the input_buffers[0] corresponds to the input DSM raster, input_buffers[1] corresponds to the bordernodata mask
        dsm = fill_process.iterative_filling(dsm_strip=input_buffers[0][0, :, :],
                                            nodata_value=filter_parameters["nodata"],
                                            nb_it=filter_parameters["filling_iterations"],
                                            border_nodata_strip=input_buffers[1][0, :, :])
    else:
        dsm = fill_process.iterative_filling(input_buffers[0][0, :, :],
                                            nodata_value=filter_parameters["nodata"],
                                            nb_it=filter_parameters["filling_iterations"])
        
    return [dsm.astype(np.float32)]


def downsample_profile(profile, factor : float) :

    transform = profile["transform"]

    newprofile = profile.copy()
    dst_transform = Affine.translation(transform[2], transform[5]) * Affine.scale(transform[0]*factor, transform[4]*factor)

    newprofile.update({
        "transform": dst_transform,
    })
    
    return newprofile


@Runtime
def fill_dsm(dsm_key: str,
             regular_key: str,
             border_nodata_key: str,
             nodata: float,
             max_object_size: int,
             eomanager: eom.EOContextManager,
             dev_mode: bool = False,
             dev_dir: str = "") -> dict:
    """
    This fills the nodata of the input DSM for the following dtm extraction step.

    Args:
        dsm_key: input DSM.
        regular_key: regular mask.
        border_nodata_key: border nodata mask.
        nodata: DSM nodata value (if nan, the nodata is set to -32768).
        eomanager: eoscale context manager.
        dev_mode: if True, dev mode activated.
        dev_dir: path to save dev files.

    Returns:
        the filled DSM.
    """
    
    if dev_mode:
        dev_dir += "/filling_DSM/"
        if not os.path.isdir(dev_dir):
            os.makedirs(dev_dir)
            
    filled_dsm = eomanager.get_array(key=dsm_key)[0]
    regular = eomanager.get_array(key=regular_key)[0]
    
    # We're also filling the irregular areas
    filled_dsm[regular==0] = nodata
    
    if dev_mode:
        filled_dsm_with_regular_path: str = os.path.join(dev_dir, "regular_dsm.tif")
        eomanager.write(key=dsm_key, img_path=filled_dsm_with_regular_path)
    
    dsm_resolution = eomanager.get_profile(key=dsm_key)["transform"][0]
    # Setting parameters for the DSM filling method
    regular_parameters: dict = {
        "nodata": nodata,
        # if max_object_size is less than 2+sqrt(2) overides the computed value to avoid information loss during dezoom
        "filling_iterations": int(np.max([int(2+np.sqrt(2)), np.floor((max_object_size / dsm_resolution) / 2)])), # Nb iterations = max_object_size (px) / 2 (allow to fill a hole between two points max_object_size apart)
        "use_bordernodata_mask": True
    }
    # if computed value for dezoom_factor is less than 2 overides the value with 2 to ensure a dezoom during the filling process
    dezoom_factor = int(np.max([2, np.floor(regular_parameters["filling_iterations"] * (2-np.sqrt(2)))])) # sqrt(2) to handle the diagonal neighbors
    nb_max_level = int(np.floor(math.log(np.min([filled_dsm.shape[0],filled_dsm.shape[1]]), dezoom_factor))) # limits the number of dezoom iterations
    BulldozerLogger.log("DSM filling parameters : filling_iterations={} / dezoom_factor={} / nb_max_level={}".format(regular_parameters["filling_iterations"], dezoom_factor,nb_max_level), logging.DEBUG)


    # First iterative filling for small no data areas
    [dsm_key] = eoexe.n_images_to_m_images_filter(inputs=[dsm_key, border_nodata_key],
                                                  image_filter=fill_dsm_method,
                                                  filter_parameters=regular_parameters,
                                                  generate_output_profiles=filled_dsm_profile,
                                                  context_manager=eomanager,
                                                  stable_margin=regular_parameters["filling_iterations"],
                                                  filter_desc="Iterative filling DSM level 0") 
    
    if dev_mode:
        filled_dsm_1stpass_path: str = os.path.join(dev_dir, "filled_dsm_downsample_level_0.tif")
        eomanager.write(key=dsm_key, img_path=filled_dsm_1stpass_path)
    
    filled_dsm = eomanager.get_array(key=dsm_key)[0]
    border_nodata = eomanager.get_array(key=border_nodata_key)[0]
    
    # Identifying the remaining inner no data areas
    remaining_nodata = (filled_dsm == nodata) & (border_nodata == 0)
    
    # Putting nan values instead of no data for the sampling function
    filled_dsm[filled_dsm == nodata] = np.nan  

    # if nodata areas are still in the DSM
    has_nodata = np.any(remaining_nodata)
    
    dezoom_level = 1
    # Downsampling until there is nodata remaining or reaching max level
    while has_nodata and dezoom_level<=nb_max_level:
        
        regular_parameters["use_bordernodata_mask"] = False
        
        filled_dsm = eomanager.get_array(key=dsm_key)[0]

        # Downsampling the DSM to fill the large nodata areas. the order is set to 1 because it's the only one that handle no data 
        # The mode is set to nearest because it expands the image when zooming if the resampling factor is not proportional to the image size
        filled_dsm_downsampled = zoom(filled_dsm, 1/(dezoom_factor**dezoom_level), order=1, mode="nearest") 
        # Putting back nodata values
        filled_dsm_downsampled = np.where(np.isnan(filled_dsm_downsampled), nodata, filled_dsm_downsampled) 

        # Creating new profile for downsampled data
        downsampled_profile = downsample_profile(profile=eomanager.get_profile(key=dsm_key), factor=dezoom_factor**dezoom_level)
        downsampled_profile.update(width=np.shape(filled_dsm_downsampled)[1], height=np.shape(filled_dsm_downsampled)[0])
        downsampled_filled_dsm_key = eomanager.create_image(downsampled_profile)

        filled_dsm_downsample = eomanager.get_array(key=downsampled_filled_dsm_key)[0]
        filled_dsm_downsample[:] = filled_dsm_downsampled    
        
        # HOTFIX to remove: until we change eoscale we have to compute the tile size manually 
        BulldozerLogger.log("DSM filling during downsampling step : level={} / specific_tile_size={} / default_tile_size={}".format(dezoom_level, int(np.ceil(dsm_resolution*dezoom_factor**dezoom_level)), math.sqrt((filled_dsm.shape[0] * filled_dsm.shape[1]) // eomanager.nb_workers)), logging.DEBUG)
        if np.ceil(dsm_resolution*dezoom_factor**dezoom_level) > math.sqrt((filled_dsm.shape[0] * filled_dsm.shape[1]) // eomanager.nb_workers):
            specific_tile_size = int(np.ceil(dsm_resolution*dezoom_factor**dezoom_level))
        else:
            specific_tile_size = None
       
        # Iterative filling for the remaining no data areas
        [downsampled_filled_dsm_key] = eoexe.n_images_to_m_images_filter(inputs=[downsampled_filled_dsm_key],
                                                                         image_filter=fill_dsm_method,
                                                                         filter_parameters=regular_parameters,
                                                                         generate_output_profiles=filled_dsm_profile,
                                                                         context_manager=eomanager,
                                                                         stable_margin=regular_parameters["filling_iterations"],
                                                                         filter_desc="Iterative filling DSM level "+str(dezoom_level),
                                                                         specific_tile_size = specific_tile_size) 
                    
        filled_dsm_downsample = eomanager.get_array(key=downsampled_filled_dsm_key)[0]
        
        # Putting nan values instead of no data for the sampling function
        filled_dsm_downsample[filled_dsm_downsample == nodata] = np.nan
        
        if dev_mode:
            filled_dsm_downsample_path: str = os.path.join(dev_dir, "filled_dsm_downsampled_level_"+str(dezoom_level)+".tif")
            eomanager.write(key=downsampled_filled_dsm_key, img_path=filled_dsm_downsample_path)
        
        # Merging the current level with the first one
        scale_y = filled_dsm.shape[0] / filled_dsm_downsample.shape[0]
        scale_x = filled_dsm.shape[1] / filled_dsm_downsample.shape[1]
        filled_dsm_resample = zoom(filled_dsm_downsample, (scale_y, scale_x), order=1, mode="nearest")

        filled_dsm[:] = np.where(remaining_nodata == 1, filled_dsm_resample, filled_dsm)
        
        remaining_nodata = (np.isnan(filled_dsm)) & (border_nodata == 0)
        
        has_nodata = np.any(remaining_nodata)
        
        eomanager.release(key=downsampled_filled_dsm_key)
        
        dezoom_level+=1

    # After the downsampling step it may remains nodata due to a number of iteration too low at each level or due to the downsampling method (e.g. noada pixel in a batch that convert to a neigbour value after dezoom step and that never fill later)
    # Another multi(resolution iterative filling step is apply to fill the remaining nodata. In this case we try to fill the remaining nodata with the lowest resolution and iterate until reaching full resolution.
    # At each resolution we iterate until reaching the theorical maximal number of iteration based on the resolution or until there is no remaining nodata.
    dezoom_level-=2

    # upsampling until retrieving the original resolution
    while has_nodata and dezoom_level>=1:

        filled_dsm = eomanager.get_array(key=dsm_key)[0]

        # We keep the same method for dezoom
        filled_dsm_upsampled = zoom(filled_dsm, 1/(dezoom_factor**dezoom_level), order=1, mode="nearest") 
        filled_dsm_upsampled = np.where(np.isnan(filled_dsm_upsampled), nodata, filled_dsm_upsampled)

        # The number of iteration is set to the maximum at the current resolution (we consider the max distance to reach with current resolution considering the diagional of the image)
        regular_parameters["filling_iterations"] = int(np.floor(np.sqrt(filled_dsm.shape[0]**2+filled_dsm.shape[1]**2) // dezoom_factor**dezoom_level))
        BulldozerLogger.log("DSM filling during upsampling step : level={} / filling_iterations={}".format(dezoom_level, regular_parameters["filling_iterations"]), logging.DEBUG)

        # Creating new profile for downsampled data
        downsampled_profile = downsample_profile(profile=eomanager.get_profile(key=dsm_key), factor=dezoom_factor**dezoom_level)
        downsampled_profile.update(width=np.shape(filled_dsm_upsampled)[1], height=np.shape(filled_dsm_upsampled)[0])
        downsampled_filled_dsm_key = eomanager.create_image(downsampled_profile)

        filled_dsm_downsample = eomanager.get_array(key=downsampled_filled_dsm_key)[0]
        filled_dsm_downsample[:] = filled_dsm_upsampled        
        
        # HOTFIX to remove: until we change eoscale we have to compute the tile size manually 
        BulldozerLogger.log("DSM filling during upsampling step : level={} / specific_tile_size={} / default_tile_size={}".format(dezoom_level, int(np.ceil(dsm_resolution*dezoom_factor**dezoom_level)), math.sqrt((filled_dsm.shape[0] * filled_dsm.shape[1]) // eomanager.nb_workers)), logging.DEBUG)
        if np.ceil(dsm_resolution*dezoom_factor**dezoom_level) > math.sqrt((filled_dsm.shape[0] * filled_dsm.shape[1]) // eomanager.nb_workers):
            specific_tile_size = int(np.ceil(dsm_resolution*dezoom_factor**dezoom_level))
        else:
            specific_tile_size = None

        # Iterative filling for the remaining no data areas
        [downsampled_filled_dsm_key] = eoexe.n_images_to_m_images_filter(inputs=[downsampled_filled_dsm_key],
                                                                         image_filter=fill_dsm_method,
                                                                         filter_parameters=regular_parameters,
                                                                         generate_output_profiles=filled_dsm_profile,
                                                                         context_manager=eomanager,
                                                                         stable_margin=regular_parameters["filling_iterations"],
                                                                         filter_desc="Iterative filling DSM level -"+str(dezoom_level),
                                                                         specific_tile_size = specific_tile_size) 
                    
        filled_dsm_downsample = eomanager.get_array(key=downsampled_filled_dsm_key)[0]
        
        # Putting nan values instead of no data for the sampling function
        filled_dsm_downsample[filled_dsm_downsample == nodata] = np.nan
        
        if dev_mode:
            filled_dsm_upsample_path: str = os.path.join(dev_dir, "filled_dsm_upsampled_level_"+str(dezoom_level)+".tif")
            eomanager.write(key=downsampled_filled_dsm_key, img_path=filled_dsm_upsample_path)
        
        # Merging the current level with the first one
        scale_y = filled_dsm.shape[0] / filled_dsm_downsample.shape[0]
        scale_x = filled_dsm.shape[1] / filled_dsm_downsample.shape[1]
        filled_dsm_resample = zoom(filled_dsm_downsample, (scale_y, scale_x), order=1, mode="nearest")

        filled_dsm[:] = np.where(remaining_nodata == 1, filled_dsm_resample, filled_dsm)
        
        remaining_nodata = (np.isnan(filled_dsm)) & (border_nodata == 0)
        
        has_nodata = np.any(remaining_nodata)
        
        eomanager.release(key=downsampled_filled_dsm_key)
        
        dezoom_level-=1

    
    if has_nodata:
        # Last iterative filling for small no data areas
        [dsm_key] = eoexe.n_images_to_m_images_filter(inputs=[dsm_key, border_nodata_key],
                                                    image_filter=fill_dsm_method,
                                                    filter_parameters=regular_parameters,
                                                    generate_output_profiles=filled_dsm_profile,
                                                    context_manager=eomanager,
                                                    stable_margin=regular_parameters["filling_iterations"],
                                                    filter_desc="Iterative filling DSM Final level") 
        
        if dev_mode:
            filled_dsm_lastpass_path: str = os.path.join(dev_dir, "filled_dsm_upsample_level_0.tif")
            eomanager.write(key=dsm_key, img_path=filled_dsm_lastpass_path)

    # Set the border nodata to very high value in order to avoid underestimation of the DTM on the border
    #TODO change to another method?
    filled_dsm[border_nodata==1] = 9999
    
    return {
        "filled_dsm" : dsm_key
    }