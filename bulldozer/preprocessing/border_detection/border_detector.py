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
This module is used to detect border and inner nodata in the input DSM.
"""
import numpy as np
# from scipy.spatial import ConvexHull
from scipy.ndimage import zoom, binary_erosion, binary_fill_holes
from skimage.morphology import convex_hull_image
from skimage.draw import polygon

import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe
from bulldozer.utils.bulldozer_logger import Runtime
import bulldozer.preprocessing.border as border


def nodata_mask_profile(input_profile: list,
                        params: dict) -> dict:
    """
        This method is used in the main `detect_border_nodata` 
        method to provide the output mask profile (binary profile).

        Args:
            input_profiles: input profile.
            params: extra parameters.
        
        Returns:
            updated profile.
    """
    output_profile = input_profile[0]
    output_profile['dtype'] = np.ubyte
    output_profile['nodata'] = None
    return output_profile


def border_nodata_filter(input_buffers: list,
                         input_profiles: list,
                         filter_parameters: dict) -> np.ndarray:
    """ 
        This method is used in the main `detect_border_nodata` method.
        It calls the Cython method to extract border nodata along an axis (vertical or horizontal).

    Args:
        input_buffers: input DSM.
        input_profiles: DSM profile.
        filter_parameters: dictionary containing nodata value and the axis for the detection (True: vertical or False: horizontal).
    
    Returns:
        border nodata mask along specified axis.
    """
    dsm = input_buffers[0]
    nodata = filter_parameters['nodata']

    border_nodata = border.PyBorderNodata()

    if filter_parameters["doTranspose"]:
        # Vertical border nodata detection case
        border_nodata_mask = border_nodata.build_border_nodata_mask(dsm.T, nodata, True).astype(np.ubyte)
        return border_nodata_mask.T
    else:
        # Horizontal border nodata detection case
        return border_nodata.build_border_nodata_mask(dsm, nodata, False).astype(np.ubyte)

    


def inner_nodata_filter(input_buffers: list,
                        input_profiles: list,
                        filter_parameters: dict) -> np.ndarray:
    """ 
        This method is used in the main `detect_border_nodata` method.
        It calls the Cython method to extract inner nodata.

    Args:
        input_buffers: input DSM.
        input_profiles: DSM profile.
        filter_parameters: dictionary containing nodata value.

    Returns:
        inner nodata mask along specified axis.
    """
    
    dsm = input_buffers[0]
    border_nodata_mask = input_buffers[1]
    nodata = filter_parameters['nodata']

    inner_nodata_mask = np.logical_and(np.logical_not(border_nodata_mask), dsm == nodata)
    
    return inner_nodata_mask


@Runtime
def detect_border_nodata(dsm_key: str,
                         nodata: float,
                         eomanager: eom.EOContextManager) -> np.ndarray:
    """
    This method returns the binary masks flagging the border and inner nodata.
    The border nodata correpond to the nodata points on the edges if the DSM is skewed and the inner nodata correspond to the other nodata points.

    Args:
        dsm_key: path to the input DSM.
        nodata: DSM nodata value (if nan, the nodata is set to default value: -32768.0).
        eomanager: eoscale context manager.

    Returns:
        border and inner nodata masks.
    """
    # Horizontal border nodata detection
    border_nodata_parameters: dict = {
        'nodata': nodata,
        'doTranspose': False
    }
    [hor_border_nodata_mask_key] = eoexe.n_images_to_m_images_filter(inputs=[dsm_key],
                                                                      image_filter=border_nodata_filter,
                                                                      filter_parameters=border_nodata_parameters,
                                                                      generate_output_profiles=nodata_mask_profile,
                                                                      context_manager=eomanager,
                                                                      stable_margin=0,
                                                                      filter_desc="Horizontal nodata mask processing...",
                                                                      tile_mode=False)
    # Vertical border nodata detection
    border_nodata_parameters: dict = {
        'nodata': nodata,
        'doTranspose': True
    }
    [border_nodata_mask_key] = eoexe.n_images_to_m_images_filter(inputs=[dsm_key],
                                                                 image_filter=border_nodata_filter,
                                                                 filter_parameters=border_nodata_parameters,
                                                                 generate_output_profiles=nodata_mask_profile,
                                                                 context_manager=eomanager,
                                                                 stable_margin=0,
                                                                 filter_desc="Vertical nodata mask processing...",
                                                                 tile_mode=False,
                                                                 strip_along_lines=True)

    hor_mask = eomanager.get_array(key=hor_border_nodata_mask_key)[0]
    border_mask = eomanager.get_array(key=border_nodata_mask_key)[0]
    np.logical_and(hor_mask, border_mask, out=border_mask)

    eomanager.release(key=hor_border_nodata_mask_key)
    
    
    #############
     ### 1st Method: ConvexHull on resampled data
    
#     # ConvexHull on the downasampled bordernodata
#     convex_hull = convex_hull_image(border_mask_downsample == 0)
#     border_mask_downsample[convex_hull] = 0
    
#     # Back to DSM resolution
#     scale_y = border_mask.shape[0] / border_mask_downsample.shape[0]
#     scale_x = border_mask.shape[1] / border_mask_downsample.shape[1]
#     border_mask_resample = zoom(border_mask_downsample, (scale_y, scale_x), order=1, mode='nearest')
#     border_mask_resample = binary_erosion(border_mask_resample, structure=np.ones((3, 3)))
    
#     # New border mask with ConvexHull without adding data
#     # border_mask[border_mask_resample==0] = 0      
#     new_border_mask = border_mask.copy()
#     new_border_mask[(border_mask_resample == 0) & (border_mask == 1)] = 0
#     new_border_mask[border_mask == 0] = 0
#     border_mask = new_border_mask
    ###########
        
    ### 2nd Method: ConvexHull on resampled data, erosion and binary fill holes  
    # Downsample bordernodata
#     dsm_resolution = eomanager.get_profile(key=dsm_key)['transform'][0]
#     dezoom_factor = max(2, int(np.floor((max_object_size / dsm_resolution) / 2 * (2 - np.sqrt(2)))))
#     border_mask = np.where(border_mask == 0, 1, 0)
#     border_mask_downsample = zoom(border_mask, 1 / dezoom_factor, output=np.uint8, order=1, mode='nearest')
    
#     # ConvexHull on the downasampled bordernodata
#     convex_hull = convex_hull_image(border_mask_downsample)
#     border_mask_downsample[convex_hull] = 1
    
#     # Erosion to avoid adding new data
#     # border_mask_downsample = binary_erosion(border_mask_downsample, structure=[[0,1,0],[1,1,1],[0,1,0]])
#     border_mask_downsample = binary_erosion(border_mask_downsample, structure=np.ones((5,5)))
    
#     # Back to DSM resolution
#     scale_y = border_mask.shape[0] / border_mask_downsample.shape[0]
#     scale_x = border_mask.shape[1] / border_mask_downsample.shape[1]
#     border_mask_resample = zoom(border_mask_downsample, (scale_y, scale_x), output=np.uint8, order=1, mode='nearest')
    
#     # New border mask with ConvexHull with corrections to avoid adding data    
#     np.logical_or(border_mask_resample, border_mask, out=border_mask_resample)
#     binary_fill_holes(border_mask_resample, output=border_mask_resample) # To fill the remaining holes
    
#     border_mask_resample = np.where(border_mask_resample == 0, 1, 0)
#     border_mask = eomanager.get_array(key=border_nodata_mask_key)[0]
#     border_mask[:] = border_mask_resample
    
    
    ###############
    ### Filling the holes inside the border nodata mask
    border_mask_reverted = np.where(border_mask == 0, 1, 0).astype(np.uint8)
    binary_fill_holes(border_mask_reverted, output=border_mask_reverted)
    border_mask_reverted = np.where(border_mask_reverted == 0, 1, 0)
    border_mask = eomanager.get_array(key=border_nodata_mask_key)[0]
    border_mask[:] = border_mask_reverted
            
    # Inner nodata detection
    [inner_nodata_mask_key] = eoexe.n_images_to_m_images_filter(inputs=[dsm_key, border_nodata_mask_key],
                                                                image_filter=inner_nodata_filter,
                                                                filter_parameters=border_nodata_parameters,
                                                                generate_output_profiles=nodata_mask_profile,
                                                                context_manager=eomanager,
                                                                stable_margin=0,
                                                                filter_desc="Build Inner NoData Mask")    
    

    return {
        "border_nodata_mask": border_nodata_mask_key,
        "inner_nodata_mask": inner_nodata_mask_key
    }
