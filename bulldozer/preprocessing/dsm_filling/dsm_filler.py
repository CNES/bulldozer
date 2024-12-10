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
    output_profile = input_profiles[0]
    return output_profile

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
                                         input_buffers[2][0, :, :],
                                         nodata_value=filter_parameters["nodata"])
    return dsm.astype(np.float32)  

@Runtime
def fill_dsm(dsm_key: str,
             regular_key: str,
             border_nodata_key: str,
             unfilled_dsm_mask_key :str,
             nodata: float,
             eomanager: eom.EOContextManager) -> dict:
    """
    This fills the nodata of the input DSM for the following dtm extraction step.

    Args:
        dsm_key: input DSM.
        regular_key: regular mask.
        border_nodata_key: border nodata mask.
        nodata: DSM nodata value (if nan, the nodata is set to -32768).
        eomanager: eoscale context manager.

    Returns:
        the filled DSM.
    """

    regular_parameters: dict = {
        "nodata": nodata
    }
    
    [dsm_key] = eoexe.n_images_to_m_images_filter(inputs=[dsm_key, regular_key, border_nodata_key],
                                                  image_filter=fill_dsm_method,
                                                  filter_parameters=regular_parameters,
                                                  generate_output_profiles=filled_dsm_profile,
                                                  context_manager=eomanager,
                                                  stable_margin=1,
                                                  filter_desc="Iterative filling DSM")

    
    #TODO - HOTFIX to remove
    unfilled_dsm_mask = eomanager.get_array(key=unfilled_dsm_mask_key)[0]
    filled_dsm = eomanager.get_array(key=dsm_key)[0]
    unfilled_dsm_mask[filled_dsm == nodata] = 1
    filled_dsm[filled_dsm == nodata] = 9999

    return {
        "filled_dsm" : dsm_key,
        "unfilled_dsm_mask_key" : unfilled_dsm_mask_key
    }