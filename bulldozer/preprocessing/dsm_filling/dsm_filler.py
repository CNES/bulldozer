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

@Runtime
def fill_dsm(dsm_key: str,
             regular_key: str,
             border_no_data_key: str,
             eomanager: eom.EOContextManager) -> dict:
    """
    This method returns the binary masks of the borrder and inner nodata.
    The border nodata correpond to the nodata points on the edges if the DSM is skewed and the inner nodata correspond to the other nodata points.

    Args:
        dsm_path: path to the input DSM.
        nb_max_workers: number of available workers (multiprocessing).
        nodata: nodata value of the input DSM. If None, retrieve this value from the input DSM metadata.

    Returns:
        border and inner nodata masks.
    """
    #TODO run with eoscale
    no_data = eomanager.get_profile(dsm_key)['nodata']
    filled_dsm = eomanager.get_array(key=dsm_key)[0]
    regular_mask = eomanager.get_array(key=regular_key)[0]
    border_mask = eomanager.get_array(key=border_no_data_key)[0]

    inv_msk = np.logical_not(regular_mask)
    inv_msk[border_mask == 1] = 0

    # Fill the inner nodata and not regular areas
    filled_dsm[:] = fill.iterative_filling(filled_dsm, inv_msk, no_data)[:]
    # For the border nodata, put an high value to clip the DTM to the ground
    filled_dsm[filled_dsm == no_data] = 9999
    
    return {
        "filled_dsm": dsm_key
    }
