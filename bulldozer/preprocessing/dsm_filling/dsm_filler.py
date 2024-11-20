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
    #TODO - run with eoscale
    filled_dsm = eomanager.get_array(key=dsm_key)[0]
    regular_mask = eomanager.get_array(key=regular_key)[0]
    border_mask = eomanager.get_array(key=border_nodata_key)[0]

    #TODO - Hotfix to remove
    unfilled_dsm_mask = eomanager.get_array(key=unfilled_dsm_mask_key)[0]

    inv_msk = np.logical_not(regular_mask)
    inv_msk[border_mask == 1] = 0

    #TODO - replace for statement by while nodata & iter <  max iter + add a flag : if iter_final == max_iter => replace 9999 values by dsm_nodata in the last step
    # Fill the inner nodata and not regular areas
    filled_dsm[:] = fill.iterative_filling(filled_dsm, inv_msk, nodata)[:]
    # For the border nodata, put an high value to clip the DTM to the ground during the extraction step
    #TODO - Hotfix to remove
    unfilled_dsm_mask[filled_dsm == nodata] = 1
    filled_dsm[filled_dsm == nodata] = 9999

    
    return {
        "filled_dsm" : dsm_key,
        "unfilled_dsm_mask_key" : unfilled_dsm_mask_key
    }
