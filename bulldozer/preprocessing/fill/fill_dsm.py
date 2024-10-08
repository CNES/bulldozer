from typing import List

import numpy as np

import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe
import bulldozer.preprocessing.fill.filltoground as ftg


def run(dsm_key: str,
        mask_key: str,
        border_no_data_key: str,
        eomanager: eom.EOContextManager) -> dict:
    """
    
    """
    no_data = eomanager.get_profile(dsm_key)['nodata']
    filled_dsm = eomanager.get_array(key=dsm_key)[0]
    mask = eomanager.get_array(key=mask_key)[0]
    border_mask = eomanager.get_array(key=border_no_data_key)[0]

    inv_msk = np.logical_not(mask)
    inv_msk[border_mask == 1] = 0

    filled_dsm[:] = ftg.iterative_filling(filled_dsm, inv_msk, no_data)[:]
    filled_dsm[filled_dsm == no_data] = 9999
    
    return {
        "filled_dsm": dsm_key
    }
