from typing import List
import copy

import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe


def run(dsm_key: str,
        mask_key: str,
        eomanager: eom.EOContextManager) -> dict:
    """ """
    # TODO update with n_images_to_m_images_filter when eoscale can handle in memory modification
    filled_dsm = eomanager.get_array(key=dsm_key)[0]
    mask = eomanager.get_array(key=mask_key)[0]
    filled_dsm[mask == 0] = 9999  # WARNING: this line impact the shared memory of input dsm
    filled_dsm[filled_dsm == (eomanager.get_profile(dsm_key))['nodata']] = 9999
    return {
        "filled_dsm": dsm_key
    }
