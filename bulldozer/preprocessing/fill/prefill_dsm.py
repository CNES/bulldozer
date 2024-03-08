import copy
import numpy as np
import rasterio
from rasterio.fill import fillnodata
import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe


def run(input_dsm_key: str,
        refined_min_z: float,
        refined_max_z: float,
        eomanager: eom.EOContextManager) -> dict :
    """ """
    # TODO update with n_images_to_m_images_filter when eoscale can handle in memory modification
    filled_dsm = eomanager.get_array(key=input_dsm_key)[0]
    filled_dsm[filled_dsm < refined_min_z] = 9999  # WARNING: this line impact the shared memory of input dsm
    filled_dsm[filled_dsm > refined_max_z] = 9999 
    filled_dsm[filled_dsm == (eomanager.get_profile(input_dsm_key))['nodata']] = 9999
    return {
        "filled_dsm": input_dsm_key
    }