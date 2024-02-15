from typing import List

import numpy as np
from rasterio.fill import fillnodata
from scipy import ndimage

import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe
import rasterio


def run(dsm_key: str,
        mask_key: str,
        no_data_mask_key: str,
        fill_search_radius: int,
        eomanager: eom.EOContextManager) -> dict:
    """
    
    """
    # TODO update with n_images_to_m_images_filter when eoscale can handle in memory modification
    no_data = eomanager.get_profile(dsm_key)['nodata']
    filled_dsm = eomanager.get_array(key=dsm_key)[0]
    mask = eomanager.get_array(key=mask_key)[0]
    
    input_no_data_mask = np.where(filled_dsm==no_data, 0, 1)
    full_msk = np.ones(filled_dsm.shape)
    full_msk = np.logical_and(full_msk, mask)
    full_msk = np.logical_and(full_msk, input_no_data_mask)

    filled_dsm = fillnodata(filled_dsm, full_msk, fill_search_radius)
    filled_dsm[filled_dsm == no_data] = 9999
    
    # construct no data mask to apply after drape cloth
    #struct =  ndimage.generate_binary_structure(2, 1)
    radius = fill_search_radius
    iter = 1
    if fill_search_radius > 6:
        iter = fill_search_radius // 5
        radius = 6
    
    xx, yy = np.mgrid[:2*radius+1, :2*radius+1]
    circle = (xx - radius) ** 2 + (yy - radius) ** 2
    struct = np.sqrt(circle) <= radius
    no_data_mask = eomanager.get_array(key=no_data_mask_key)[0]
    morpho_no_data = ndimage.binary_closing(np.pad(input_no_data_mask, fill_search_radius, 'constant', constant_values=0), struct, iterations=iter)
    no_data_mask[~morpho_no_data[fill_search_radius:-fill_search_radius, fill_search_radius:-fill_search_radius]] = 1
    no_data_mask[filled_dsm == 9999] = 1
    
    return {
        "filled_dsm": dsm_key,
        "no_data_mask": no_data_mask_key
    }
