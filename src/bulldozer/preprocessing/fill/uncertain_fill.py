import copy
import numpy as np
import rasterio
from rasterio.fill import fillnodata
import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe
import bulldozer.preprocessing.uncertain as pyunc

def prefill_uncertain_dsm(input_buffers : list, 
                          input_profiles: list, 
                          filter_parameters: dict) -> np.ndarray:

    unc_filter = pyunc.PyUncertain()
    return  unc_filter.prefillUncertain(dsm  = input_buffers[0][0],
                                        uncertain_mask = input_buffers[1][0],
                                        regular_mask = input_buffers[2][0],
                                        search_radius = filter_parameters["search_radius"],
                                        max_slope_percent = filter_parameters["max_slope_percent"],
                                        dsm_resolution = input_profiles[0]["transform"][0])

def prefill_uncertain_profiles(input_profiles: list,
                                params: dict) -> list:
    """ """
    input_profiles[1]['nodata'] = input_profiles[0]['nodata']
    input_profiles[1]['dtype'] = input_profiles[0]['dtype']
    return [input_profiles[0], input_profiles[1]]

def rasterio_fillnodata_filter(input_buffers : list, 
                               input_profiles: list, 
                               filter_parameters: dict) -> np.ndarray:
    """ """
    input_dsm = input_buffers[0][0,:,:]
    nodata_mask = np.where(input_dsm == float(profile["nodata"]), 0, 1)

    copy_dsm = copy.deepcopy(input_dsm)
    copy_dsm[:,:] = fillnodata(copy_dsm[:,:], 
                               mask=nodata_mask, 
                               max_search_distance=filter_parameters["search_radius"], 
                               smoothing_iterations=0)

    return copy_dsm

def fill_remaining_nodata(prefilled_dsm_key: str,
                          uncertain_map_key: str,
                          max_slope_percent: float,
                          search_radius: float,
                          eomanager: eom.EOContextManager) -> str :
    """ """
    arr = eomanager.get_array(key=prefilled_dsm_key)[0,:,:]
    nodata_value: float = eomanager.get_profile(key=prefilled_dsm_key)["nodata"]
    uncertain_map = eomanager.get_array(key=uncertain_map_key)[0,:,:],
    profile = eomanager.get_profile(key=prefilled_dsm_key)
    nodata_mask = np.where(arr == nodata_value, 0, 1)

    if np.min(nodata_mask) < 1:
        uncertain_map[ nodata_mask < 1 ] = (max_slope_percent * profile["transform"][0] * search_radius) / 100.0
    
    rasterio_fillnodata_parameters:dict = {
        "search_radius": search_radius
    }

    use_rasterio: bool = False

    while np.min(nodata_mask) < 1:

        use_rasterio = True

        [filled_dsm_key ]  = eoexe.n_images_to_m_images_filter(inputs = [prefilled_dsm_key],
                                                               image_filter = rasterio_fillnodata_filter,
                                                               filter_parameters = rasterio_fillnodata_parameters,
                                                               context_manager = eomanager,
                                                               stable_margin = search_radius,
                                                               filter_desc= "Rasterio Fillnodata processing...")

        nodata_mask = np.where(eomanager.get_array(key=filled_dsm_key)[0,:,:] == nodata_value, 0, 1)

        if np.min(nodata_mask) < 1:
            eomanager.release(key = prefilled_dsm_key)
            prefilled_dsm_key = filled_dsm_key
    
    if use_rasterio:
        eomanager.release(key = prefilled_dsm_key)
        return filled_dsm_key
    else:
        return prefilled_dsm_key

def run(input_dsm_key: str,
        uncertain_mask_key: str,
        regular_mask_key: str,
        eomanager: eom.EOContextManager,
        search_radius: int,
        max_slope_percent: float) -> dict :
    """ """

    # Prefill the dsm and create the uncertain map
    prefill_parameters: dict = {
        "search_radius": search_radius,
        "max_slope_percent": max_slope_percent
    }

    [prefilled_dsm_key, uncertain_map_key] = eoexe.n_images_to_m_images_filter(inputs = [input_dsm_key, uncertain_mask_key, regular_mask_key],
                                                                                image_filter = prefill_uncertain_dsm,
                                                                                filter_parameters = prefill_parameters,
                                                                                generate_output_profiles = prefill_uncertain_profiles,
                                                                                context_manager = eomanager,
                                                                                stable_margin = search_radius,
                                                                                filter_desc= "Prefill and uncertainty processing...")

    # While remaining no data, use rasterio.fillnodata
    filled_dsm_key = fill_remaining_nodata(prefilled_dsm_key = prefilled_dsm_key,
                                           uncertain_map_key = uncertain_map_key,
                                           max_slope_percent = max_slope_percent,
                                           search_radius = search_radius,
                                           eomanager = eomanager)
    
    return {
        "filled_dsm": filled_dsm_key,
        "uncertain_map": uncertain_map_key
    }