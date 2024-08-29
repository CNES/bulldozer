import numpy as np
import rasterio

import bulldozer.preprocessing.bordernodata.bordernodata as bnodata

import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe
from bulldozer.utils.helper import Runtime, DefaultValues

def generate_output_profile_for_mask(input_profile: list,
                                     params: dict) -> dict:
    output_profile = input_profile[0]
    output_profile['dtype'] = np.ubyte
    output_profile['nodata'] = None
    return output_profile



def border_nodata_computer( inputBuffers: list,
                            input_profiles: list,
                            filter_parameters: dict) -> np.ndarray:
    """ 
    This method computes the border nodata mask in a given window of the input DSM.

    Args:
        inputBuffers: contain just one DSM buffer.
        filter_parameters:  dictionary containing:
            nodata value: DSM potentially custom nodata 
            doTranspose: boolean flag to computer either horizontally or vertically the border no data.
    Returns:
        mask flagging the border nodata areas
    """
    dsm = inputBuffers[0]
    nodata = filter_parameters['nodata']

    if np.isnan(nodata):
        dsm = np.nan_to_num(dsm, False, nan=DefaultValues['NODATA'])
        nodata = DefaultValues['NODATA']
      
    # We're using our C++ implementation to perform this computation
    border_nodata = bnodata.PyBorderNodata()

    if filter_parameters["doTranspose"]:
        return border_nodata.build_border_nodata_mask(dsm, nodata).astype(np.ubyte)
    else:
        # Vertical border nodata detection case (axis = 1)
        dsm = dsm.T
        border_nodata_mask = border_nodata.build_border_nodata_mask(dsm, nodata).astype(np.ubyte)
        return border_nodata_mask.T
    

def inner_nodata_computer( inputBuffers: list,
                            input_profiles: list,
                            filter_parameters: dict) -> np.ndarray:
    """ 
    This method computes the inner nodata mask in a given window of the input DSM.

    Args:
        inputBuffers: contain one DSM buffer and the border no data buffer.
        filter_parameters:  dictionary containing:
            nodata value: DSM potentially custom nodata 
            doTranspose: boolean flag to computer either horizontally or vertically the border no data.
    Returns:
        mask flagging the inner nodata areas
    """
    
    dsm = inputBuffers[0]
    border_nodata_mask = inputBuffers[1]
    nodata = filter_parameters['nodata']

    inner_nodata_mask = np.logical_and(np.logical_not(border_nodata_mask), dsm == nodata)
    
    return inner_nodata_mask


@Runtime
def run(dsm_key : str, 
        eomanager: eom.EOContextManager,
        nodata : float) -> np.ndarray:
    
    """
    This method builds a mask corresponding to the inner and border nodata values.
    Those areas correpond to the nodata points on the edges if the DSM is skewed.

    Args:
        dsm_path: path to the input DSM.
        nb_max_workers: number of availables workers (multiprocessing).
        nodata: nodata value of the input DSM. If None, retrieve this value from the input DSM metadata.

    Returns:
        border nodata boolean masks.
    """

    border_nodata_parameters: dict = {
        'nodata': nodata,
        'doTranspose': False
    }
    
    [border_nodata_mask_key] = eoexe.n_images_to_m_images_filter(inputs=[dsm_key],
                                                                        image_filter=border_nodata_computer,
                                                                        filter_parameters=border_nodata_parameters,
                                                                        generate_output_profiles=generate_output_profile_for_mask,
                                                                        context_manager=eomanager,
                                                                        stable_margin=0,
                                                                        filter_desc="Build Border NoData Mask")

    [inner_nodata_mask_key] = eoexe.n_images_to_m_images_filter(inputs=[dsm_key, border_nodata_mask_key],
                                                                image_filter=inner_nodata_computer,
                                                                filter_parameters=border_nodata_parameters,
                                                                generate_output_profiles=generate_output_profile_for_mask,
                                                                context_manager=eomanager,
                                                                stable_margin=0,
                                                                filter_desc="Build Inner NoData Mask")

    return {
        "border_no_data_mask": border_nodata_mask_key,
        "inner_no_data_mask": inner_nodata_mask_key
    }