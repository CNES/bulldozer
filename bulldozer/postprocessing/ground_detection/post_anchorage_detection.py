import numpy as np
import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe


def post_anchorage_filter(input_buffers: list,
                          input_profiles: list, 
                          filter_parameters: dict) -> np.ndarray:

    inter_dtm = input_buffers[0][0, :, :]
    dsm = input_buffers[1][0, :, :]
    regular_mask = input_buffers[2][0, :, :]
    post_anchorage_mask = np.where(np.logical_and(np.absolute(inter_dtm-dsm) <= filter_parameters["error_threshold"],regular_mask), 1, 0).astype(np.uint8)
    return post_anchorage_mask


def post_anchorage_profile(input_profiles: list,
                           params: dict) -> dict:
    return input_profiles[2]


def run(intermediate_dtm_key: str,
        dsm_key: str,
        regular_mask_key: str,
        error_threshold: float,
        eomanager: eom.EOContextManager) -> dict:
    """ """

    post_anchorage_parameters: dict = {
        "error_threshold": error_threshold
    }
    
    [post_anchorage_mask] = eoexe.n_images_to_m_images_filter(inputs=[intermediate_dtm_key, dsm_key, regular_mask_key],
                                                              image_filter=post_anchorage_filter,
                                                              filter_parameters=post_anchorage_parameters,
                                                              generate_output_profiles=post_anchorage_profile,
                                                              context_manager=eomanager,
                                                              stable_margin=0,
                                                              filter_desc="Post anchorage processing...")

    return {
        "post_process_anchorage": post_anchorage_mask
    }
