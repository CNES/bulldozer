import scipy
import copy
import numpy as np
from tqdm import tqdm
from rasterio import Affine

import bulldozer.eoscale.manager as eom
import bulldozer.eoscale.eo_executors as eoexe


def next_power_of_2(x: int) -> int:
    """
    This function returns the smallest power of 2 that is greater than or equal to a given non-negative integer x.

    Args:
        x : non negative integer.

    Returns:
        the corresponding power index power (2**index >= x).
    """
    return 0 if x == 0 else (1 << (x-1).bit_length()).bit_length() - 1


def get_max_pyramid_level(max_object_size_pixels: float) -> int :
    """ 
        Given the max size of an object on the ground,
        this method computes the max level of the pyramid
        for drape cloth algorithm
    """
    power = next_power_of_2(int(max_object_size_pixels))
    
    # Take the closest power to the max object size
    if abs(2**(power-1) - max_object_size_pixels) < abs(2**power - max_object_size_pixels):
        power -= 1

    if power < 0 :
        power = 0
    return power


def allocate_dezoom_dtm(level: int,
                        dezoom_shape: tuple,
                        eomanager: eom.EOContextManager) -> str:
    
    dezoomed_dtm_profile = {
        "count": 1,
        "height": dezoom_shape[1],
        "width": dezoom_shape[2],
        "dtype": np.float32
    }
    dtm_key = eomanager.create_image(profile=dezoomed_dtm_profile)
    return dtm_key


def apply_first_tension(dtm_key: str,
                        filled_dsm_key: str,
                        ground_mask_key: str,
                        eomanager: eom.EOContextManager,
                        nb_levels: int,
                        prevent_unhook_iter: int) -> None:
    
    dsm = eomanager.get_array(key=filled_dsm_key)[0, ::2**(nb_levels-1), ::2**(nb_levels-1)]
    predicted_anchors = eomanager.get_array(key=ground_mask_key)[0, ::2**(nb_levels-1), ::2**(nb_levels-1)]
    dtm = eomanager.get_array(key=dtm_key)[0, :, :]
    dtm[:, :] = dsm[:, :]
    snap_mask = predicted_anchors > 0

    # Prevent unhook from hills
    for i in tqdm(range(prevent_unhook_iter), desc="Prevent unhook from hills..."):
        dtm = scipy.ndimage.uniform_filter(dtm, output=dtm, size=3)
        dtm[snap_mask] = dsm[snap_mask]


def upsample(dtm_key: str,
             filled_dsm_key: str,
             level: int,
             dezoom_profile: dict,
             eomanager: eom.EOContextManager) -> str:
    
    next_shape = eomanager.get_array(key=filled_dsm_key).shape
    next_dtm_key = allocate_dezoom_dtm(level=level, dezoom_shape=next_shape, eomanager=eomanager)
    prev_dtm = eomanager.get_array(key=dtm_key)[0, :, :]
    next_dtm = eomanager.get_array(key=next_dtm_key)[0, :, :]

    # Adjust the slicing for odd row count
    if next_dtm.shape[0] % 2 == 1:
        s0 = np.s_[:-1]
    else:
        s0 = np.s_[:]
    
    # Adjust the slicing for odd column count
    if next_dtm.shape[1] % 2 == 1:
        s1 = np.s_[:-1]
    else:
        s1 = np.s_[:]

    # Only fill upsampled value since we are working on the same shared memory
    next_dtm[::2, ::2] = prev_dtm[:, :]
    next_dtm[1::2, ::2] = prev_dtm[s0, :]
    next_dtm[::2, 1::2] = prev_dtm[:, s1]
    next_dtm[1::2, 1::2] = prev_dtm[s0, s1]

    # Can release the prev dtm buffer
    eomanager.release(key=dtm_key)

    return next_dtm_key


def drape_cloth_filter(input_buffers: list, 
                       input_profiles: list, 
                       params: dict):
    """ """
    num_outer_iterations: int = params['num_outer_iterations']
    num_inner_iterations: int = params['num_inner_iterations']
    step: float = params['step']
    dtm = copy.deepcopy(input_buffers[0][0, :, :])
    dsm = input_buffers[1][0, :, :]
    predicted_anchors = input_buffers[2][0, :, :]
    snap_mask = predicted_anchors > 0

    for i in range(num_outer_iterations):
            
        dtm += step
            
        for j in range(num_inner_iterations):

            # Snap dtm to anchors point
            dtm[snap_mask] = dsm[snap_mask]

            # handle DSM intersections
            np.minimum(dtm, dsm, out=dtm)

            # apply spring tension forces (blur the DTM)
            dtm = scipy.ndimage.uniform_filter(dtm, size=3)
        
    # One final intersection check
    dtm[snap_mask] = dsm[snap_mask]
    np.minimum(dtm, dsm, out=dtm)
    
    return dtm


def drape_cloth_filter_gradient(input_buffers: list,
                                input_profiles: list,
                                params: dict):
    """ """
    num_outer_iterations: int = params['num_outer_iterations']
    num_inner_iterations: int = params['num_inner_iterations']
    step_scale: float = params['step_scale']
    nodata: int = params['nodata']
    dtm = copy.deepcopy(input_buffers[0][0, :, :])
    
    dsm = input_buffers[1][0, :, :]
    predicted_anchors = input_buffers[2][0, :, :]
    snap_mask = predicted_anchors > 0

    grad = np.abs(np.gradient(dtm))

    step = np.maximum(grad[0, :, :], grad[1, :, :]) * step_scale
    step = scipy.ndimage.maximum_filter(step, 5)
    #bfilters = sf.PyBulldozerFilters()
    for i in range(num_outer_iterations):


        dtm += step

        for j in range(num_inner_iterations):
            # Snap dtm to anchors point
            dtm[snap_mask] = dsm[snap_mask]

            # handle DSM intersections
            np.minimum(dtm, dsm, out=dtm)

            # apply spring tension forces (blur the DTM)
            #dtm[input_buffers[0][0, :, :] == nodata] = np.nan
            dtm = scipy.ndimage.uniform_filter(dtm, size=3)
            #dtm = bfilters.run(dtm, 3, nodata)

    # One final intersection check
    dtm[snap_mask] = dsm[snap_mask]
    np.minimum(dtm, dsm, out=dtm)
    return dtm


def drape_cloth_profiles(input_profiles: list,
                         params: dict) -> dict:
    return input_profiles[0]


def downsample_profile(profile, factor : float) :

    transform = profile['transform']

    newprofile = profile.copy()
    dst_transform = Affine.translation(transform[2], transform[5]) * Affine.scale(transform[0]*factor, transform[4]*factor)

    newprofile.update({
        'transform': dst_transform,
    })
    
    return newprofile


def drape_cloth(filled_dsm_key: str,
                ground_mask_key: str,
                eomanager: eom.EOContextManager,
                max_object_size: float,
                prevent_unhook_iter: int,
                num_outer_iterations: int,
                num_inner_iterations: int,
                nodata: float) -> str:
    """ """

    dsm_profile: dict = eomanager.get_profile(key=filled_dsm_key)
    dsm_resolution: float = dsm_profile["transform"][0]

    # Determine max object size in pixels
    max_object_size_pixels = max_object_size / dsm_resolution

    # Determine the dezoom factor wrt to max size of an object
    # on the ground.
    nb_levels = get_max_pyramid_level(max_object_size_pixels/2) + 1

    # Allocate memory for the max dezoomed dtm
    init_dtm_shape = eomanager.get_array(key=filled_dsm_key)[:, ::2**(nb_levels - 1), ::2**(nb_levels - 1)].shape
    dtm_key = allocate_dezoom_dtm(level=nb_levels - 1,
                                  dezoom_shape=init_dtm_shape,
                                  eomanager=eomanager)

    apply_first_tension(dtm_key=dtm_key,
                        filled_dsm_key=filled_dsm_key,
                        ground_mask_key=ground_mask_key,
                        eomanager=eomanager,
                        nb_levels=nb_levels,
                        prevent_unhook_iter=prevent_unhook_iter)

    # Init classical parameters of drape cloth
    level = nb_levels - 1
    current_num_outer_iterations = num_outer_iterations

    while level >= 0:

        print(f"Process level {level} ...")

        # Create the memviews of the filled dsm map of this level

        current_dezoom_profile: dict = downsample_profile(profile=eomanager.get_profile(key=filled_dsm_key),
                                                          factor=2**level)
        
        filled_dsm_memview = eomanager.create_memview(
            key=filled_dsm_key,
            arr_subset=eomanager.get_array(key=filled_dsm_key)[:, ::2**level, ::2**level],
            arr_subset_profile=current_dezoom_profile)
        
        ground_mask_key_memview = \
            eomanager.create_memview(key=ground_mask_key,
                                     arr_subset=eomanager.get_array(key=ground_mask_key)[:, ::2**level, ::2**level],
                                     arr_subset_profile=current_dezoom_profile)
        
        if level < nb_levels - 1:
            dtm_key = upsample(dtm_key=dtm_key,
                               filled_dsm_key=filled_dsm_memview,
                               level=level,
                               dezoom_profile=current_dezoom_profile,
                               eomanager=eomanager)

        drape_cloth_parameters: dict = {
            'num_outer_iterations': current_num_outer_iterations,
            'num_inner_iterations': num_inner_iterations,
            'nodata': nodata
        }

        drape_cloth_parameters['step_scale'] = 1. / (2 ** (nb_levels - level))

        [new_dtm_key] = eoexe.n_images_to_m_images_filter(
            inputs=[dtm_key, filled_dsm_memview, ground_mask_key_memview],
            image_filter=drape_cloth_filter_gradient,
            generate_output_profiles=drape_cloth_profiles,
            filter_parameters=drape_cloth_parameters,
            stable_margin=int(current_num_outer_iterations * num_inner_iterations * (3 / 2)), # 3 correspond to filter size
            context_manager=eomanager,
            filter_desc="Drape cloth simulation...")

        eomanager.release(key=dtm_key)
        dtm_key = new_dtm_key

        level -= 1
        current_num_outer_iterations = max(1, int(num_outer_iterations / 2**(nb_levels - 1 - level)))
    
    # dtm_key contains the final dtm, we can save it to disk
    dtm_key = eomanager.update_profile(key=dtm_key, profile=eomanager.get_profile(key=filled_dsm_key))
    
    return dtm_key