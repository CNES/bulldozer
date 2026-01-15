#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022-2026 Centre National d'Etudes Spatiales (CNES).
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
This module is used to simulate a drape cloth.
"""

import os
import shutil
from typing import Union

import numpy as np
import rasterio
import scipy
from tqdm import tqdm

from bulldozer.multiprocessing.bulldozer_executor import mp_n_to_m_images
from bulldozer.multiprocessing.bulldozer_manager import BulldozerContextManager
from bulldozer.multiprocessing.utils import write
from bulldozer.utils.bulldozer_logger import Runtime
from bulldozer.utils.helper import downsample_profile, ubyte_profile


def next_power_of_2(x: int) -> int:
    """
    This function returns the smallest power of 2 that is greater than or equal to a given non-negative integer x.

    Args:
        x : non negative integer.

    Returns:
        the corresponding power index power (2**index >= x).
    """
    return 0 if x == 0 else (1 << (x - 1).bit_length()).bit_length() - 1


def get_max_pyramid_level(max_object_size_pixels: float) -> int:
    """
    Given the max size of an object on the ground, this method computes the max level of the pyramid
    for drape cloth algorithm
    """
    power = next_power_of_2(int(max_object_size_pixels))

    # Take the closest power to the max object size
    if abs(2 ** (power - 1) - max_object_size_pixels) < abs(2**power - max_object_size_pixels):
        power -= 1

    power = max(power, 0)

    return power


def apply_first_tension(
    dsm: np.ndarray,
    prevent_unhook_iter: int,
    ground_mask: Union[np.ndarray, None] = None,
) -> np.ndarray:
    dtm = dsm.copy()
    # Prevent unhook from hills
    for _ in tqdm(range(prevent_unhook_iter), desc="Prevent unhook from hills..."):
        dtm = scipy.ndimage.uniform_filter(dtm, output=dtm, size=3)
        if ground_mask is not None:
            dtm[ground_mask > 0] = dsm[ground_mask > 0]

    return dtm


def upsample(dtm: np.ndarray, filled_dsm: np.ndarray) -> np.ndarray:
    """Dezoom and upsample input DTM."""
    next_dtm = np.zeros(filled_dsm.shape, dtype=dtm.dtype)

    # Adjust the slicing for odd row count
    if next_dtm.shape[0] % 2 == 1:
        s0 = np.s_[:-1]
    else:
        s0 = np.s_[:]  # type: ignore

    # Adjust the slicing for odd column count
    if next_dtm.shape[1] % 2 == 1:
        s1 = np.s_[:-1]
    else:
        s1 = np.s_[:]  # type: ignore

    # Only fill upsampled value since we are working on the same shared memory
    next_dtm[::2, ::2] = dtm[:, :]
    next_dtm[1::2, ::2] = dtm[s0, :]
    next_dtm[::2, 1::2] = dtm[:, s1]
    next_dtm[1::2, 1::2] = dtm[s0, s1]

    return next_dtm


def drape_cloth_filter_gradient(
    dtm: np.ndarray,
    dsm: np.ndarray,
    predicted_anchors: Union[np.ndarray, None],
    num_outer_iterations: int,
    num_inner_iterations: int,
    step_scale: float,
) -> np.ndarray:
    """Filter DTM with drape cloth gradient."""
    grad = np.abs(np.gradient(dtm))

    step = np.maximum(grad[0, :, :], grad[1, :, :]) * step_scale
    step = scipy.ndimage.maximum_filter(step, 5)

    for _i in range(num_outer_iterations):

        dtm += step

        for _j in range(num_inner_iterations):
            # Snap dtm to anchors point
            if predicted_anchors is not None:
                dtm[predicted_anchors > 0] = dsm[predicted_anchors > 0]

            # handle DSM intersections
            np.minimum(dtm, dsm, out=dtm)

            # apply spring tension forces (blur the DTM)
            dtm = scipy.ndimage.uniform_filter(dtm, size=3)

    # One final intersection check
    if predicted_anchors is not None:
        dtm[predicted_anchors > 0] = dsm[predicted_anchors > 0]

    np.minimum(dtm, dsm, out=dtm)

    return dtm


@Runtime
def drape_cloth(
    filled_dsm_key: Union[str, np.ndarray],
    filled_dsm_profile: dict,
    max_object_size: float,
    prevent_unhook_iter: int,
    num_outer_iterations: int,
    num_inner_iterations: int,
    inter_dtm_filename: str,
    manager: BulldozerContextManager,
    ground_mask_key: Union[str, np.ndarray, None] = None,
) -> Union[str, np.ndarray]:
    """ """
    dsm_resolution: float = filled_dsm_profile["transform"][0]

    # Determine max object size in pixels
    max_object_size_pixels = max_object_size / dsm_resolution

    # Determine the dezoom factor wrt to max size of an object on the ground.
    nb_levels = get_max_pyramid_level(max_object_size_pixels / 2) + 1

    if isinstance(filled_dsm_key, str):
        with rasterio.open(filled_dsm_key) as src:
            filled_dsm = src.read(1)
    else:
        filled_dsm = filled_dsm_key
    del filled_dsm_key

    if ground_mask_key is not None:
        use_ground_mask = True
        if isinstance(ground_mask_key, str):
            with rasterio.open(ground_mask_key) as src:
                ground_mask = src.read(1)
        else:
            ground_mask = ground_mask_key
        dtm = apply_first_tension(
            dsm=filled_dsm[:: 2 ** (nb_levels - 1), :: 2 ** (nb_levels - 1)],
            prevent_unhook_iter=prevent_unhook_iter,
            ground_mask=ground_mask[:: 2 ** (nb_levels - 1), :: 2 ** (nb_levels - 1)],  # type: ignore
        )
    else:
        use_ground_mask = False
        dtm = apply_first_tension(
            dsm=filled_dsm[:: 2 ** (nb_levels - 1), :: 2 ** (nb_levels - 1)],
            prevent_unhook_iter=prevent_unhook_iter,
        )
    del ground_mask_key

    # Init classical parameters of drape cloth
    dtm_key: Union[str, np.ndarray]
    filled_dsm_memview_key: Union[str, np.ndarray]
    ground_mask_memview_key: Union[str, np.ndarray, None]
    extracted_dtm_key: Union[str, np.ndarray]

    level = nb_levels - 1
    current_num_outer_iterations = num_outer_iterations
    dtm_key = dtm

    while level >= 0:

        print(f"Process level {level} ...")

        # Create the memviews of the filled dsm map of this level
        filled_dsm_memview = filled_dsm[:: 2**level, :: 2**level]
        ground_mask_memview = ground_mask[:: 2**level, :: 2**level] if use_ground_mask else None

        current_dezoom_profile: dict = downsample_profile(profile=filled_dsm_profile, factor=2**level)
        current_dezoom_profile.update({"height": filled_dsm_memview.shape[0], "width": filled_dsm_memview.shape[1]})

        current_dtm_dezoom_profile = current_dezoom_profile.copy()
        current_dtm_dezoom_profile.update({"count": 1, "dtype": np.float32})

        if use_ground_mask:
            current_ground_mask_dezoom_profile = ubyte_profile(current_dezoom_profile)

        if level < nb_levels - 1:
            if isinstance(dtm_key, str):
                with rasterio.open(dtm_key) as ds_dtm:
                    dtm = upsample(ds_dtm.read(1), filled_dsm_memview)
                os.remove(dtm_key)
            else:
                dtm = upsample(dtm_key, filled_dsm_memview)

        if manager.in_memory:
            dtm_key = dtm
            filled_dsm_memview_key = filled_dsm_memview
            ground_mask_memview_key = ground_mask_memview
        else:  # Create tifs
            dtm_path = manager.get_path("dezoom_dtm.tif", key="tmp")
            write(dtm, dtm_path, current_dtm_dezoom_profile)
            filled_dsm_memview_path = manager.get_path("dezoom_dsm.tif", key="tmp")
            write(filled_dsm_memview, filled_dsm_memview_path, current_dezoom_profile)
            if use_ground_mask:
                ground_mask_memview_path = manager.get_path("dezoom_ground_mask.tif", key="tmp")
                write(ground_mask_memview, ground_mask_memview_path, current_ground_mask_dezoom_profile)  # type: ignore
            else:
                ground_mask_memview_path = None
            dtm_key, filled_dsm_memview_key, ground_mask_memview_key = (
                dtm_path,
                filled_dsm_memview_path,
                ground_mask_memview_path,
            )
        del dtm, filled_dsm_memview, ground_mask_memview

        print("Drape cloth simulation...")

        drape_cloth_parameters: dict = {
            "num_outer_iterations": current_num_outer_iterations,
            "num_inner_iterations": num_inner_iterations,
            "step_scale": 1.0 / (2 ** (nb_levels - level)),
        }

        if manager.pool is None:
            # no multiprocessing
            if (
                isinstance(dtm_key, str)
                or isinstance(filled_dsm_memview_key, str)
                or isinstance(ground_mask_memview_key, str)
            ):
                raise ValueError("Without multiprocessing the inputs must be numpy arrays.")
            extracted_dtm_key = drape_cloth_filter_gradient(
                dtm_key, filled_dsm_memview_key, ground_mask_memview_key, **drape_cloth_parameters
            )
        else:
            # multiprocessing
            mp_parameters = {
                "image_height": current_dezoom_profile["height"],
                "image_width": current_dezoom_profile["width"],
                "output_profiles": [current_dtm_dezoom_profile],
                "output_keys": ["extracted_dtm.tif"],
                "func": drape_cloth_filter_gradient,
                "context_manager": manager,
                "stable_margin": int(
                    current_num_outer_iterations * num_inner_iterations * (3 / 2)
                ),  # 3 correspond to filter size
            }

            if use_ground_mask:
                [extracted_dtm_key] = mp_n_to_m_images(
                    inputs=[dtm_key, filled_dsm_memview_key, ground_mask_memview_key],
                    func_parameters=drape_cloth_parameters,
                    **mp_parameters,
                )
            else:
                drape_cloth_parameters["predicted_anchors"] = None
                [extracted_dtm_key] = mp_n_to_m_images(
                    inputs=[dtm_key, filled_dsm_memview_key], func_parameters=drape_cloth_parameters, **mp_parameters
                )

        if not manager.in_memory:
            os.remove(dtm_key)  # type: ignore
            os.remove(filled_dsm_memview_key)  # type: ignore
            if use_ground_mask:
                os.remove(ground_mask_memview_key)  # type: ignore

        dtm_key = extracted_dtm_key

        level -= 1
        current_num_outer_iterations = max(1, int(num_outer_iterations / 2 ** (nb_levels - 1 - level)))

    if not manager.in_memory:
        key = "dev" if manager.dev_mode else "tmp"
        dtm_path = manager.get_path(inter_dtm_filename, key=key)
        if isinstance(dtm_key, str):
            shutil.move(dtm_key, dtm_path)
        else:
            write(dtm_key, dtm_path, filled_dsm_profile)
        return dtm_path

    # else in memory
    if manager.dev_mode:
        dtm_path = manager.get_path(inter_dtm_filename, key="dev")
        write(dtm_key, dtm_path, filled_dsm_profile)

    return dtm_key
