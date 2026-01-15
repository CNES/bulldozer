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
This module is used to prefill the input DSM before the DTM extraction.
"""

import logging
import math
import os
import shutil
from typing import Union

import numpy as np
import rasterio
from scipy.ndimage import zoom

from bulldozer.multiprocessing.bulldozer_executor import mp_n_to_m_images
from bulldozer.multiprocessing.bulldozer_manager import BulldozerContextManager
from bulldozer.multiprocessing.utils import write
from bulldozer.preprocessing import fill  # type: ignore
from bulldozer.utils.bulldozer_logger import BulldozerLogger, Runtime
from bulldozer.utils.helper import downsample_profile


def fill_dsm_method_with_border_mask(
    dsm: np.ndarray, border_mask: np.ndarray, nodata: float, filling_iterations: float
) -> np.ndarray:
    """
    This method is used in the main `fill_dsm_process`.
    It calls the Cython method to fill a DSM with a border nodata mask.

     Args:
        dsm: input DSM.
        border_mask: border nodata mask.
        nodata: DSM nodata value (if nan, the nodata is set to -32768).
        filling_iterations: the number of iterations.

    Returns:
        regular areas mask.
    """

    fill_process = fill.PyFill()
    dsm = fill_process.iterative_filling(
        dsm_strip=dsm, nodata_value=nodata, nb_it=filling_iterations, border_nodata_strip=border_mask
    )

    return dsm.astype(np.float32)


def fill_dsm_method(dsm: np.ndarray, nodata: float, filling_iterations: float) -> np.ndarray:
    """
    This method is used in the main `fill_dsm_process`.
    It calls the Cython method to fill a DSM.

    Args:
        dsm: input DSM.
        nodata: DSM nodata value (if nan, the nodata is set to -32768).
        filling_iterations: the number of iterations.

    Returns:
        regular areas mask.
    """
    fill_process = fill.PyFill()
    dsm = fill_process.iterative_filling(dsm_strip=dsm, nodata_value=nodata, nb_it=filling_iterations)
    # Putting nan values instead of no data for the sampling function
    dsm[dsm == nodata] = np.nan

    return dsm.astype(np.float32)


@Runtime
def fill_dsm(
    dsm_key: Union[str, np.ndarray],
    regular_mask_key: Union[str, np.ndarray],
    border_nodata_mask_key: Union[str, np.ndarray],
    dsm_profile: dict,
    nodata: float,
    max_object_size: int,
    manager: BulldozerContextManager,
) -> Union[str, np.ndarray]:
    """
    This fills the nodata of the input DSM for the following dtm extraction step.

    Args:
        dsm_key: input DSM (numpy array or path to file).
        regular_mask_key: regular mask (numpy array or path to file).
        border_nodata_mask_key: border nodata mask (numpy array or path to file).
        dsm_profile: input DSM profile.
        nodata: DSM nodata value (if nan, the nodata is set to -32768).
        max_object_size: max object size
        manager: bulldozer context manager.

    Returns:
        the filled DSM and its profile
    """
    filling_dev = "filling_DSM"
    if manager.dev_mode:
        manager.add_out_directory(filling_dev, key="dev")

    filled_dsm_profile = dsm_profile.copy()
    filled_dsm_profile.update({"driver": "GTiff", "interleave": "band", "nodata": nodata})

    if isinstance(dsm_key, str):
        with rasterio.open(dsm_key) as src:
            filled_dsm = src.read(1)
    else:
        filled_dsm = dsm_key
    del dsm_key

    # We're also filling the irregular areas
    if isinstance(regular_mask_key, str):
        with rasterio.open(regular_mask_key) as regular:
            filled_dsm[regular.read(1) == 0] = nodata
    else:
        filled_dsm[regular_mask_key == 0] = nodata
    del regular_mask_key

    filled_dsm_with_regular_filename = "regular_dsm.tif"
    if manager.dev_mode:
        filled_dsm_with_regular_path = manager.get_path(filled_dsm_with_regular_filename, key=filling_dev)
        write(filled_dsm, filled_dsm_with_regular_path, filled_dsm_profile)
    else:
        if not manager.in_memory:
            filled_dsm_with_regular_path = manager.get_path(filled_dsm_with_regular_filename, key="tmp")
            write(filled_dsm, filled_dsm_with_regular_path, filled_dsm_profile)

    # Setting parameters for the DSM filling method
    dsm_resolution = dsm_profile["transform"][0]
    # if max_object_size is less than 2+sqrt(2) overrides the computed value to avoid information loss during dezoom
    filling_iterations = int(np.max([int(2 + np.sqrt(2)), np.floor((max_object_size / dsm_resolution) / 2)]))
    # Nb iterations = max_object_size (px) / 2 (allow to fill a hole between two points max_object_size apart)

    # if computed value for dezoom_factor is less than 2
    # overrides the value with 2 to ensure a dezoom during the filling process
    dezoom_factor = int(
        np.max([2, np.floor(filling_iterations * (2 - np.sqrt(2)))])
    )  # sqrt(2) to handle the diagonal neighbors
    nb_max_level = int(
        np.floor(math.log(np.min([filled_dsm.shape[0], filled_dsm.shape[1]]), dezoom_factor))
    )  # limits the number of dezoom iterations
    BulldozerLogger.log(
        "DSM filling parameters : "
        + f"filling_iterations={filling_iterations} / "
        + f"dezoom_factor={dezoom_factor} / "
        + f"nb_max_level={nb_max_level}",
        logging.DEBUG,
    )

    # Identifying the remaining inner no data areas
    if isinstance(border_nodata_mask_key, str):
        with rasterio.open(border_nodata_mask_key) as src:
            border_nodata_mask = src.read(1)
    else:
        border_nodata_mask = border_nodata_mask_key
    remaining_nodata = (filled_dsm == nodata) & (border_nodata_mask == 0)

    # if nodata areas are still in the DSM
    has_nodata = np.any(remaining_nodata)
    dezoom_level = 0
    downsample = True

    filled_dsm_key: Union[str, np.ndarray]
    filled_dsm_downsampled_key: Union[str, np.ndarray]

    # Downsample until there is nodata remaining or reaching max level
    while has_nodata and 0 <= dezoom_level <= nb_max_level:
        if dezoom_level == 0:  # When level is 0
            filled_dsm_1stpass_filename = (
                "filled_dsm_downsample_level_0.tif" if downsample else "filled_dsm_upsample_level_0.tif"
            )
            if manager.in_memory:
                filled_dsm_key = filled_dsm
            else:
                if downsample:
                    filled_dsm_key = filled_dsm_with_regular_path
                else:
                    # Upsample case
                    tmp_filled_dsm_path = manager.get_path("tmp_filled_dsm.tif", key="tmp")
                    write(filled_dsm, tmp_filled_dsm_path, filled_dsm_profile)
                    filled_dsm_key = tmp_filled_dsm_path
            del filled_dsm

            # First iterative filling for small no data areas
            BulldozerLogger.log("Iterative filling DSM level 0", logging.INFO)
            fill_dsm_parameters = {"nodata": nodata, "filling_iterations": filling_iterations}
            if manager.pool is None:
                # no multiprocessing
                if isinstance(filled_dsm_key, str) or isinstance(border_nodata_mask_key, str):
                    raise ValueError("Without multiprocessing the inputs must be numpy arrays.")
                filled_dsm_key = fill_dsm_method_with_border_mask(
                    filled_dsm_key, border_nodata_mask_key, **fill_dsm_parameters
                )
            else:
                # multiprocessing
                [filled_dsm_key] = mp_n_to_m_images(
                    inputs=[filled_dsm_key, border_nodata_mask_key],
                    image_height=dsm_profile["height"],
                    image_width=dsm_profile["width"],
                    output_profiles=[filled_dsm_profile],
                    output_keys=[filled_dsm_1stpass_filename],
                    func=fill_dsm_method_with_border_mask,
                    func_parameters=fill_dsm_parameters,
                    stable_margin=filling_iterations,
                    context_manager=manager,
                )

            if manager.dev_mode:
                filled_dsm_1stpass_path = manager.get_path(filled_dsm_1stpass_filename, key=filling_dev)
                if isinstance(filled_dsm_key, np.ndarray):
                    write(filled_dsm_key, filled_dsm_1stpass_path, filled_dsm_profile)
                else:  # already saved in tmp folder
                    shutil.move(filled_dsm_key, filled_dsm_1stpass_path)
                    filled_dsm_key = filled_dsm_1stpass_path

            if isinstance(filled_dsm_key, str):
                with rasterio.open(filled_dsm_key) as src:
                    filled_dsm = src.read(1)
            else:
                filled_dsm = filled_dsm_key
            del filled_dsm_key

            # Identifying the remaining inner no data areas
            remaining_nodata = (filled_dsm == nodata) & (border_nodata_mask == 0)

            # if nodata areas are still in the DSM
            has_nodata = np.any(remaining_nodata)

            if downsample:
                dezoom_level += 1
            else:
                dezoom_level -= 1

        else:  # For every other level than 0 (with downsampling)
            # Putting nan values instead of no data for the sampling function
            filled_dsm[filled_dsm == nodata] = np.nan

            # Downsample the DSM to fill the large nodata areas.
            # The order is set to 1 because it's the only one that handle no data
            # The mode is set to nearest because it expands the image when zooming if the resampling factor is not
            # proportional to the image size
            filled_dsm_downsampled = zoom(filled_dsm, 1 / (dezoom_factor**dezoom_level), order=1, mode="nearest")
            # Putting back nodata values
            filled_dsm_downsampled = np.where(np.isnan(filled_dsm_downsampled), nodata, filled_dsm_downsampled)

            # Creating new profile for downsampled data
            downsampled_profile = downsample_profile(profile=dsm_profile, factor=dezoom_factor**dezoom_level)
            downsampled_profile.update(
                width=np.shape(filled_dsm_downsampled)[1], height=np.shape(filled_dsm_downsampled)[0]
            )

            # pylint: disable=line-too-long
            if downsample:
                # TODO HOTFIX to remove: until we change eoscale we have to compute the tile size manually
                BulldozerLogger.log(
                    "DSM filling during downsampling step : "
                    + f"level={dezoom_level} / "
                    + f"computed_specific_tile_size={int(np.ceil(dsm_resolution * dezoom_factor ** dezoom_level))} / "
                    + f"default_tile_size={int(math.sqrt((filled_dsm.shape[0] * filled_dsm.shape[1]) // manager.nb_workers))}",
                    logging.DEBUG,
                )

            else:
                # The number of iteration is set to the maximum at the current resolution
                # (we consider the max distance to reach with current resolution considering the diagional of the image)
                filling_iterations = int(
                    np.floor(
                        np.sqrt(filled_dsm.shape[0] ** 2 + filled_dsm.shape[1] ** 2) // dezoom_factor**dezoom_level
                    )
                )
                # TODO HOTFIX to remove: until we change eoscale we have to compute the tile size manually
                BulldozerLogger.log(
                    "DSM filling during upsampling step : "
                    + f"level={dezoom_level} / "
                    + f"computed_specific_tile_size={int(np.ceil(dsm_resolution * dezoom_factor ** dezoom_level))} / "
                    + f"default_tile_size={int(math.sqrt((filled_dsm.shape[0] * filled_dsm.shape[1]) // manager.nb_workers))} / "
                    + f"filling_iterations={filling_iterations}",
                    logging.DEBUG,
                )
            # pylint: enable=line-too-long

            if np.ceil(dsm_resolution * dezoom_factor**dezoom_level) > math.sqrt(
                (filled_dsm.shape[0] * filled_dsm.shape[1]) // manager.nb_workers
            ):
                # TODO HOTFIX to remove: until we change eoscale we have to compute the tile size manually
                specific_tile_size = int(np.ceil(dsm_resolution * dezoom_factor**dezoom_level))
            else:
                specific_tile_size = None

            # Iterative filling for the remaining no data areas
            BulldozerLogger.log("Iterative filling DSM level " + str(dezoom_level), logging.INFO)
            filled_dsm_downsampled_filename = (
                f"filled_dsm_downsampled_level_{dezoom_level}.tif"
                if downsample
                else f"filled_dsm_upsampled_level_{dezoom_level}.tif"
            )

            if manager.in_memory:
                filled_dsm_downsampled_key = filled_dsm_downsampled
            else:
                tmp_filled_dsm_resample_path = manager.get_path("tmp_filled_dsm_resampled.tif", key="tmp")
                if os.path.exists(tmp_filled_dsm_resample_path):
                    os.remove(tmp_filled_dsm_resample_path)
                write(filled_dsm_downsampled, tmp_filled_dsm_resample_path, downsampled_profile)
                filled_dsm_downsampled_key = tmp_filled_dsm_resample_path
            del filled_dsm_downsampled

            fill_dsm_parameters = {"nodata": nodata, "filling_iterations": filling_iterations}
            if manager.pool is None:
                # no multiprocessing
                if isinstance(filled_dsm_downsampled_key, str):
                    raise ValueError("Without multiprocessing the input DSM must be a numpy array.")
                filled_dsm_downsampled_key = fill_dsm_method(filled_dsm_downsampled_key, **fill_dsm_parameters)
            else:
                # multiprocessing
                [filled_dsm_downsampled_key] = mp_n_to_m_images(
                    inputs=[filled_dsm_downsampled_key],
                    image_height=downsampled_profile["height"],
                    image_width=downsampled_profile["width"],
                    output_profiles=[downsampled_profile],
                    output_keys=[filled_dsm_downsampled_filename],
                    func=fill_dsm_method,
                    func_parameters=fill_dsm_parameters,
                    stable_margin=filling_iterations,
                    context_manager=manager,
                    specific_tile_size=specific_tile_size,
                )

            if manager.dev_mode:
                filled_dsm_downsampled_path = manager.get_path(filled_dsm_downsampled_filename, key=filling_dev)
                if isinstance(filled_dsm_downsampled_key, np.ndarray):
                    write(filled_dsm_downsampled_key, filled_dsm_downsampled_path, downsampled_profile)
                else:  # already saved in tmp folder
                    shutil.move(filled_dsm_downsampled_key, filled_dsm_downsampled_path)
                    filled_dsm_downsampled_key = filled_dsm_downsampled_path

            if isinstance(filled_dsm_downsampled_key, str):
                with rasterio.open(filled_dsm_downsampled_key) as src:
                    filled_dsm_downsampled = src.read(1)
            else:
                filled_dsm_downsampled = filled_dsm_downsampled_key
            del filled_dsm_downsampled_key

            # Merging the current level with the first one
            scale_y = filled_dsm.shape[0] / filled_dsm_downsampled.shape[0]
            scale_x = filled_dsm.shape[1] / filled_dsm_downsampled.shape[1]
            filled_dsm_downsampled = zoom(filled_dsm_downsampled, (scale_y, scale_x), order=1, mode="nearest")

            filled_dsm[:] = np.where(remaining_nodata == 1, filled_dsm_downsampled, filled_dsm)

            del filled_dsm_downsampled

            remaining_nodata = (np.isnan(filled_dsm)) & (border_nodata_mask == 0)

            has_nodata = np.any(remaining_nodata)

            filled_dsm[:] = np.where(np.isnan(filled_dsm), nodata, filled_dsm)

            if downsample:
                dezoom_level += 1
            else:
                dezoom_level -= 1

            if dezoom_level == nb_max_level:
                # For upsampling we prefer to start at the level just after the max_level
                dezoom_level -= 2
                downsample = False

    # Set the border nodata to very high value in order to avoid underestimation of the DTM on the border
    # TODO change to another method?
    filled_dsm[border_nodata_mask == 1] = 9999

    final_filled_dsm_filename = "filled_dsm.tif"

    if not manager.in_memory:
        key = "dev" if manager.dev_mode else "tmp"
        final_filled_dsm_path = manager.get_path(final_filled_dsm_filename, key=key)
        write(filled_dsm, final_filled_dsm_path, dsm_profile)
        return final_filled_dsm_path

    # else in memory
    if manager.dev_mode:
        final_filled_dsm_path = manager.get_path(final_filled_dsm_filename, key="dev")
        write(filled_dsm, final_filled_dsm_path, dsm_profile)

    return filled_dsm
