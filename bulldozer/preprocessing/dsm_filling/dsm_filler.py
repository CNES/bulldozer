#!/usr/bin/env python
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

import numpy as np
from scipy.ndimage import zoom

from bulldozer.eomultiprocessing.bulldozer_executor import mp_n_to_m_images
from bulldozer.eomultiprocessing.bulldozer_manager import BulldozerContextManager
from bulldozer.eomultiprocessing.utils import read, write
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
    dsm[:] = fill_process.iterative_filling(
        dsm_strip=dsm, nodata_value=nodata, nb_it=filling_iterations, border_nodata_strip=border_mask
    )

    return dsm


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
    dsm[:] = fill_process.iterative_filling(dsm_strip=dsm, nodata_value=nodata, nb_it=filling_iterations)
    # Putting nan values instead of no data for the sampling function
    dsm[dsm == nodata] = np.nan

    return dsm


@Runtime
def fill_dsm(
    dsm_key: str | np.ndarray,
    regular_mask_key: str | np.ndarray,
    border_nodata_mask_key: str | np.ndarray,
    dsm_profile: dict,
    nodata: float,
    max_object_size: int,
    manager: BulldozerContextManager,
) -> str | np.ndarray:
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
        manager.add_out_directory(filling_dev, key=manager.debug_key)
    if not manager.in_memory:
        manager.add_out_directory(filling_dev, key=manager.tmp_key)

    filled_dsm_profile = dsm_profile.copy()
    filled_dsm_profile.update({"driver": "GTiff", "interleave": "band", "nodata": nodata})

    filled_dsm = read(dsm_key) if isinstance(dsm_key, str) else dsm_key

    # We're also filling the irregular areas
    regular_mask = read(regular_mask_key) if isinstance(regular_mask_key, str) else regular_mask_key
    filled_dsm[regular_mask == 0] = nodata
    del regular_mask

    filled_dsm_with_regular_path = ""
    if manager.dev_mode or not manager.in_memory:
        filled_dsm_with_regular_path = manager.write_tif(
            filled_dsm, f"{filling_dev}/regular_dsm.tif", filled_dsm_profile
        )

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
    border_nodata_mask = (
        read(border_nodata_mask_key) if isinstance(border_nodata_mask_key, str) else border_nodata_mask_key
    )

    remaining_nodata = (filled_dsm == nodata) & (border_nodata_mask == 0)

    # if nodata areas are still in the DSM
    has_nodata = np.any(remaining_nodata)
    dezoom_level = 0
    downsample = True
    sampling = "downsample"

    filled_dsm_key: str | np.ndarray
    filled_dsm_downsampled_key: str | np.ndarray

    # Downsample until there is nodata remaining or reaching max level
    while has_nodata and 0 <= dezoom_level <= nb_max_level:
        if dezoom_level == 0:  # When level is 0
            if manager.in_memory:
                filled_dsm_key = filled_dsm
            else:
                if downsample:
                    filled_dsm_key = filled_dsm_with_regular_path
                else:
                    # Upsample case
                    filled_dsm_key = manager.write_tif(
                        filled_dsm, "tmp_filled_dsm.tif", filled_dsm_profile, key=manager.tmp_key
                    )
            del filled_dsm

            # First iterative filling for small no data areas
            BulldozerLogger.log("Iterative filling DSM level 0", logging.INFO)
            # multiprocessing
            [filled_dsm_key] = mp_n_to_m_images(
                inputs=[filled_dsm_key, border_nodata_mask_key],
                image_height=dsm_profile["height"],
                image_width=dsm_profile["width"],
                output_profiles=[filled_dsm_profile],
                output_keys=[f"{filling_dev}/filled_dsm_{sampling}_level_0.tif"],
                func=fill_dsm_method_with_border_mask,
                func_parameters={"nodata": nodata, "filling_iterations": filling_iterations},
                stable_margin=filling_iterations,
                context_manager=manager,
                debug=True,
            )

            filled_dsm = read(filled_dsm_key) if isinstance(filled_dsm_key, str) else filled_dsm_key
            del filled_dsm_key

            # Identifying the remaining inner no data areas
            remaining_nodata[:] = (filled_dsm == nodata) & (border_nodata_mask == 0)

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
                    + f"computed_specific_tile_size={int(np.ceil(dsm_resolution * dezoom_factor**dezoom_level))} / "
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
                    + f"computed_specific_tile_size={int(np.ceil(dsm_resolution * dezoom_factor**dezoom_level))} / "
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

            if manager.in_memory:
                filled_dsm_downsampled_key = filled_dsm_downsampled
            else:
                tmp_filled_dsm_resample_path = manager.get_path("tmp_filled_dsm_resampled.tif", key=manager.tmp_key)
                if os.path.exists(tmp_filled_dsm_resample_path):
                    os.remove(tmp_filled_dsm_resample_path)
                write(filled_dsm_downsampled, tmp_filled_dsm_resample_path, downsampled_profile)
                filled_dsm_downsampled_key = tmp_filled_dsm_resample_path
            del filled_dsm_downsampled

            [filled_dsm_downsampled_key] = mp_n_to_m_images(
                inputs=[filled_dsm_downsampled_key],
                image_height=downsampled_profile["height"],
                image_width=downsampled_profile["width"],
                output_profiles=[downsampled_profile],
                output_keys=[f"{filling_dev}/filled_dsm_{sampling}d_level_{dezoom_level}.tif"],
                func=fill_dsm_method,
                func_parameters={"nodata": nodata, "filling_iterations": filling_iterations},
                stable_margin=filling_iterations,
                context_manager=manager,
                specific_tile_size=specific_tile_size,
                debug=True,
            )

            filled_dsm_downsampled = (
                read(filled_dsm_downsampled_key)
                if isinstance(filled_dsm_downsampled_key, str)
                else filled_dsm_downsampled_key
            )
            del filled_dsm_downsampled_key

            # Merging the current level with the first one
            scale_y = filled_dsm.shape[0] / filled_dsm_downsampled.shape[0]
            scale_x = filled_dsm.shape[1] / filled_dsm_downsampled.shape[1]
            filled_dsm_downsampled = zoom(filled_dsm_downsampled, (scale_y, scale_x), order=1, mode="nearest")

            filled_dsm[:] = np.where(remaining_nodata == 1, filled_dsm_downsampled, filled_dsm)
            del filled_dsm_downsampled

            remaining_nodata[:] = (np.isnan(filled_dsm)) & (border_nodata_mask == 0)

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
                sampling = "upsample"

    # Set the border nodata to very high value in order to avoid underestimation of the DTM on the border
    # TODO change to another method?
    filled_dsm[border_nodata_mask == 1] = 9999

    final_filled_dsm_filename = "filled_dsm.tif"

    if not manager.in_memory:
        final_filled_dsm_path = manager.write_tif(filled_dsm, final_filled_dsm_filename, dsm_profile)
        return final_filled_dsm_path

    # else in memory
    if manager.dev_mode:
        manager.write_tif(filled_dsm, final_filled_dsm_filename, dsm_profile)

    return filled_dsm
