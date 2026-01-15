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
This module is used to run multiprocessing in bulldozer
"""

import math
from multiprocessing.synchronize import Lock
from typing import Callable, List, Union

import numpy as np
import rasterio
import tqdm
from rasterio.windows import Window

from bulldozer.multiprocessing.bulldozer_manager import BulldozerContextManager
from bulldozer.multiprocessing.utils import MpTile, write_window


def compute_mp_strips(
    image_height: int,
    image_width: int,
    nb_workers: int,
    stable_margin: int,
    along_line: bool = False,
    specific_strip_size: Union[int, None] = None,
) -> List[MpTile]:
    """
    This method computes strips according to the image size and the number of workers.

    Args:
        image_height: height of the input image.
        image_width: width of the input image.
        nb_workers: number of workers.
        stable_margin: margin to add to each tile.
        along_line: horizontal strips (true) or vertical strips (false).
        specific_strip_size: specific strip size (by default size of the height/width divided by the number of workers).

    Returns:
        a list of strips
    """
    total_size = image_width if along_line else image_height
    if specific_strip_size:
        strip_size = specific_strip_size
    else:
        strip_size = total_size // nb_workers

    nb_tiles = total_size // strip_size
    if total_size % strip_size > 0:
        nb_tiles += 1

    strips = []

    for t in range(nb_tiles):
        start = t * strip_size
        end = min((t + 1) * strip_size - 1, total_size - 1)
        start_margin = stable_margin if start - stable_margin >= 0 else start
        end_margin: int = stable_margin if end + stable_margin <= total_size - 1 else total_size - 1 - end

        if along_line:
            strips.append(
                MpTile(
                    start_x=start,
                    start_y=0,
                    end_x=end,
                    end_y=image_height - 1,
                    top_margin=0,
                    right_margin=end_margin,
                    bottom_margin=0,
                    left_margin=start_margin,
                    height=image_height,
                    width=end - start + 1,
                    height_margin=image_height,
                    width_margin=(end + end_margin) - (start - start_margin) + 1,
                )
            )
        else:
            strips.append(
                MpTile(
                    start_x=0,
                    start_y=start,
                    end_x=image_width - 1,
                    end_y=end,
                    top_margin=start_margin,
                    right_margin=0,
                    bottom_margin=end_margin,
                    left_margin=0,
                    height=end - start + 1,
                    width=image_width,
                    height_margin=(end + end_margin) - (start - start_margin) + 1,
                    width_margin=image_width,
                )
            )

    return strips


def compute_mp_tiles(
    image_height: int,
    image_width: int,
    nb_workers: int,
    stable_margin: int,
    tile_mode: bool,
    specific_tile_size: Union[int, None] = None,
    strip_along_lines: bool = False,
) -> List[MpTile]:
    """
    Given an input image size and nb_workers, this method computes the list of strips that will be processed
    in parallel within a stream strip or tile

    Args:
        image_height: height of the input image.
        image_width: width of the input image.
        nb_workers: number of workers.
        stable_margin: margin to add to each tile.
        tile_mode: tiles (true) or strips (false).
        specific_tile_size: specific tile size (by default size of the input image divided by the number of workers).
        strip_along_lines: horizontal strips (true) or vertical strips (false).

    Returns:
        a list of strips
    """

    if tile_mode:

        nb_tiles_x: int = 0
        nb_tiles_y: int = 0
        end_x: int = 0
        start_y: int = 0
        end_y: int = 0
        top_margin: int = 0
        right_margin: int = 0
        bottom_margin: int = 0
        left_margin: int = 0

        # Force to make square tiles (except the last one unfortunately)
        nb_pixels_per_worker: int = (image_width * image_height) // nb_workers
        if specific_tile_size:
            tile_size = specific_tile_size
        else:
            tile_size = int(math.sqrt(nb_pixels_per_worker))
        nb_tiles_x = image_width // tile_size
        nb_tiles_y = image_height // tile_size
        if image_width % tile_size > 0:
            nb_tiles_x += 1
        if image_height % tile_size > 0:
            nb_tiles_y += 1

        strips: list = []

        for ty in range(nb_tiles_y):

            for tx in range(nb_tiles_x):

                # Determine the stable and unstable boundaries of the tile
                start_x = tx * tile_size
                start_y = ty * tile_size
                end_x = min((tx + 1) * tile_size - 1, image_width - 1)
                end_y = min((ty + 1) * tile_size - 1, image_height - 1)
                top_margin = stable_margin if start_y - stable_margin >= 0 else start_y
                left_margin = stable_margin if start_x - stable_margin >= 0 else start_x
                bottom_margin = stable_margin if end_y + stable_margin <= image_height - 1 else image_height - 1 - end_y
                right_margin = stable_margin if end_x + stable_margin <= image_width - 1 else image_width - 1 - end_x

                strips.append(
                    MpTile(
                        start_x=start_x,
                        start_y=start_y,
                        end_x=end_x,
                        end_y=end_y,
                        top_margin=top_margin,
                        right_margin=right_margin,
                        bottom_margin=bottom_margin,
                        left_margin=left_margin,
                        height=end_y - start_y + 1,
                        width=end_x - start_x + 1,
                        height_margin=(end_y + bottom_margin) - (start_y - top_margin) + 1,
                        width_margin=(end_x + right_margin) - (start_x - left_margin) + 1,
                    )
                )

    else:
        strips = compute_mp_strips(
            image_height=image_height,
            image_width=image_width,
            nb_workers=nb_workers,
            stable_margin=stable_margin,
            along_line=strip_along_lines,
            specific_strip_size=specific_tile_size,
        )

    return strips


def mp_n_to_m_images(
    inputs: list,
    image_height: int,
    image_width: int,
    output_keys: List[str],
    output_profiles: List[dict],
    context_manager: BulldozerContextManager,
    func: Callable,
    func_parameters: Union[dict, None] = None,
    stable_margin: int = 0,
    tile_mode: Union[bool, None] = None,
    specific_tile_size: Union[int, None] = None,
    strip_along_lines: bool = False,
    binary: bool = False,
) -> Union[List[str], List[np.ndarray]]:
    """
    Generic paradigm to process n images providing m resulting images using a paradigm similar to the old map/reduce

    func is processed in parallel with a multiprocessing starmap

    If tile_mode is set to False, the image will be cropped as strips.
    specific_tile_size: hotfix to handle dezoom in filling dsm method.
    If strip_along_line is set to True, those strips will be vertical.

    Strong hypothesis: all input image are in the same geometry and have the same size
    """

    if len(inputs) < 1:
        raise ValueError("At least one input image must be given.")

    if func is None:
        raise ValueError("A filter must be set !")

    if context_manager is None:
        raise ValueError("The Context Manager must be given !")

    if context_manager.pool is None:
        raise ValueError("The Context Manager must contain a pool of process !")

    # Sometimes filter does not need parameters
    if func_parameters is None:
        func_parameters = {}

    # compute the strips
    tiles = compute_mp_tiles(
        image_height=image_height,
        image_width=image_width,
        stable_margin=stable_margin,
        nb_workers=context_manager.nb_workers,
        tile_mode=tile_mode if tile_mode is not None else context_manager.tile_mode,
        specific_tile_size=specific_tile_size,
        strip_along_lines=strip_along_lines,
    )

    out: Union[List[str], List[np.ndarray]]
    if context_manager.in_memory:
        # inputs are numpy arrays
        list_input = [
            (
                [
                    inputs[i][
                        tile.start_y - tile.top_margin : tile.end_y + tile.bottom_margin + 1,
                        tile.start_x - tile.left_margin : tile.end_x + tile.right_margin + 1,
                    ]
                    for i in range(len(inputs))
                ],
                func,
                func_parameters,
                tile,
            )
            for tile in tiles
        ]

        out_chunks = context_manager.pool.starmap(mp_execute_from_arrays, tqdm.tqdm(list_input, total=len(list_input)))

        out = [
            np.zeros((image_height, image_width), dtype=output_profiles[i]["dtype"])
            for i in range(len(output_profiles))
        ]
        for chunk_res_dict in out_chunks:
            tile = chunk_res_dict["tile"]
            for i in range(len(output_profiles)):
                out[i][tile.start_y : tile.end_y + 1, tile.start_x : tile.end_x + 1] = chunk_res_dict["data"][i]

    else:
        # inputs are paths
        output_paths = [context_manager.get_path(output_key, key="tmp") for output_key in output_keys]
        list_input = [
            (inputs, output_paths, func, func_parameters, output_profiles, context_manager.lock, tile, binary)
            for tile in tiles
        ]
        context_manager.pool.starmap(mp_execute_from_paths, tqdm.tqdm(list_input))
        out = output_paths

    return out


def mp_execute_from_paths(
    input_paths: List[str],
    output_paths: List[str],
    func: Callable,
    func_parameters: dict,
    output_profiles: List[dict],
    lock: Lock,
    tile: MpTile,
    binary: bool,
) -> None:
    """
    This method is called within multiprocessing.
    It opens all input files along a window, execute the given function on it, then save the results in files.

    Args:
        input_paths: list of input file paths.
        output_paths: list of output file paths.
        func: callable to execute.
        func_parameters: all parameters for func.
        output_profiles: the rasterio profile for each output.
        lock: manager lock to avoid concurrent write accesses.
        tile: tile to process.
    """
    # Read the inputs
    inputs = []
    for path in input_paths:
        with rasterio.open(path) as src:
            inputs.append(
                src.read(
                    1,
                    window=Window(
                        col_off=tile.start_x - tile.left_margin,
                        row_off=tile.start_y - tile.top_margin,
                        width=tile.width_margin,
                        height=tile.height_margin,
                    ),
                )
            )

    # Run the function
    outputs = func(*inputs, **func_parameters)

    # Write the outputs
    if len(output_paths) > 1:
        for index, out_path in enumerate(output_paths):
            with lock:
                write_window(
                    outputs[index][
                        tile.top_margin : tile.top_margin + tile.height,
                        tile.left_margin : tile.left_margin + tile.width,
                    ],
                    out_path,
                    output_profiles[index],
                    tile,
                    binary,
                )
    else:
        with lock:
            write_window(
                outputs[
                    tile.top_margin : tile.top_margin + tile.height, tile.left_margin : tile.left_margin + tile.width
                ],
                output_paths[0],
                output_profiles[0],
                tile,
                binary,
            )


def mp_execute_from_arrays(inputs: List[np.ndarray], func: Callable, func_parameters: dict, tile: MpTile) -> dict:
    """
    This method is called within multiprocessing.
    It executes the given function on given input arrays.

    Args:
        inputs: list of input numpy arrays.
        func: callable to execute.
        func_parameters: all parameters for func.
        tile: tile to process.

    Returns:
        a dictionary containing the tile infos and the outputs without margin.

    """
    outputs = func(*inputs, **func_parameters)

    if not isinstance(outputs, tuple):
        return {
            "data": [
                outputs[
                    tile.top_margin : tile.top_margin + tile.height, tile.left_margin : tile.left_margin + tile.width
                ]
            ],
            "tile": tile,
        }

    return {
        "data": [
            output[tile.top_margin : tile.top_margin + tile.height, tile.left_margin : tile.left_margin + tile.width]
            for output in outputs
        ],
        "tile": tile,
    }
