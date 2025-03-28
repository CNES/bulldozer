#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022-2025 Centre National d'Etudes Spatiales (CNES).
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

from typing import Callable
import concurrent.futures
import tqdm
import numpy
import math
import copy

import bulldozer.eoscale.shared as eosh
import bulldozer.eoscale.utils as eotools
import bulldozer.eoscale.manager as eom


def isPowerOfTwo(x: int):
    # First x in the below expression
    # is for the case when x is 0
    return x and (not(x & (x - 1)))


def compute_mp_strips(image_height: int,
                      image_width: int,
                      nb_workers: int,
                      stable_margin: int,
                      along_line: bool = False) -> list:
    """
        Return a list of strips 
    """
    if along_line:
        strip_width = image_width // nb_workers
    else:
        strip_height = image_height // nb_workers

    strips = []
    if along_line:
        end_x: int = None
        end_y: int = image_height - 1
        start_x: int = None
        start_y: int = 0
    else:
        end_x: int = image_width - 1
        end_y: int = None
        start_x: int = 0
        start_y: int = None

    top_margin: int = None
    right_margin: int = 0
    bottom_margin: int = None
    left_margin: int = 0

    for worker in range(nb_workers):
        if along_line:
            start_x = worker * strip_width
        else:
            start_y = worker * strip_height

        if worker == nb_workers - 1:
            if along_line:
                end_x = image_width - 1
            else:
                end_y = image_height - 1
        else:
            if along_line:
                end_x = (worker + 1) * strip_width - 1
            else:
                end_y = (worker + 1) * strip_height - 1
        
        top_margin = stable_margin if start_y - stable_margin >= 0 else start_y
        bottom_margin = stable_margin if end_y + stable_margin <= image_height - 1 else image_height - 1 - end_y
        strips.append(eotools.MpTile(start_x=start_x,
                                     start_y=start_y,
                                     end_x=end_x,
                                     end_y=end_y,
                                     top_margin=top_margin,
                                     right_margin=right_margin, 
                                     bottom_margin=bottom_margin,
                                     left_margin=left_margin))

    return strips


def compute_mp_tiles(inputs: list,
                     stable_margin: int,
                     nb_workers: int,
                     tile_mode: bool,
                     context_manager: eom.EOContextManager,
                     specific_tile_size: int = None,
                     strip_along_lines: bool = False):
    """
        Given an input eoscale virtual path and nb_workers,
        this method computes the list of strips that will
        be processed in parallel within a stream strip or tile 
    """

    img_idx: int = 0
    image_width: int = None
    image_height: int = None
    for img_key in inputs:
        # Warning, the key can be either a memory view or a shared resource key
        arr = context_manager.get_array(key=img_key)
        if img_idx < 1:
            image_height = arr.shape[1]
            image_width = arr.shape[2]
        else:
            if image_width != arr.shape[2] or image_height != arr.shape[1]:
                raise ValueError("ERROR: all input images must have the same width and the same height !")
        img_idx += 1

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
                end_x = min((tx+1) * tile_size - 1, image_width - 1)
                end_y = min((ty+1) * tile_size - 1, image_height - 1)
                top_margin = stable_margin if start_y - stable_margin >= 0 else start_y
                left_margin = stable_margin if start_x - stable_margin >= 0 else start_x
                bottom_margin = stable_margin if end_y + stable_margin <= image_height - 1 else image_height - 1 - end_y
                right_margin = stable_margin if end_x + stable_margin <= image_width - 1 else image_width - 1 - end_x
                
                strips.append(eotools.MpTile(start_x=start_x,
                                             start_y=start_y,
                                             end_x=end_x,
                                             end_y=end_y,
                                             top_margin=top_margin,
                                             right_margin=right_margin, 
                                             bottom_margin=bottom_margin,
                                             left_margin=left_margin))
        
        return strips

    else:
        return compute_mp_strips(image_height=image_height,
                                 image_width=image_width,
                                 nb_workers=nb_workers,
                                 stable_margin=stable_margin,
                                 along_line=strip_along_lines)


def default_generate_output_profiles(input_profiles: list) -> list:
    """
        This method makes a deep copy of the input profiles 
    """
    return [copy.deepcopy(input_profile) for input_profile in input_profiles]


def allocate_outputs(profiles: list,
                     context_manager: eom.EOContextManager) -> list:
    """
        Given a list of profiles, this method creates
        shared memory instances of the outputs
    """

    output_eoshared_instances: list = [eosh.EOShared() for i in range(len(profiles))]

    for i in range(len(profiles)):
        output_eoshared_instances[i].create_array(profile=profiles[i])
        # Be careful to not close theses shared instances, because they are referenced in
        # the context manager.
        context_manager.shared_resources[output_eoshared_instances[i].virtual_path] = output_eoshared_instances[i]

    return output_eoshared_instances


def execute_filter_n_images_to_n_images(image_filter: Callable,
                                        filter_parameters: dict,
                                        inputs: list,
                                        tile: eotools.MpTile,
                                        context_manager: eom.EOContextManager) -> tuple:
    
    """
        This method execute the filter on the inputs and then extract the stable 
        area from the resulting outputs before returning them.
    """

    # Get references to input numpy array buffers
    input_buffers = []
    input_profiles = []
    for i in range(len(inputs)):
        input_buffers.append(context_manager.get_array(key=inputs[i], tile=tile))
        input_profiles.append(context_manager.get_profile(key=inputs[i]))

    output_buffers = image_filter(input_buffers, input_profiles, filter_parameters)

    if not isinstance(output_buffers, list):
        if not isinstance(output_buffers, numpy.ndarray):
            raise ValueError("Output of the image filter must be either a Python list or a numpy array")
        else:
            output_buffers = [output_buffers]

    # Reshape some output buffers if necessary since even for one channel image eoscale
    # needs a shape like this (channel, height, width) and it is really shitty to ask 
    # the developer to take care of this...
    for o in range(len(output_buffers)):
        if len(output_buffers[o].shape) == 2:
            output_buffers[o] = output_buffers[o].reshape((1, output_buffers[o].shape[0], output_buffers[o].shape[1]))
        # We need to check now that input image dimensions are the same of outputs
        if output_buffers[o].shape[1] != input_buffers[0].shape[1] or output_buffers[o].shape[2] != input_buffers[0].shape[2]:
            raise ValueError("ERROR: Output images must have the same height and width of input images for this filter !")

    stable_start_x: int = None
    stable_start_y: int = None
    stable_end_x: int = None
    stable_end_y: int = None

    for i in range(len(output_buffers)):
        stable_start_x = tile.left_margin
        stable_start_y = tile.top_margin
        stable_end_x = stable_start_x + tile.end_x - tile.start_x + 1
        stable_end_y = stable_start_y + tile.end_y - tile.start_y + 1
        output_buffers[i] = output_buffers[i][:, stable_start_y:stable_end_y, stable_start_x:stable_end_x]
    
    return output_buffers, tile


def default_reduce(outputs: list, 
                   chunk_output_buffers: list, 
                   tile: eotools.MpTile) -> None:
    """ Fill the outputs buffer with the results provided by the map filter from a strip """
    for c in range(len(chunk_output_buffers)):
        outputs[c][:, tile.start_y: tile.end_y + 1, tile.start_x : tile.end_x + 1] = chunk_output_buffers[c][:, :, :]


def n_images_to_m_images_filter(inputs: list = None, 
                                image_filter: Callable = None,
                                filter_parameters: dict = None,
                                generate_output_profiles: Callable = None,
                                output_map = None,
                                concatenate_filter: Callable = None,
                                stable_margin: int = 0,
                                context_manager: eom.EOContextManager = None,
                                filter_desc: str = "N Images to M images MultiProcessing...",
                                tile_mode: bool = None,
                                specific_tile_size : int = None,
                                strip_along_lines: bool = None) -> list:
    """
        Generic paradigm to process n images providing m resulting images using a paradigm
        similar to the good old map/reduce

        image_filter is processed in parallel

        generate_output_profiles is a callable taking as input a list of rasterio profiles and the dictionnary
        filter_parameters and returning a list of output profiles. 
        This callable is used by EOScale to allocate new shared images given their profile. It determines the value m
        of the n_image_to_m_image executor.

        concatenate_filter is processed by the master node to aggregate results

        If tile_mode is set to False, the image will be cropped as strips.
        specific_tile_size: hotfix to handle dezoom in filling dsm method.
        If strip_along_line is set to True, those strips will be vertical.

        Strong hypothesis: all input image are in the same geometry and have the same size
    """

    if len(inputs) < 1:
        raise ValueError("At least one input image must be given.")
    
    if image_filter is None:
        raise ValueError("A filter must be set !")

    if context_manager is None:
        raise ValueError("The EOScale Context Manager must be given !")

    # Sometimes filter does not need parameters    
    if filter_parameters is None:
        filter_parameters = dict()
    
    # compute the strips
    tiles = compute_mp_tiles(inputs=inputs,
                             stable_margin=stable_margin,
                             nb_workers=context_manager.nb_workers,
                             tile_mode=tile_mode if tile_mode is not None else context_manager.tile_mode,
                             context_manager=context_manager,
                             specific_tile_size = specific_tile_size,
                             strip_along_lines=strip_along_lines)

    # Call the generate output profile callable. Use the default one
    # if the developer did not assign one
    output_profiles: list = []
    if generate_output_profiles is None:
        for key in inputs:
            output_profiles.append(context_manager.get_profile(key=key))    
    else:
        copied_input_mtds: list = []
        for key in inputs:
            copied_input_mtds.append(context_manager.get_profile(key=key))
        output_profiles = generate_output_profiles( copied_input_mtds, filter_parameters)
        if not isinstance(output_profiles, list):
            output_profiles = [output_profiles]

    # Allocate and share the outputs
    output_eoshareds = allocate_outputs(profiles=output_profiles,
                                        context_manager=context_manager)
    
    outputs = [ eoshared_inst.get_array() for eoshared_inst in output_eoshareds]

    # For debug, comment this section below in production
    # for tile in tiles:
    #     print("process tile ", tile)
    #     chunk_output_buffers, tile = execute_filter_n_images_to_n_images(image_filter,
    #                                                         filter_parameters,
    #                                                         inputs,
    #                                                         tile)
    #     default_reduce(outputs, chunk_output_buffers, tile )

    # # Multi processing execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(context_manager.nb_workers, len(tiles))) as executor:

        futures = { executor.submit(execute_filter_n_images_to_n_images,
                                    image_filter,
                                    filter_parameters,
                                    inputs,
                                    tile,
                                    context_manager) for tile in tiles}
        
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=filter_desc):

            chunk_output_buffers, tile = future.result()
            if concatenate_filter is None:
                default_reduce(outputs, chunk_output_buffers, tile)
            else : 
                concatenate_filter(outputs, chunk_output_buffers, tile)

    output_virtual_paths = [eoshared_inst.virtual_path for eoshared_inst in output_eoshareds]
    
    return output_virtual_paths


def execute_filter_n_images_to_m_scalars(image_filter: Callable,
                                         filter_parameters: dict,
                                         inputs: list,
                                         tile: eotools.MpTile,
                                         context_manager: eom.EOContextManager) -> tuple:
    
    """
        This method execute the filter on the inputs and then extract the stable
        area from the resulting outputs before returning them.
    """
    # Get references to input numpy array buffers
    input_buffers = []
    input_profiles = []
    for i in range(len(inputs)):
        input_buffers.append(context_manager.get_array(key=inputs[i], tile = tile))
        input_profiles.append(context_manager.get_profile(key=inputs[i]))

    output_scalars = image_filter(input_buffers, input_profiles, filter_parameters)

    if not isinstance(output_scalars, list):
        output_scalars = [output_scalars]
    
    return output_scalars, tile


def n_images_to_m_scalars(inputs: list = None, 
                          image_filter: Callable = None,
                          filter_parameters: dict = None,
                          nb_output_scalars: int = None,
                          output_scalars: list = None,
                          concatenate_filter: Callable = None,
                          context_manager: eom.EOContextManager = None,
                          filter_desc: str = "N Images to M Scalars MultiProcessing...") -> list:
    """
        Generic paradigm to process n images providing m resulting scalars using a paradigm
        similar to the good old map/reduce

        image_filter is processed in parallel

        concatenate_filter is processed by the master node to aggregate results

        Strong hypothesis: all input image are in the same geometry and have the same size 
    """

    if len(inputs) < 1:
        raise ValueError("At least one input image must be given.")
    
    if image_filter is None:
        raise ValueError("A filter must be set !")
    
    if concatenate_filter is None:
        raise ValueError("A concatenate filter must be set !")
    
    if nb_output_scalars is None:
        raise ValueError("The number of output scalars must be set (integer value expected) !")
    
    if context_manager is None:
        raise ValueError("The EOScale Context Manager must be given !")

    # Sometimes filter does not need parameters    
    if filter_parameters is None:
        filter_parameters = dict()
    
    # compute the strips
    tiles = compute_mp_tiles(inputs=inputs,
                             stable_margin=0,
                             nb_workers=context_manager.nb_workers,
                             tile_mode=context_manager.tile_mode,
                             context_manager=context_manager)

    # Initialize the output scalars if the user doesn't provide it 
    if output_scalars is None:
        output_scalars: list = [0.0 for i in range(nb_output_scalars) ]

    with concurrent.futures.ProcessPoolExecutor(max_workers= min(context_manager.nb_workers, len(tiles))) as executor:

        futures = {executor.submit(execute_filter_n_images_to_m_scalars,
                                    image_filter,
                                    filter_parameters,
                                    inputs,
                                    tile,
                                    context_manager,
                                   ) for tile in tiles}
        
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=filter_desc):

            chunk_output_scalars, tile = future.result()
            concatenate_filter(output_scalars, chunk_output_scalars, tile)

    return output_scalars
