#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022 Centre National d'Etudes Spatiales (CNES).
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
    This module is used to postprocess the DTM in order to improve its quality. It required a DTM generated from Bulldozer.
"""
import os
import sys
import logging
from shutil import copy
from datetime import datetime
from sys import stdout
import argparse
import multiprocessing
import argcomplete
import numpy as np
import rasterio
from bulldozer.utils.config_parser import ConfigParser
from bulldozer.utils.logging_helper import BulldozerLogger
from bulldozer.utils.helper import Runtime, DefaultValues

import bulldozer.eoscale.manager as eom

# Preprocessing steps of Bulldozer
import bulldozer.preprocessing.outliers.histogram as preprocess_histogram_outliers
import bulldozer.preprocessing.regular_detection.regular_detector as preprocess_regular_detector
import bulldozer.preprocessing.fill.prefill_dsm as prefill_dsm

# Drape cloth filter
import bulldozer.extraction.drape_cloth as dtm_extraction

# Postprocessing steps of Bulldozer
import bulldozer.postprocessing.ground_detection.post_anchorage_detection as postprocess_anchorage

__version__ = "2.0.0"



@Runtime
def dsm_to_dtm(config_path : str = None, **kwargs) -> None:
    """
        Main pipeline orchestrator.
        
        Args:
            config_path: path to the config file (YAML file expected, refers to the provided template in /conf).
            **kwargs: bulldozer parameters (used if the user don't provide a configuration file).

    """
    # Retrieves Bulldozer settings from the config file, the CLI parameters or the Python API parameters
    params = retrieve_params(config_path, **kwargs)


    # If the target output directory does not exist, creates it
    if not os.path.isdir(params['output_dir']):
        os.makedirs(params['output_dir']) 

    logger = BulldozerLogger.getInstance(logger_file_path=os.path.join(params['output_dir'], "trace_" + datetime.now().strftime("%d.%m.%Y_%H:%M:%S") +".log"))

    BulldozerLogger.log("Bulldozer input parameters: \n" + "".join("\t- " + str(key) +": " + str(value) + "\n" for key, value in params.items()), logging.DEBUG)

    with eom.EOContextManager(nb_workers = params['nb_max_workers'], tile_mode = True) as eomanager:

        # Open the input dsm that might be noisy and full of nodatas...
        input_dsm_key = eomanager.open_raster(raster_path = params['dsm_path'])

        # Step 1
        # Compute the height histogram of the input DSM with a bin width equal to the
        # height dsm precision X 2 and then determine lower height cut to determine
        # the real robust minimum height of the DSM
        BulldozerLogger.log("Detect low height aberrant values: Starting...", logging.INFO)
        outliers_output = preprocess_histogram_outliers.run( input_dsm_key = input_dsm_key,
                                                             eomanager = eomanager,
                                                             dsm_z_precision = params['dsm_z_precision'] )
        BulldozerLogger.log("Detect low height aberrant values: Done.", logging.INFO)
        

        noisy_mask_key = outliers_output["noisy_mask"]
        dsm_min = outliers_output["robust_min_z"]
        dsm_max = outliers_output["max_z"]
        if params["developer_mode"]:
            noisy_mask_path: str = os.path.join(params["output_dir"], "noisy_mask.tif")
            eomanager.write(key = noisy_mask_key, img_path = noisy_mask_path)
        
        # Step 2
        # Compute the regular area mask
        BulldozerLogger.log("Regular mask computation: Starting...", logging.INFO)
        regular_slope: float = max(float(params["max_ground_slope"]) * eomanager.get_profile(key=input_dsm_key)["transform"][0] / 100.0, params['dsm_z_precision'])
        max_object_size: float = 1.0 / params['min_object_spatial_frequency']
        anchorage_exploration_size = int(max_object_size + 1)
        regular_outputs = preprocess_regular_detector.run(dsm_key= input_dsm_key,
                                                          noisy_key= noisy_mask_key,
                                                          eomanager = eomanager,
                                                          regular_slope = regular_slope,
                                                          anchorage_exploration_size = anchorage_exploration_size)
        BulldozerLogger.log("Regular mask computation: Starting...", logging.INFO)

        outliers_output = preprocess_histogram_outliers.run( input_dsm_key = input_dsm_key,
                                                            input_invalid_key=regular_outputs['regular_mask'],
                                                             eomanager = eomanager,
                                                             dsm_z_precision = params['dsm_z_precision'] )
        BulldozerLogger.log("Noisy mask computation: Done.", logging.INFO)

        regular_mask_key = regular_outputs["regular_mask"]
        if params["developer_mode"]:
            regular_mask_path : str = os.path.join(params["output_dir"], "regular_mask.tif")
            preprocess_anchorage_mask_path: str = os.path.join(params["output_dir"], "preprocess_anchorage_mask.tif")
            eomanager.write(key = regular_mask_key, img_path = regular_mask_path)
        
        # # Step 2.5
        # dsm_profile = eomanager.get_profile(key = input_dsm_key)
        # max_object_size_pixels = int(max_object_size / dsm_profile["transform"][0])
        # anchorage_tile_diameter: int = 2 * max_object_size_pixels
        # nb_anchorage_tiles_x = dsm_profile["width"] // anchorage_tile_diameter
        # nb_anchorage_tiles_y = dsm_profile["height"] // anchorage_tile_diameter
        # offset_x: int = (dsm_profile["width"] % anchorage_tile_diameter) // 2
        # offset_y: int = (dsm_profile["height"] % anchorage_tile_diameter) // 2
        # optimized_tile_size: int = 32 * 20
        # nb_optimized_tiles_x = nb_anchorage_tiles_x // 20
        # nb_optimized_tiles_y = nb_anchorage_tiles_y // 20
        # if nb_anchorage_tiles_x % 20 > 0:
        #     nb_optimized_tiles_x += 1
        # if nb_anchorage_tiles_y % 20 > 0:
        #     nb_optimized_tiles_y += 1

        # tiles = []
        # for ty in range(nb_optimized_tiles_y):
        #     for tx in range(nb_optimized_tiles_x):
        #         sx = tx * 20
        #         sy = ty * 20
        #         ey = min( nb_anchorage_tiles_y - 1,  (ty +1) * 20 - 1)
        #         ex = min( nb_anchorage_tiles_x - 1,  (tx +1) * 20 - 1)
        #         tiles.append( (sx, sy, ex, ey) )

        # anchor_profile = dsm_profile
        # anchor_profile["dtype"] = np.uint8
        # anchor_profile["nodata"] = None
        # preprocess_anchorage_mask_key = eomanager.create_image(anchor_profile)
        # preprocess_anchor_mask = eomanager.get_array(key = preprocess_anchorage_mask_key)

        # with concurrent.futures.ProcessPoolExecutor(max_workers= min(eomanager.nb_workers, len(tiles))) as executor:

        #     futures = { executor.submit(preprocess_anchorage_run,
        #                                 input_dsm_key,
        #                                 regular_mask_key,
        #                                 preprocess_anchorage_mask_key,
        #                                 dsm_min,
        #                                 max_object_size_pixels,
        #                                 tile,
        #                                 eomanager) for tile in tiles }
            
        #     for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=filter_desc):

        #         chunk_buffer, tile = future.result()
        #         stable_x = offset_x + tile[0] * optimized_tile_size
        #         stable_y = offset_y + tile[1] * optimized_tile_size
        #         stable_end_x = stable_x + optimized_tile_size
        #         stable_end_y = stable_y + optimized_tile_size
        #         preprocess_anchor_mask[0,stable_x:stable_end_x,stable_y:stable_end_y] = chunk_buffer[:,:]


        # Step 3
        # Fill the input DSM and compute the uncertainties
        refined_min_z = outliers_output["robust_min_z"]
        BulldozerLogger.log("Filling the DSM : Starting...", logging.INFO)
        fill_outputs = prefill_dsm.run(input_dsm_key = input_dsm_key, 
                                       refined_min_z = refined_min_z, 
                                       eomanager = eomanager)
        BulldozerLogger.log("Filling the DSM and computing the uncertainties: Done", logging.INFO)
        
        filled_dsm_key = fill_outputs["filled_dsm"]

        filled_dsm_path: str = os.path.join(params["output_dir"], "filled_dsm.tif")
        eomanager.write(key =  filled_dsm_key, img_path = filled_dsm_path)

        # Step 4
        # First pass of the drape cloth filter
        BulldozerLogger.log("First pass of a drape cloth filter: Starting...", logging.INFO)
        #TODO handle Land use map (convert it to reach: ground=1/else=0)
        preprocess_anchorage_mask_key = eomanager.create_image(eomanager.get_profile(regular_mask_key))
        dtm_key = dtm_extraction.drape_cloth(filled_dsm_key = filled_dsm_key,
                                            predicted_anchorage_mask_key=preprocess_anchorage_mask_key,
                                            eomanager = eomanager,
                                            max_object_size = max_object_size,
                                            dsm_min_z = refined_min_z,
                                            dsm_max_z = dsm_max,
                                            prevent_unhook_iter = params["prevent_unhook_iter"],
                                            spring_tension = params["cloth_tension_force"],
                                            num_outer_iterations = params["num_outer_iter"],
                                            num_inner_iterations = params["num_inner_iter"])
        BulldozerLogger.log("First pass of a drape cloth filter: Done.", logging.INFO)
        
        if params["developer_mode"]:
            inter_dtm_path: str = os.path.join(params["output_dir"], "dtm_first_pass.tif")
            eomanager.write(key = dtm_key, img_path = inter_dtm_path)
        

        # #TODO for steps 5 and 6: add a conditional statement to activate the second pass

        # Step 5
        # Attempt to detect terrain pixels
        # Brute force post process to minimize a side effet of the drap that often underestimates the terrain height
        # All regular pixels where the diff Z is lower or equal than dtm_max_error meters will be labeled as possible terrain points.
        # Knowing that the drape cloth will be run again.
        if params["dtm_max_error"] is None:
            params["dtm_max_error"] = 2.0 * params['dsm_z_precision']
        BulldozerLogger.log("Post detection of Terrain pixels: Starting...", logging.INFO)
        post_anchorage_output = postprocess_anchorage.run(intermediate_dtm_key=dtm_key, 
                                                          dsm_key=filled_dsm_key, 
                                                          regular_mask_key=regular_mask_key,
                                                          error_threshold=params["dtm_max_error"], 
                                                          eomanager=eomanager)
        BulldozerLogger.log("Post detection of Terrain pixels: Done.", logging.INFO)
        
        post_anchorage_mask_key = post_anchorage_output["post_process_anchorage"]

        if params["developer_mode"]:
            output_post_anchorage_path: str = os.path.join(params["output_dir"], "post_anchorage_mask.tif")
            eomanager.write(key = post_anchorage_mask_key, img_path = output_post_anchorage_path)
        
        eomanager.release(key = dtm_key)

        # Step 6
        # Compute final DTM with post processed predicted terrain point
        BulldozerLogger.log("Second pass of a drape cloth filter: Starting...", logging.INFO)
        dtm_key = dtm_extraction.drape_cloth(filled_dsm_key = filled_dsm_key,
                                             predicted_anchorage_mask_key=post_anchorage_mask_key,
                                             eomanager = eomanager,
                                             max_object_size = max_object_size,
                                             dsm_min_z = refined_min_z,
                                             dsm_max_z = dsm_max,
                                             prevent_unhook_iter = params["prevent_unhook_iter"],
                                             spring_tension = params["cloth_tension_force"],
                                             num_outer_iterations = params["num_outer_iter"],
                                             num_inner_iterations = params["num_inner_iter"])
        BulldozerLogger.log("Second pass of a drape cloth filter: Done.", logging.INFO)

        eomanager.write(key = dtm_key, img_path = os.path.join(params["output_dir"], "dtm_second_pass.tif"))


        # Step 7: reverse drape cloth
        # Needs: dtm_key and post_anchors to snap and dsm_max_z for int
        BulldozerLogger.log("Reverse pass of a drape cloth filter: Starting...", logging.INFO)
        reverse_dtm_key = dtm_extraction.reverse_drape_cloth(filled_dsm_key = filled_dsm_key,
                                             first_pass_dtm_key = dtm_key,
                                             pre_anchorage_mask_key = preprocess_anchorage_mask_key,
                                             post_anchorage_mask_key=post_anchorage_mask_key,
                                             eomanager = eomanager,
                                             max_object_size = max_object_size,
                                             dsm_min_z = refined_min_z,
                                             dsm_max_z = dsm_max,
                                             prevent_unhook_iter = params["prevent_unhook_iter"],
                                             spring_tension = params["cloth_tension_force"],
                                             num_outer_iterations = params["num_outer_iter"],
                                             num_inner_iterations = params["num_inner_iter"])
        
        if params["developer_mode"]:
            eomanager.write(key = reverse_dtm_key, img_path = os.path.join(params["output_dir"], "reverse_dtm.tif"))
        
        final_dtm = eomanager.get_array(key = dtm_key)
        reverse_dtm = eomanager.get_array(key = reverse_dtm_key)
        final_dtm[0,:, :] +=  reverse_dtm[0,:,:]
        final_dtm[0,:, :] /= 2.0
        eomanager.release(key = reverse_dtm_key)
        eomanager.write(key = dtm_key, img_path = os.path.join(params["output_dir"], "final_dtm.tif"))


        # Si more steps dont forget (if reverse not activated) to release the DSM.

        
        # dtm_path: str = os.path.join(params["output_dir"], "dtm.tif")
        # eomanager.write(key = dtm_key, img_path = dtm_path)

        # if params["generate_dhm"]:
        #     BulldozerLogger.log("Generating DHM: Starting...", logging.INFO)
        #     dsm = eomanager.get_array(key = filled_dsm_key)[0,:,:]
        #     dtm = eomanager.get_array(key = dtm_key)[0,:,:]
        #     dhm = dsm - dtm
        #     with rasterio.open(os.path.join(params["output_dir"], "dhm.tif"), "w", **eomanager.get_profile(key = filled_dsm_key)) as dhm_out:
        #         dhm_out.write(dhm, 1)
        #     BulldozerLogger.log("Generating DHM: Done.", logging.INFO)

        # # And finally we are done ! It is exhausting to extract a DTM dont you think ?
        # BulldozerLogger.log("And finally we are done ! It is exhausting to extract a DTM don't you think ?", logging.INFO)



def retrieve_params(config_path : str = None, **kwargs):
    """
        Defines the input parameters based on the provided configuration file (if provided), or the kwargs (CLI or Python API).
        For the missing parameters the Bullodzer default values are set.
        
        Args:
            config_path: path to the config file (YAML file expected, refers to the provided template in /conf).
            **kwargs: list of expected arguments if the urser don't provide a configuration file:
                - dsm_path: str (required)
                - output_dir: str (required)
                - nb_max_workers: int (optionnal, 8 by default)
                - dsm_z_precision: float (optional, 1.0 by default)
                - fill_search_radius: int (optional, 100 by default)
                - max_ground_slope: float (optional, 20.0 % by default)
                - min_object_spatial_frequency: float (optional, 0.0625 by default)
                - dtm_max_error: float (optional, 2.0 meters by default)
                - cloth_tension_force: int (optionnal, 3 by default)
                - prevent_unhook_iter: int (optionnal, 10 by default)
                - num_outer_iter: int (optionnal, 100 by default)
                - num_inner_iter: int (optionnal, 10 by default)
                - output_resolution: float (optionnal, null by default)
                - generate_dhm: bool (optionnal, True by default)
                - developper_mode : bool (optionnal, False by default)
                refers to the documentation to understand the use of each parameter.
    """
    bulldozer_params = dict()
    # Config path provided case

    input_params = dict()

    if config_path:
        # Configuration file format check
        if not (config_path.endswith('.yaml') or config_path.endswith('.yml')) :
            raise ValueError('Expected yaml configuration file: \'config_path\' argument should be a path to a Yaml file (here: {})'.format(config_path))

        # Configuration file existence check
        if not os.path.isfile(config_path):
            raise FileNotFoundError('The input configuration file \'{}\' doesn\'t exist'.format(config_path))
        
        # Retrieves all the settings
        parser = ConfigParser(False)
        input_params = parser.read(config_path)
        if 'dsm_path' not in input_params.keys():
            raise ValueError('No DSM path provided or invalid YAML key syntax. Expected: dsm_path="<path>/<dsm_file>.<[tif/tiff]>"')
        else:
            bulldozer_params['dsm_path'] = input_params['dsm_path']

        if 'output_dir' not in input_params.keys():
            raise ValueError('No output diectory path provided or invalid YAML key syntax. Expected: output_dir="<path>")')
        else:
            bulldozer_params['output_dir'] = input_params['output_dir']
    else :
        # User directly provides the input parameters (kwargs)
        input_params = kwargs
        if not 'dsm_path' in input_params:
            raise ValueError('No DSM path provided or invalid argument syntax. Expected: \n\t-Python API: dsm_to_dtm(dsm_path="<path>")\n\t-CLI: bulldozer -in <path>')
        bulldozer_params['dsm_path'] = input_params['dsm_path']
        if not 'output_dir' in input_params:
            raise ValueError('No output diectory path provided or invalid argument syntax. Expected: \n\t-Python API:  dsm_to_dtm(dsm_path="<path>, output_dir="<path>")\n\t-CLI: bulldozer -in <path> -out path')
        bulldozer_params['output_dir'] = input_params['output_dir']

    # For each optional parameters of Bulldozer check if the user provide a specific value, otherwise retrieve the default value from DefaultValues enum
    for key, value in DefaultValues.items():
        bulldozer_params[key.lower()] = input_params[key.lower()] if key.lower() in input_params.keys() else value
    
    # Retrieves the number of CPU if the number of available workers if the user didn't provides a specific value
    if bulldozer_params['nb_max_workers'] is None:
        bulldozer_params['nb_max_workers'] = multiprocessing.cpu_count()
    
    return bulldozer_params
    

def get_parser():
    """
    Argument parser for Bulldozer (CLI).

    Returns:
        the parser.
    """
    parser = argparse.ArgumentParser(description=("Bulldozer"))

    parser.add_argument(
        "--conf", required=True, type=str, help="Bulldozer config file"
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    
    #TODO add all the parameters
    return parser


def bulldozer_cli() -> None:
    """
        Call bulldozer main pipeline.
    """
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # The first parameter should be the path to the configuration file
    config_path = args.conf

    dsm_to_dtm(config_path)

if __name__ == "__main__":
    bulldozer_cli()