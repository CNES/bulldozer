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

"""
    This module is used to postprocess the DTM in order to improve its quality. It required a DTM generated from Bulldozer.
"""
import os
from typing import List
import logging
from shutil import copy
from datetime import datetime
import argparse
import multiprocessing
import argcomplete
from copy import copy
import numpy as np
import rasterio
from bulldozer.utils.config_parser import ConfigParser
from bulldozer.utils.bulldozer_logger import BulldozerLogger, Runtime
from bulldozer.pipeline.bulldozer_parameters import bulldozer_pipeline_params
from bulldozer._version import __version__

import bulldozer.eoscale.manager as eom

# Preprocessing steps of Bulldozer
import bulldozer.preprocessing.regular_detection.regular_detector as preprocess_regular_detector
import bulldozer.preprocessing.border_detection.border_detector as preprocess_border_detector
import bulldozer.preprocessing.fill.prefill_dsm as prefill_dsm
import bulldozer.preprocessing.fill.fill_dsm as fill_dsm

# Drape cloth filter
import bulldozer.extraction.drape_cloth as dtm_extraction

# Postprocessing steps of Bulldozer
import bulldozer.postprocessing.ground_detection.post_anchorage_detection as postprocess_anchorage
import bulldozer.postprocessing.fill_pits as fill_pits

@Runtime
def dsm_to_dtm(config_path: str = None, **kwargs: int) -> None:
    """
        Main pipeline orchestrator.
        
        Args:
            config_path: path to the config file (YAML file expected, refers to the provided template in /conf).
            **kwargs: bulldozer parameters (used if the user don't provide a configuration file).

    """
    # Retrieves Bulldozer settings from the config file, the CLI parameters or the Python API parameters
    params = retrieve_params(config_path, **kwargs)

    # If the target output directory does not exist, creates it
    if not os.path.isdir(params["output_dir"]):
        os.makedirs(params["output_dir"]) 

    # If the target output directory does not exist, creates it
    output_masks_dir = os.path.join(params["output_dir"], "masks")
    if not os.path.isdir(output_masks_dir):
        os.makedirs(output_masks_dir)

    # In the developer mode, if the developer directory does not exist, creates it
    if params["developer_mode"]:
        developer_dir = os.path.join(params["output_dir"], "developer")
        if not os.path.isdir(developer_dir):
            os.makedirs(developer_dir)

    logger = BulldozerLogger.getInstance(logger_file_path=os.path.join(params["output_dir"], "bulldozer_" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + ".log"))

    BulldozerLogger.log("Bulldozer input parameters: \n" + "".join("\t- " + str(key) + ": " + str(value) + "\n" for key, value in params.items()), logging.DEBUG)

    # Warns the user that he/she provides parameters that are not used
    if "ignored_params" in params.keys():
        BulldozerLogger.log("The following input parameters are ignored: {}. \nPlease refer to the documentation for the list of valid parameters.".format(", ".join(params["ignored_params"])), logging.WARNING)

    with eom.EOContextManager(nb_workers=params["nb_max_workers"], tile_mode=True) as eomanager:

        # Open the input dsm that might be noisy and full of nodatas...
        input_dsm_key = eomanager.open_raster(raster_path=params["dsm_path"])

        # Step 1: Compute the regular area mask 
        # Take the maximum slope between the slope provided by the user (converted in meter) and the slope derived from the altimetric dsm precision 
        regular_slope: float = max(float(params["max_ground_slope"]) * eomanager.get_profile(key=input_dsm_key)["transform"][0] / 100.0, params["dsm_z_precision"])
        regular_outputs = preprocess_regular_detector.detect_regular_areas(dsm_key=input_dsm_key,
                                                                           eomanager=eomanager,
                                                                           regular_slope=regular_slope)
        regular_mask_key = regular_outputs["regular_mask"]

        if params["developer_mode"]:
            regular_mask_path: str = os.path.join(developer_dir, "regular_mask.tif")
            eomanager.write(key=regular_mask_key, img_path=regular_mask_path, binary=True)

        # Step 2: Detect inner and border nodata masks
        #TODO: check if user provide specific noadata value otherwise retrieve the nodata value from DSM profile. If None/null replace it by default value -32768
        nodata_value = eomanager.get_profile(key=input_dsm_key)["nodata"]

        inner_outer_result = preprocess_border_detector.detect_border_nodata(dsm_key=input_dsm_key,
                                        eomanager=eomanager,
                                        nodata=nodata_value)
        
        inner_no_data_mask_key = inner_outer_result["inner_no_data_mask"]
        border_no_data_mask_key = inner_outer_result["border_no_data_mask"]

        border_no_data_mask_path: str = os.path.join(output_masks_dir, "border_no_data.tif")
        eomanager.write(key=border_no_data_mask_key, img_path=border_no_data_mask_path, binary=True)

        inner_no_data_mask_path: str = os.path.join(output_masks_dir, "inner_no_data.tif")
        eomanager.write(key=inner_no_data_mask_key, img_path=inner_no_data_mask_path, binary=True)

        if not params["generate_dhm"]:
            # Release the memory of inner nodata mask if the DHM is not generated
            eomanager.release(key=inner_no_data_mask_key)

        # Step 3: Fill the input DSM and compute the uncertainties
        BulldozerLogger.log("Filling the DSM : Starting...", logging.INFO)
        fill_outputs = fill_dsm.run(dsm_key=input_dsm_key,
                                    mask_key=regular_mask_key,
                                    border_no_data_key=border_no_data_mask_key,
                                    eomanager=eomanager)
        BulldozerLogger.log("Filling the DSM and computing the uncertainties: Done", logging.INFO)

        filled_dsm_key = fill_outputs["filled_dsm"]

        if params["developer_mode"]:
            filled_dsm_path: str = os.path.join(params["output_dir"], "filled_dsm.tif")
            eomanager.write(key=filled_dsm_key, img_path=filled_dsm_path)

        BulldozerLogger.log("End fill dsm", logging.INFO)

        # Step 4.5 - optional: post anchor mask computation (first drape cloth + terrain pixel detection)
        # Brute force post process to minimize a side effect of the drape that often underestimates the terrain height
        # All regular pixels where the diff Z is lower or equal than dtm_max_error meters will be labeled as possible terrain points.
        # Knowing that the drape cloth will be run again.
        #TODO if cos:
        #    cos_mask_key = data)
        #else:
        #    create image profile à 0
        if params["post_anchor_points_activation"]:
            BulldozerLogger.log("First pass of a drape cloth filter: Starting...", logging.INFO)
            # TODO handle Land use map (convert it to reach: ground=1/else=0)
            cos_mask_key = eomanager.create_image(eomanager.get_profile(regular_mask_key))
            dtm_key = dtm_extraction.drape_cloth(filled_dsm_key=filled_dsm_key,
                                                 predicted_anchorage_mask_key=cos_mask_key,
                                                 eomanager=eomanager,
                                                 max_object_size=params["max_object_size"],
                                                 prevent_unhook_iter=params["prevent_unhook_iter"],
                                                 spring_tension=params["cloth_tension_force"],
                                                 num_outer_iterations=params["num_outer_iter"],
                                                 num_inner_iterations=params["num_inner_iter"])
            BulldozerLogger.log("First pass of a drape cloth filter: Done.", logging.INFO)
            eomanager.release(key=cos_mask_key)

            if params["developer_mode"]:
                inter_dtm_path: str = os.path.join(params["output_dir"], "dtm_first_pass.tif")
                eomanager.write(key=dtm_key, img_path=inter_dtm_path)

            # TODO for steps 4.5: add a conditional statement to activate the second pass
            # Attempt to detect terrain pixels
            #TODO remplacer max error par dsm_z_precision
            if params["dtm_max_error"] is None:
                params["dtm_max_error"] = 2.0 * params["dsm_z_precision"]
            BulldozerLogger.log("Post detection of Terrain pixels: Starting...", logging.INFO)
            post_anchorage_output = postprocess_anchorage.run(intermediate_dtm_key=dtm_key,
                                                              dsm_key=filled_dsm_key,
                                                              regular_mask_key=regular_mask_key,
                                                              error_threshold=params["dtm_max_error"],
                                                              eomanager=eomanager)
            BulldozerLogger.log("Post detection of Terrain pixels: Done.", logging.INFO)

            post_anchorage_mask_key = post_anchorage_output["post_process_anchorage"]
            eomanager.release(key=dtm_key)

            if params["developer_mode"]:
                output_post_anchorage_path: str = os.path.join(params["output_dir"], "post_anchorage_mask.tif")
                eomanager.write(key=post_anchorage_mask_key, img_path=output_post_anchorage_path)
        else:
            post_anchorage_mask_key = eomanager.create_image(eomanager.get_profile(regular_mask_key))

        eomanager.release(key=regular_mask_key)


        #TODO Copier le même foncionnement pour le COS et anchor points
        # if params["pre_anchor_points_activation"]:
        #     # Union of post_anchorage_mask with pre_process_anchorage_mask
        #     post_anchors = eomanager.get_array(key=post_anchorage_mask_key)
        #     pre_anchors = eomanager.get_array(key=preprocess_anchorage_mask_key)
        #     np.logical_or(pre_anchors[0, :, :], post_anchors[0, :, :], out=post_anchors[0, :, :])
        #     eomanager.release(key=preprocess_anchorage_mask_key)


        # Step 5: Compute final DTM with post processed predicted terrain point
        BulldozerLogger.log("Main pass of a drape cloth filter: Starting...", logging.INFO)
        dtm_key = dtm_extraction.drape_cloth(filled_dsm_key=filled_dsm_key,
                                             predicted_anchorage_mask_key=post_anchorage_mask_key,
                                             eomanager=eomanager,
                                             max_object_size=params["max_object_size"],
                                             prevent_unhook_iter=params["prevent_unhook_iter"],
                                             spring_tension=params["cloth_tension_force"],
                                             num_outer_iterations=params["num_outer_iter"],
                                             num_inner_iterations=params["num_inner_iter"])
        BulldozerLogger.log("Main pass of a drape cloth filter: Done.", logging.INFO)
        eomanager.release(key=post_anchorage_mask_key)

        if params["developer_mode"]:
            eomanager.write(key=dtm_key, img_path=os.path.join(params["output_dir"], "dtm_second_pass.tif"))

        # Step 8: remove pits
        BulldozerLogger.log("Pits removal: Starting.", logging.INFO)
        dtm_key, pits_mask_key = fill_pits.run(dtm_key, border_no_data_mask_key, eomanager)
        if params["developer_mode"]:
            eomanager.write(key=pits_mask_key, img_path=os.path.join(params["output_dir"], "fill_pits_mask.tif"))
        BulldozerLogger.log("Pits removal: Done.", logging.INFO)
        eomanager.release(key=pits_mask_key)


        # last step: Apply border_no_data_mask
        BulldozerLogger.log("Applying border no data: Starting...", logging.INFO)
        final_dtm = eomanager.get_array(key=dtm_key)[0]
        border_no_data_mask = eomanager.get_array(key=border_no_data_mask_key)[0]
        final_dtm[border_no_data_mask==1] = eomanager.get_profile(key=dtm_key)["nodata"]
        BulldozerLogger.log("Applying border no data: Done...", logging.INFO)

        # Write final outputs
        # optional - write final dhm
        if params["generate_dhm"]:
            BulldozerLogger.log("Generating DHM: Starting...", logging.INFO)
            dsm = eomanager.get_array(key=filled_dsm_key)[0, :, :]
            dtm = eomanager.get_array(key=dtm_key)[0, :, :]
            dhm = dsm - dtm
            BulldozerLogger.log("Applying border no data to DHM: Starting...", logging.INFO)
            dhm[border_no_data_mask==1] = eomanager.get_profile(key=dtm_key)["nodata"]
            eomanager.release(key=border_no_data_mask_key)
            inner_no_data_mask = eomanager.get_array(key=inner_no_data_mask_key)[0]
            dhm[inner_no_data_mask==1] = eomanager.get_profile(key=dtm_key)["nodata"]
            eomanager.release(key=inner_no_data_mask_key)
            BulldozerLogger.log("Applying border no data to DHM: Done...", logging.INFO)
            with rasterio.open(os.path.join(params["output_dir"], "dhm.tif"), "w", **eomanager.get_profile(key=filled_dsm_key)) as dhm_out:
                dhm_out.write(dhm, 1)
            BulldozerLogger.log("Generating DHM: Done.", logging.INFO)
        else:
            # if the DHM is not generated, release the border_nodata_mask memory
            eomanager.release(key=border_no_data_mask_key)
        eomanager.release(key=filled_dsm_key)

        # write final dtm
        eomanager.write(key=dtm_key, img_path=os.path.join(params["output_dir"], "final_dtm.tif"))
        eomanager.release(key=input_dsm_key)
        eomanager.release(key=dtm_key)
        


        # And finally we are done ! It is exhausting to extract a DTM dont you think ?
        BulldozerLogger.log("And finally we are done ! It is exhausting to extract a DTM don't you think ?", logging.INFO)


def retrieve_params(config_path: str = None, **kwargs: int) -> dict:
    """
        Defines the input parameters based on the provided configuration file (if provided), or the kwargs (CLI or Python API).
        For the missing parameters the Bulldozer default values are used.
        
        Args:
            config_path: path to the config file (YAML file expected, refers to the provided template in /conf).
            **kwargs: list of expected arguments if the user doesn't provide a configuration file. Refers to the documentation to get the full parameter list.
            
        Returns:
            the dict containing the input parameters.
    """
    bulldozer_params = dict()
    # Config path provided case

    input_params = dict()

    if config_path:
        # Configuration file format check
        if not (config_path.endswith(".yaml") or config_path.endswith(".yml")):
            raise ValueError("Expected yaml configuration file: \"config_path\" argument should be a path to a Yaml file (here: {})".format(config_path))

        # Configuration file existence check
        if not os.path.isfile(config_path):
            raise FileNotFoundError("The input configuration file \"{}\" doesn't exist".format(config_path))
        
        # Retrieves all the settings
        parser = ConfigParser(False)
        input_params = parser.read(config_path)
        if "dsm_path" not in input_params.keys() or input_params["dsm_path"] is None:
            raise ValueError("No DSM path provided or invalid YAML key syntax. Expected: dsm_path=\"<path>/<dsm_file>.<[tif/tiff]>\"")
        bulldozer_params["dsm_path"] = input_params["dsm_path"]

        if "output_dir" not in input_params.keys() or input_params["output_dir"] is None:
            raise ValueError("No output directory path provided or invalid YAML key syntax. Expected: output_dir=\"<path>\")")
        bulldozer_params["output_dir"] = input_params["output_dir"]
    else :
        # User directly provides the input parameters (kwargs)
        input_params = kwargs
        if not "dsm_path" in input_params or input_params["dsm_path"] is None:
            raise ValueError("No DSM path provided or invalid argument syntax. Expected: \n\t-Python API: dsm_to_dtm(dsm_path=\"<path>\")\n\t-CLI: bulldozer -dsm <path>")
        bulldozer_params["dsm_path"] = input_params["dsm_path"]
        if not "output_dir" in input_params or input_params["output_dir"] is None:
            raise ValueError("No output directory path provided or invalid argument syntax. Expected: \n\t-Python API:  dsm_to_dtm(dsm_path=\"<path>\", output_dir=\"<path>\")\n\t-CLI: bulldozer -dsm <path> -out <path>")
        bulldozer_params["output_dir"] = input_params["output_dir"]
    
    # For each optional parameters of Bulldozer check if the user provide a specific value, otherwise retrieve the default value from bulldozer_pipeline_params
    for group_name, list_params in bulldozer_pipeline_params.items():
        if "SETTINGS" in group_name:
            for param in list_params:
                bulldozer_params[param.name] = input_params[param.name] if param.name in input_params.keys() else param.default_value
 
    # Retrieves the number of CPU if the number of available workers if the user didn't provide a specific value
    if bulldozer_params["nb_max_workers"] is None:
        bulldozer_params["nb_max_workers"] = multiprocessing.cpu_count()
    
    # Retrieves ignored provided parameters (parameters not used by bulldozer)
    ignored_params = set(input_params.keys()).difference(set(bulldozer_pipeline_params[group][param].name for group in bulldozer_pipeline_params.keys() for param in range(len(bulldozer_pipeline_params[group]))))
    if len(ignored_params) > 0:
        bulldozer_params["ignored_params"] = ignored_params

    return bulldozer_params
    

def get_parser():
    """
    Argument parser for Bulldozer (CLI).

    Returns:
        the parser.
    """
    parser = argparse.ArgumentParser(description="Bulldozer: CNES pipeline designed to extract DTM from DSM")
    
    parser.add_argument("config_path", type=str, nargs="?", help="Input configuration file")
    parser.add_argument("-v", "--version", action="version", 
                        version="%(prog)s {version}".format(version=__version__))
    
    for group_name, list_params in bulldozer_pipeline_params.items():
        group = parser.add_argument_group(description=f"*** {group_name} ***")
        for param in list_params:
            if param.param_type == bool:
                param_action = "store_true" if param.default_value is False else "store_false"
                group.add_argument(f"-{param.alias}", f"--{param.name}", action=param_action, help=param.description)
            else:
                group.add_argument(f"-{param.alias}", f"--{param.name}", type=param.param_type, metavar="VALUE",
                                   default=param.default_value, action="store", help=param.description)

    return parser


def bulldozer_cli() -> None:
    """
        Call bulldozer main pipeline.
    """
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    dsm_to_dtm(**vars(args))


if __name__ == "__main__":
    bulldozer_cli()