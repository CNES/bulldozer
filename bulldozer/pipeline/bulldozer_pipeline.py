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
import logging
from datetime import datetime
import multiprocessing
import numpy as np
import rasterio
import sys
import argparse
import argcomplete
# Building arguments parser
from bulldozer.utils.bulldozer_logger import BulldozerLogger, Runtime
from bulldozer.utils.config_parser import ConfigParser
from bulldozer.utils.bulldozer_argparse import BulldozerArgumentParser, REQ_PARAM_KEY, OPT_PARAM_KEY, EXPERT_PARAM_KEY
from bulldozer.pipeline.bulldozer_parameters import bulldozer_pipeline_params, DEFAULT_NODATA
from bulldozer._version import __version__

import bulldozer.eoscale.manager as eom

# Preprocessing steps of Bulldozer
import bulldozer.preprocessing.regular_detection.regular_detector as preprocess_regular_detector
import bulldozer.preprocessing.border_detection.border_detector as preprocess_border_detector
import bulldozer.preprocessing.dsm_filling.dsm_filler as preprocess_dsm_filler
import bulldozer.preprocessing.ground_detection.ground_anchors_detector as ground_anchors_detector
# Drape cloth filter
import bulldozer.extraction.drape_cloth as dtm_extraction

# Postprocessing steps of Bulldozer
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
    else:
        developer_dir=''

    logger = BulldozerLogger.getInstance(logger_file_path=os.path.join(params["output_dir"], "bulldozer_" + datetime.now().strftime("%Y-%m-%dT%Hh%Mm%S") + ".log"))

    BulldozerLogger.log("Bulldozer input parameters: \n" + "".join("\t- " + str(key) + ": " + str(value) + "\n" for key, value in params.items()), logging.DEBUG)

    with rasterio.open(params["dsm_path"]) as ds:
        BulldozerLogger.log("Input DSM profile: \n" + "".join("\t- " + str(key) + ": " + str(value) + "\n" for key, value in ds.profile.items()), logging.DEBUG)

    # Warns the user that he/she provides parameters that are not used
    if "ignored_params" in params.keys():
        BulldozerLogger.log("The following input parameters are ignored: {}. \nPlease refer to the documentation for the list of valid parameters.".format(", ".join(params["ignored_params"])), logging.WARNING)

    # Retrieves the number of CPU if the number of available workers if the user didn't provide a specific value
    if params["nb_max_workers"] is None:
        params["nb_max_workers"] = multiprocessing.cpu_count()
        BulldozerLogger.log("\"nb_max_workers\" parameter is not set. The default value is used: maximum number of available CPU core ({}).".format(params["nb_max_workers"]), logging.DEBUG)

    with eom.EOContextManager(nb_workers=params["nb_max_workers"], tile_mode=True) as eomanager:

        # Open the input dsm that might be noisy and full of nodatas...
        input_dsm_key = eomanager.open_raster(raster_path=params["dsm_path"])

        # Nodata value handling
        input_nodata = eomanager.get_profile(key=input_dsm_key)["nodata"]
        if input_nodata is None:
            BulldozerLogger.log("The provided nodata value is None. Bulldozer will use it's own nodata default value ({}) during the pipeline run (Cython constraint).".format(DEFAULT_NODATA), logging.DEBUG)
            dsm = eomanager.get_array(key=input_dsm_key)[0]
            dsm[dsm == None] = DEFAULT_NODATA
            dsm = np.nan_to_num(dsm, copy=False, nan=DEFAULT_NODATA)
            pipeline_nodata = DEFAULT_NODATA
        elif np.isnan(input_nodata):
            BulldozerLogger.log("The provided nodata value is NaN. Bulldozer will use it's own nodata default value ({}) during the pipeline run (Cython constraint).".format(DEFAULT_NODATA), logging.DEBUG)
            dsm = eomanager.get_array(key=input_dsm_key)[0]
            dsm = np.nan_to_num(dsm, copy=False, nan=DEFAULT_NODATA)
            pipeline_nodata = DEFAULT_NODATA
        else:
            pipeline_nodata = input_nodata
            BulldozerLogger.log("Nodata retrieved and used in the pipeline: {}".format(pipeline_nodata), logging.DEBUG)    

        # If the user doesn't provide an DSM altimetric accuracy, set it to default value: 2*planimetric resolution 
        if params["dsm_z_accuracy"] is None:
            params["dsm_z_accuracy"] =  2*eomanager.get_profile(key=input_dsm_key)["transform"][0]
            BulldozerLogger.log("\"dsm_z_accuracy\" parameter is null, used default value: 2*planimetric resolution ({}m).".format(params["dsm_z_accuracy"]), logging.DEBUG)

        # Step 1: Compute the regular area mask 
        # Take the maximum slope between the slope provided by the user (converted in meter) and the slope derived from the altimetric dsm accuracy 
        regular_slope: float = max(float(params["max_ground_slope"]) * eomanager.get_profile(key=input_dsm_key)["transform"][0] / 100.0, params["dsm_z_accuracy"])
        regular_outputs = preprocess_regular_detector.detect_regular_areas(dsm_key=input_dsm_key,
                                                                           regular_slope=regular_slope,
                                                                           nodata=pipeline_nodata,
                                                                           max_object_size=params["max_object_size"],
                                                                           reg_filtering_iter=params["reg_filtering_iter"],
                                                                           eomanager=eomanager,
                                                                           dev_mode=params["developer_mode"],
                                                                           dev_dir=developer_dir)
        regular_mask_key = regular_outputs["regular_mask_key"]

        if params["developer_mode"]:
            regular_mask_path: str = os.path.join(developer_dir, "regular_mask.tif")
            eomanager.write(key=regular_mask_key, img_path=regular_mask_path, binary=True)

        # Step 2: Detect inner and border nodata masks
        inner_outer_result = preprocess_border_detector.detect_border_nodata(dsm_key=input_dsm_key,
                                                                             eomanager=eomanager,
                                                                             nodata=pipeline_nodata)
        
        inner_nodata_mask_key = inner_outer_result["inner_nodata_mask"]
        border_nodata_mask_key = inner_outer_result["border_nodata_mask"]

        border_nodata_mask_path: str = os.path.join(output_masks_dir, "border_nodata.tif")
        eomanager.write(key=border_nodata_mask_key, img_path=border_nodata_mask_path, binary=True)

        inner_nodata_mask_path: str = os.path.join(output_masks_dir, "inner_nodata.tif")
        eomanager.write(key=inner_nodata_mask_key, img_path=inner_nodata_mask_path, binary=True)

        if not params["generate_dhm"]:
            # Release the memory of inner nodata mask if the DHM is not generated
            eomanager.release(key=inner_nodata_mask_key)

        # Step 3: Fill the input DSM and compute the uncertainties
        fill_outputs = preprocess_dsm_filler.fill_dsm(dsm_key=input_dsm_key,
                                                      regular_key=regular_mask_key,
                                                      border_nodata_key=border_nodata_mask_key,
                                                      nodata=pipeline_nodata,
                                                      max_object_size=params["max_object_size"],
                                                      eomanager=eomanager,
                                                      dev_mode=params["developer_mode"],
                                                      dev_dir=developer_dir)

        filled_dsm_key = fill_outputs["filled_dsm"]

        if params["developer_mode"]:
            filled_dsm_path: str = os.path.join(developer_dir, "filled_dsm.tif")
            eomanager.write(key=filled_dsm_key, img_path=filled_dsm_path)


        if params["ground_mask_path"]:
            ground_mask_key = eomanager.open_raster(params["ground_mask_path"])
        else:
            ground_mask_key = eomanager.create_image(eomanager.get_profile(regular_mask_key))

        # Step 4 [optional]: post anchor mask computation (first drape cloth + terrain pixel detection)
        # Run a first drape cloth simulation to minimize the underestimation the terrain height (common issue)
        # All regular pixels where the diff Z is lower or equal than dtm_max_error meters will be labeled as possible terrain points.
        # Knowing that the drape cloth will be run again.
        if params["activate_ground_anchors"]:
            BulldozerLogger.log("First pass of a drape cloth filter: Starting...", logging.INFO)
            dtm_key = dtm_extraction.drape_cloth(filled_dsm_key=filled_dsm_key,
                                                 ground_mask_key=ground_mask_key,
                                                 eomanager=eomanager,
                                                 max_object_size=params["max_object_size"],
                                                 prevent_unhook_iter=params["prevent_unhook_iter"],
                                                 num_outer_iterations=params["num_outer_iter"],
                                                 num_inner_iterations=params["num_inner_iter"],
                                                 nodata=pipeline_nodata)
            BulldozerLogger.log("First pass of a drape cloth filter: Done.", logging.INFO)

            if params["developer_mode"]:
                inter_dtm_path: str = os.path.join(developer_dir, "dtm_first_pass.tif")
                eomanager.write(key=dtm_key, img_path=inter_dtm_path)

            ground_anchors_output = ground_anchors_detector.detect_ground_anchors(intermediate_dtm_key=dtm_key,
                                                                                  dsm_key=filled_dsm_key,
                                                                                  regular_mask_key=regular_mask_key,
                                                                                  dsm_z_accuracy=params["dsm_z_accuracy"],
                                                                                  eomanager=eomanager)
            ground_anchors_mask_key = ground_anchors_output["ground_anchors_mask_key"]
            eomanager.release(key=dtm_key)

            if params["developer_mode"]:
                ground_anchors_mask_path: str = os.path.join(developer_dir, "ground_anchors_mask.tif")
                eomanager.write(key=ground_anchors_mask_key, img_path=ground_anchors_mask_path)
        else:
            ground_anchors_mask_key = eomanager.create_image(eomanager.get_profile(regular_mask_key))

        eomanager.release(key=regular_mask_key)

        # Step 5 [optional]: ground mask
        if params["ground_mask_path"]:
            # Union of detected ground anchors (ground_anchors_mask) with provided ground_mask
            ground_anchors_mask = eomanager.get_array(key=ground_anchors_mask_key)
            ground_mask = eomanager.get_array(key=ground_mask_key)
            np.logical_or(ground_anchors_mask[0, :, :], ground_mask[0, :, :], out=ground_anchors_mask[0, :, :])
            if params["developer_mode"]:
                anchorage_mask_with_ground_path: str = os.path.join(developer_dir, "anchorage_mask_with_ground.tif")
                eomanager.write(key=ground_anchors_mask_key, img_path=anchorage_mask_with_ground_path, binary=True)
            BulldozerLogger.log("Ground mask processing: Done.", logging.INFO)

        eomanager.release(key=ground_mask_key)

        # Step 6: Compute final DTM with post processed predicted terrain point
        BulldozerLogger.log("Main pass of a drape cloth filter: Starting...", logging.INFO)
        dtm_key = dtm_extraction.drape_cloth(filled_dsm_key=filled_dsm_key,
                                             ground_mask_key=ground_anchors_mask_key,
                                             eomanager=eomanager,
                                             max_object_size=params["max_object_size"],
                                             prevent_unhook_iter=params["prevent_unhook_iter"],
                                             num_outer_iterations=params["num_outer_iter"],
                                             num_inner_iterations=params["num_inner_iter"],
                                             nodata=pipeline_nodata)
        BulldozerLogger.log("Main pass of a drape cloth filter: Done.", logging.INFO)
        eomanager.release(key=ground_anchors_mask_key)

        if params["developer_mode"]:
            eomanager.write(key=dtm_key, img_path=os.path.join(developer_dir, "dtm_second_pass.tif"))

        # Step 7: remove pits
        BulldozerLogger.log("Pits removal: Starting.", logging.INFO)
        dtm_key, pits_mask_key = fill_pits.run(dtm_key, border_nodata_mask_key, eomanager)
        eomanager.write(key=pits_mask_key, img_path=os.path.join(output_masks_dir, "filled_pits.tif"), binary=True)
        BulldozerLogger.log("Pits removal: Done.", logging.INFO)
        eomanager.release(key=pits_mask_key)

        # last step: Apply border_nodata_mask
        BulldozerLogger.log("Applying border no data: Starting...", logging.INFO)
        final_dtm = eomanager.get_array(key=dtm_key)[0]
        border_nodata_mask = eomanager.get_array(key=border_nodata_mask_key)[0]
        final_dtm[border_nodata_mask==1] = input_nodata
        BulldozerLogger.log("Applying border no data: Done...", logging.INFO)

        # Write final outputs
        # Step 8[optional]: write final dhm
        if params["generate_dhm"]:
            BulldozerLogger.log("Generating DHM: Starting...", logging.INFO)
            dsm = eomanager.get_array(key=filled_dsm_key)[0, :, :]
            dtm = eomanager.get_array(key=dtm_key)[0, :, :]
            dhm = dsm - dtm
            BulldozerLogger.log("Applying border no data to DHM: Starting...", logging.INFO)
            dhm[border_nodata_mask==1] = input_nodata
            eomanager.release(key=border_nodata_mask_key)
            inner_nodata_mask = eomanager.get_array(key=inner_nodata_mask_key)[0]
            dhm[inner_nodata_mask==1] = input_nodata
            eomanager.release(key=inner_nodata_mask_key)
            BulldozerLogger.log("Applying border no data to DHM: Done...", logging.INFO)
            with rasterio.open(os.path.join(params["output_dir"], "dhm.tif"), "w", **eomanager.get_profile(key=filled_dsm_key)) as dhm_out:
                dhm_out.write(dhm, 1)
            BulldozerLogger.log("Generating DHM: Done.", logging.INFO)
        else:
            # if the DHM is not generated, release the border_nodata_mask memory
            eomanager.release(key=border_nodata_mask_key)
        eomanager.release(key=filled_dsm_key)

        # write final dtm
        eomanager.write(key=dtm_key, img_path=os.path.join(params["output_dir"], "dtm.tif"))
        eomanager.release(key=input_dsm_key)
        eomanager.release(key=dtm_key)


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
    # Parameters to use in the main pipeline
    bulldozer_params = dict()

    # Parameters provided by the user
    input_params = dict()

    # Config file case
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

        # Overrides with CLI parameters provided by the user
        input_params.update(kwargs)

        # Check if all required parameters are defined
        for param in bulldozer_pipeline_params[REQ_PARAM_KEY]:
            if param.name not in input_params.keys() or input_params[param.name] is None:
                if "path" in param.value_label.lower():
                    raise ValueError("No {} provided or invalid YAML key syntax. \nExpected: {}=\"{}\"".format(
                        param.label.lower(), param.name, param.value_label.lower()
                    ))
                else:
                    raise ValueError("No {} provided or invalid YAML key syntax. \nExpected: {}={}".format(
                        param.label.lower(), param.name, param.value_label.lower()
                    ))
            bulldozer_params[param.name] = input_params[param.name]

    else:
        # User directly provides the input parameters (kwargs)
        input_params = kwargs

        # Check if all required parameters are defined
        errors = ""
        api_args = []
        cli_command = "bulldozer"
        for param in bulldozer_pipeline_params[REQ_PARAM_KEY]:
            if param.name not in input_params.keys() or input_params[param.name] is None:
                errors += "No {} provided.\n".format(param.label.lower())
            else:
                bulldozer_params[param.name] = input_params[param.name]

            # complete exception message in case of missing parameter
            cli_command += " -{} {}".format(param.name, param.value_label)
            if "path" in param.value_label.lower():
                api_args.append("{}=\"{}\"".format(param.name, param.value_label.lower()))
            else:
                api_args.append("{}={}".format(param.name, param.value_label.lower()))

        if errors:
            raise ValueError("{} Minimum parameters expected: \n\t-Python API: dsm_to_dtm({})\n\t-CLI: {}".format(
                errors, ", ".join(api_args), cli_command
            ))
    
    # For each optional parameters of Bulldozer check if the user provide a specific value, otherwise retrieve the default value from bulldozer_pipeline_params
    for param in bulldozer_pipeline_params[OPT_PARAM_KEY] + bulldozer_pipeline_params[EXPERT_PARAM_KEY]:
        bulldozer_params[param.name] = input_params[param.name] if param.name in input_params.keys() else param.default_value

    # Retrieves ignored provided parameters (parameters not used by bulldozer)
    ignored_params = set(input_params.keys()).difference(set(bulldozer_pipeline_params[group][param].name for group in bulldozer_pipeline_params.keys() for param in range(len(bulldozer_pipeline_params[group]))))
    if len(ignored_params) > 0:
        bulldozer_params["ignored_params"] = ignored_params

    return bulldozer_params


def get_parser() -> BulldozerArgumentParser:
    """
    Argument parser for Bulldozer (CLI).

    Returns:
        the parser.
    """
    short_lu = "-lu"
    long_lu = "--long_usage"
    short_expert = "-ex"
    long_expert = "--expert_mode"

    # Create parser
    parser = BulldozerArgumentParser(
        description="Bulldozer: CNES pipeline designed to extract DTM from DSM",
        epilog=f"Note: prog {short_lu} or prog {long_lu} for full help. \n      prog {short_expert} or prog {long_expert} for full help with expert parameters.",
        add_help=False
    )

    # Global positional argument: nargs to make it optional for launch using CLI parameters
    parser.add_argument("config_path", type=str, nargs="?", help="Input configuration file.")
    
    # Common optional arguments for full help and version number
    parser.add_argument("-h", "--help", action="help", help="Show short help message and exit.")
    parser.add_argument("-v", "--version", action="version", version="%(prog)s {version}".format(version=__version__),
                        help="Show program's version number and exit.")
    parser.add_argument(short_lu, long_lu, action="store_true", help="Show complete help message and exit.")
    parser.add_argument(short_expert, long_expert, action="store_true", help="Show complete help message with expert parameters and exit.")
    
    # Add bulldozer parameters and split them into required and optional groups
    for group_name, list_params in bulldozer_pipeline_params.items():
        group = parser.add_argument_group(description=group_name)
        for param in list_params:
            # Add argument with the correct store action
            if param.param_type == bool:
                if param.default_value is False:
                    group.add_argument(f"-{param.alias}", f"--{param.name}", action="store_true", default=argparse.SUPPRESS, help=param.description)
                else:
                    group.add_argument(f"-{param.alias}", f"--{param.name}", action="store_false", default=argparse.SUPPRESS, help=param.description)
            else:
                group.add_argument(f"-{param.alias}", f"--{param.name}", type=param.param_type, metavar=param.value_label,
                                   action="store", default=argparse.SUPPRESS, help=param.description)

    return parser


def bulldozer_cli() -> None:
    """
        Call bulldozer main pipeline.
    """
    # Get arguments
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    # Check for long help then delete long help argument
    if args.expert_mode:
        parser.print_help(long_help=True, expert_mode=True)
        sys.exit(0)
    if args.long_usage:
        parser.print_help(long_help=True)
        sys.exit(0)
    del args.expert_mode
    del args.long_usage

    # Execute bulldozer pipeline
    dsm_to_dtm(**vars(args))


if __name__ == "__main__":
    bulldozer_cli()
