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
This module is used to postprocess the DTM in order to improve its quality.
It required a DTM generated from Bulldozer.
"""

import argparse
import multiprocessing
import os
import sys

import argcomplete
import numpy as np
import rasterio

# Drape cloth filter
import bulldozer.extraction.drape_cloth as dtm_extraction
from bulldozer._version import __version__
from bulldozer.eomultiprocessing.bulldozer_manager import BulldozerContextManager
from bulldozer.eomultiprocessing.utils import read, read_and_get_profile, write
from bulldozer.pipeline.bulldozer_parameters import DEFAULT_NODATA, bulldozer_pipeline_params

# Postprocessing steps of Bulldozer
from bulldozer.postprocessing import fill_pits
from bulldozer.postprocessing.bulldozer_postprocess import run_postprocess

# Preprocessing steps of Bulldozer
from bulldozer.preprocessing.border_detection import border_detector
from bulldozer.preprocessing.dsm_filling import dsm_filler
from bulldozer.preprocessing.ground_detection import ground_anchors_detector
from bulldozer.preprocessing.regular_detection import regular_detector

# Building arguments parser
from bulldozer.utils.bulldozer_argparse import EXPERT_PARAM_KEY, OPT_PARAM_KEY, REQ_PARAM_KEY, BulldozerArgumentParser
from bulldozer.utils.bulldozer_logger import Runtime, init_logfile, setup_logger
from bulldozer.utils.config_parser import ConfigParser


@Runtime
def dsm_to_dtm(config_path: str | None = None, **kwargs: int) -> None:
    """
    Main pipeline orchestrator.

    Args:
        config_path: path to the config file
                     (YAML file expected, refers to the provided template in /conf).
        **kwargs: bulldozer parameters
                  (used if the user don't provide a configuration file).

    """
    # Retrieves Bulldozer settings from the config file, the CLI parameters or the Python API parameters
    params = retrieve_params(config_path, **kwargs)

    # If the target output directories does not exist, creates it
    os.makedirs(params["output_dir"], exist_ok=True)

    logger = setup_logger(params["output_dir"])
    init_logfile()
    logger.debug(
        "Bulldozer input parameters: \n"
        + "".join("\t- " + str(key) + ": " + str(value) + "\n" for key, value in params.items())
    )

    with rasterio.open(params["dsm_path"]) as ds:
        logger.debug(
            "Input DSM profile: \n"
            + "".join("\t- " + str(key) + ": " + str(value) + "\n" for key, value in ds.profile.items())
        )

    # Warns the user that he/she provides parameters that are not used
    if "ignored_params" in params:
        logger.warning(
            f"The following input parameters are ignored: {params['ignored_params']}. "
            + "\nPlease refer to the documentation for the list of valid parameters."
        )

    # Retrieves the number of CPU if the number of available workers if
    # the user didn't provide a specific value
    if params["nb_max_workers"] is None:
        params["nb_max_workers"] = multiprocessing.cpu_count()
        logger.debug(
            '"nb_max_workers" parameter is not set. The default value is used:'
            + f" maximum number of available CPU core ({params['nb_max_workers']})."
        )

    logger.debug(f"Creates a pool of {params['nb_max_workers']} process in {params['mp_context']} context")

    with BulldozerContextManager(params, tile_mode=True) as manager:
        # Open the input dsm that might be noisy and full of nodata...
        dsm, input_profile = read_and_get_profile(params["dsm_path"])

        # Nodata value handling
        input_nodata = input_profile["nodata"]
        if input_nodata is None:
            logger.debug(
                "The provided nodata value is None. "
                + f"Bulldozer will use its own nodata default value ({DEFAULT_NODATA}) "
                + "during the pipeline run (Cython constraint)."
            )
            dsm[dsm == None] = DEFAULT_NODATA  # noqa: E711 pylint: disable=singleton-comparison
            dsm = np.nan_to_num(dsm, copy=False, nan=DEFAULT_NODATA)
            pipeline_nodata = DEFAULT_NODATA
        elif np.isnan(input_nodata):
            logger.debug(
                "The provided nodata value is NaN. "
                + f"Bulldozer will use its own nodata default value ({DEFAULT_NODATA}) "
                + "during the pipeline run (Cython constraint)."
            )
            dsm = np.nan_to_num(dsm, copy=False, nan=DEFAULT_NODATA)
            pipeline_nodata = DEFAULT_NODATA
        else:
            pipeline_nodata = input_nodata
            logger.debug(f"Nodata retrieved and used in the pipeline: {pipeline_nodata}")

        if manager.in_memory:  # cleaned_dsm_key is a numpy array
            clean_dsm_key = dsm
        else:  # we save cleaned dsm, cleaned_dsm_key is the path to the file
            if pipeline_nodata == DEFAULT_NODATA:
                clean_dsm_profile = input_profile.copy()
                clean_dsm_profile["nodata"] = pipeline_nodata
                clean_dsm_key = manager.get_path("clean_dsm.tif", key=manager.tmp_key)
                write(dsm, clean_dsm_key, clean_dsm_profile)
            else:
                clean_dsm_key = params["dsm_path"]
        del dsm

        # If the user doesn't provide an DSM altimetric accuracy, set it to default value: 2 * planimetric resolution
        if params["dsm_z_accuracy"] is None:
            params["dsm_z_accuracy"] = 2 * input_profile["transform"][0]
            logger.debug(
                '"dsm_z_accuracy" parameter is null, used default value: '
                + f"2*planimetric resolution ({params['dsm_z_accuracy']}m)."
            )

        # Step 1: Compute the regular area mask
        # Take the maximum slope between the slope provided by the user (converted in meter) and the slope derived
        # from the altimetric dsm accuracy
        regular_slope: float = max(
            float(params["max_ground_slope"]) * input_profile["transform"][0] / 100.0,
            params["dsm_z_accuracy"],
        )

        regular_mask_key = regular_detector.detect_regular_areas(
            dsm_key=clean_dsm_key,
            dsm_profile=input_profile,
            regular_slope=regular_slope,
            nodata=pipeline_nodata,
            max_object_size=params["max_object_size"],
            reg_filtering_iter=params["reg_filtering_iter"],
            manager=manager,
        )

        # Step 2: Detect inner and border nodata masks
        border_nodata_mask_key, inner_nodata_mask_path = border_detector.detect_border_nodata(
            dsm_key=clean_dsm_key, dsm_profile=input_profile, nodata=pipeline_nodata, manager=manager
        )

        # Step 3: Fill the input DSM and compute the uncertainties
        filled_dsm_key = dsm_filler.fill_dsm(
            dsm_key=clean_dsm_key,
            regular_mask_key=regular_mask_key,
            border_nodata_mask_key=border_nodata_mask_key,
            dsm_profile=input_profile,
            nodata=pipeline_nodata,
            max_object_size=params["max_object_size"],
            manager=manager,
        )

        del clean_dsm_key

        # Step 4 [optional]:
        # post anchor mask computation (first drape cloth + terrain pixel detection)
        # Run a first drape cloth simulation to minimize the underestimation the terrain height (common issue)
        # All regular pixels where the diff Z is lower or equal than dtm_max_error meters will be labeled as
        # possible terrain points knowing that the drape cloth will be run again.
        if params["activate_ground_anchors"]:
            logger.info("First pass of a drape cloth filter: Starting...")
            dtm_key = dtm_extraction.drape_cloth(
                filled_dsm_key=filled_dsm_key,
                ground_mask_key=params["ground_mask_path"],
                filled_dsm_profile=input_profile,
                manager=manager,
                max_object_size=params["max_object_size"],
                prevent_unhook_iter=params["prevent_unhook_iter"],
                num_outer_iterations=params["num_outer_iter"],
                num_inner_iterations=params["num_inner_iter"],
                inter_dtm_filename="dtm_first_pass.tif",
            )
            logger.info("First pass of a drape cloth filter: Done.")

            ground_anchors_mask_key = ground_anchors_detector.detect_ground_anchors(
                intermediate_dtm_key=dtm_key,
                dsm_key=filled_dsm_key,
                regular_mask_key=regular_mask_key,
                ground_mask_path=params["ground_mask_path"],
                dsm_profile=input_profile,
                dsm_z_accuracy=params["dsm_z_accuracy"],
                manager=manager,
            )

        else:
            ground_anchors_mask_key = None

        del regular_mask_key

        # Step 5: Compute final DTM with post processed predicted terrain point
        logger.info("Main pass of a drape cloth filter: Starting...")
        dtm_key = dtm_extraction.drape_cloth(
            filled_dsm_key=filled_dsm_key,
            ground_mask_key=ground_anchors_mask_key,
            filled_dsm_profile=input_profile,
            manager=manager,
            max_object_size=params["max_object_size"],
            prevent_unhook_iter=params["prevent_unhook_iter"],
            num_outer_iterations=params["num_outer_iter"],
            num_inner_iterations=params["num_inner_iter"],
            inter_dtm_filename="dtm_second_pass.tif",
        )
        logger.info("Main pass of a drape cloth filter: Done.")

        del ground_anchors_mask_key, filled_dsm_key

        # Step 6: remove pits (includes clean DTM)
        logger.info("Pits removal: Starting...")
        dtm_key = fill_pits.run(
            dtm_key=dtm_key,
            border_nodata_mask_key=border_nodata_mask_key,
            dtm_profile=input_profile,
            nodata=input_nodata,
            manager=manager,
        )
        logger.info("Pits removal: Done.")

        # Step 7 : post process - compare with DSM (optional) + computes nDSM (optional)
        if params["enforce_dtm_below_dsm"] or params["generate_ndsm"]:
            logger.info("Postprocessing: Starting...")
            # Compute nodata mask
            nodata_mask = (
                read(border_nodata_mask_key).astype(np.bool)
                if isinstance(border_nodata_mask_key, str)
                else border_nodata_mask_key
            )
            inner_nodata_mask = read(inner_nodata_mask_path)
            nodata_mask[:] = np.logical_or(nodata_mask, inner_nodata_mask)
            del inner_nodata_mask, border_nodata_mask_key

            # Run postprocess
            postprocess_dict = run_postprocess(
                params["dsm_path"],
                dtm_key,
                nodata_mask,
                input_nodata,
                params["enforce_dtm_below_dsm"],
                params["generate_ndsm"],
            )

            del nodata_mask
            if "dtm" in postprocess_dict.keys():
                dtm_key = postprocess_dict["dtm"]
                logger.info("Postprocessing: Done.")
        else:
            del border_nodata_mask_key

        # Step 8 : write outputs
        logger.info("Writing DTM...")
        dtm_filename = "dtm.tif"
        if isinstance(dtm_key, str):
            # DTM has already been written in tmp folder
            manager.move_tif(dtm_key, dtm_filename, key="out")
        else:  # DTM is a numpy array
            manager.write_tif(dtm_key, dtm_filename, input_profile, key="out")

        if params["generate_ndsm"]:
            ndsm = postprocess_dict["ndsm"]
            del postprocess_dict, dtm_key
            logger.info("Writing nDSM...")
            manager.write_tif(ndsm, "ndsm.tif", input_profile, key="out")


def retrieve_params(config_path: str | None = None, **kwargs: int) -> dict:
    """
    Defines the input parameters based on the provided configuration file (if provided),
    or the kwargs (CLI or Python API).
    For the missing parameters the Bulldozer default values are used.

    Args:
        config_path: path to the config file
                     (YAML file expected, refers to the provided template in /conf).
        **kwargs: list of expected arguments if the user doesn't provide a configuration
                file. Refers to the documentation to get the full parameter list.

    Returns:
        the dict containing the input parameters.
    """
    # Parameters used in the main pipeline
    bulldozer_params = {}

    # Parameters provided by the user
    input_params = {}

    # Config file case
    if config_path:
        # Configuration file format check
        if not (config_path.endswith(".yaml") or config_path.endswith(".yml")):
            raise ValueError(
                'Expected yaml configuration file: "config_path" argument should '
                + f"be a path to a Yaml file (here: {config_path})"
            )

        # Configuration file existence check
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f'The input configuration file "{config_path}" doesn\'t exist')

        # Retrieves all the settings
        parser = ConfigParser(False)
        input_params = parser.read(config_path)

        # Overrides with CLI parameters provided by the user
        input_params.update(kwargs)

        # Check if all required parameters are defined
        for param in bulldozer_pipeline_params[REQ_PARAM_KEY]:
            if param.name not in input_params.keys() or input_params[param.name] is None:
                if not param.value_label:
                    value = "<bool_value>"
                elif "path" in param.value_label.lower():
                    value = f'"{param.value_label.lower()}"'
                else:
                    value = param.value_label.lower()
                raise ValueError(
                    f"No {param.label.lower()} provided or invalid YAML key syntax. \nExpected: {param.name}={value}"
                )
            bulldozer_params[param.name] = input_params[param.name]

    else:
        # User directly provides the input parameters (kwargs)
        input_params = kwargs

        # Check if all required parameters are defined
        errors = ""
        api_args = []
        cli_command = "bulldozer"
        for param in bulldozer_pipeline_params[REQ_PARAM_KEY]:
            if param.name not in input_params or input_params[param.name] is None:
                errors += f"No {param.label.lower()} provided.\n"
            else:
                bulldozer_params[param.name] = input_params[param.name]

            # complete exception message in case of missing parameter
            cli_command += f" -{param.name} {param.value_label}"
            if not param.value_label:
                api_args.append(f"{param.name}=<bool_value>")
            elif "path" in param.value_label.lower():
                api_args.append(f'{param.name}="{param.value_label.lower()}"')
            else:
                api_args.append(f"{param.name}={param.value_label.lower()}")

        if errors:
            str_api_args = ", ".join(api_args)
            raise ValueError(
                f"{errors} Minimum parameters expected: "
                f"\n\t-Python API: dsm_to_dtm({str_api_args})\n\t-CLI: {cli_command}"
            )

    # For each optional parameters of Bulldozer
    # check if the user provide a specific value,
    # otherwise retrieve the default value from bulldozer_pipeline_params
    for param in bulldozer_pipeline_params[OPT_PARAM_KEY] + bulldozer_pipeline_params[EXPERT_PARAM_KEY]:
        bulldozer_params[param.name] = (
            input_params[param.name] if param.name in input_params.keys() else param.default_value
        )

    # Retrieves ignored provided parameters (parameters not used by bulldozer)
    all_params = [param.name for group, params in bulldozer_pipeline_params.items() for param in params]

    ignored_params = {key for key in input_params.keys() if key not in all_params}

    if len(ignored_params) > 0:
        bulldozer_params["ignored_params"] = ", ".join(ignored_params)

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
        epilog=f"Note: prog {short_lu} or prog {long_lu} for full help. \n      "
        f"prog {short_expert} or prog {long_expert} for "
        f"full help with expert parameters.",
        add_help=False,
    )

    # Global positional argument
    # nargs to make it optional for launch using CLI parameters
    parser.add_argument("config_path", type=str, nargs="?", help="Input configuration file.")

    # Common optional arguments for full help and version number
    parser.add_argument("-h", "--help", action="help", help="Show short help message and exit.")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit.",
    )
    parser.add_argument(
        short_lu,
        long_lu,
        action="store_true",
        help="Show complete help message and exit.",
    )
    parser.add_argument(
        short_expert,
        long_expert,
        action="store_true",
        help="Show complete help message with expert parameters and exit.",
    )

    # Add bulldozer parameters and split them into required and optional groups
    for group_name, list_params in bulldozer_pipeline_params.items():
        group = parser.add_argument_group(description=group_name)
        for param in list_params:
            # Add argument with the correct store action
            if param.param_type is bool:
                if param.default_value is False:
                    group.add_argument(
                        f"-{param.alias}",
                        f"--{param.name}",
                        action="store_true",
                        default=argparse.SUPPRESS,
                        help=param.description,
                    )
                else:
                    group.add_argument(
                        f"-{param.alias}",
                        f"--{param.name}",
                        action="store_false",
                        default=argparse.SUPPRESS,
                        help=param.description,
                    )
            else:
                group.add_argument(
                    f"-{param.alias}",
                    f"--{param.name}",
                    type=param.param_type,
                    metavar=param.value_label,
                    action="store",
                    default=argparse.SUPPRESS,
                    choices=param.choices,
                    help=param.description,
                )

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
