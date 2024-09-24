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
from bulldozer.utils.logging_helper import BulldozerLogger
from bulldozer.utils.helper import Runtime, DefaultValues
from bulldozer._version import __version__

import bulldozer.eoscale.manager as eom

# Preprocessing steps of Bulldozer
import bulldozer.preprocessing.outliers.histogram as preprocess_histogram_outliers
import bulldozer.preprocessing.regular_detection.regular_detector as preprocess_regular_detector
import bulldozer.preprocessing.fill.prefill_dsm as prefill_dsm
import bulldozer.preprocessing.fill.fill_dsm as fill_dsm
import bulldozer.preprocessing.anchorage_prediction.anchorage_predictor as preprocess_anchors_detector
import bulldozer.preprocessing.bordernodata as bn
import bulldozer.preprocessing.bordernodata.border as border

# Drape cloth filter
import bulldozer.extraction.drape_cloth as dtm_extraction

# Postprocessing steps of Bulldozer
import bulldozer.postprocessing.ground_detection.post_anchorage_detection as postprocess_anchorage
import bulldozer.postprocessing.fill_pits as fill_pits


def write_quality_mask(keys_to_write: List[str],
                       quality_mask_path: str,
                       eomanager: eom.EOContextManager):
    """
    Writes quality mask. Each key given in the input list will be a specific boolean band in the final quality mask.

    :param keys_to_write: List of the eomanager's masks keys to write in the quality mask
    :param quality_mask_path: path of the quality mask to write
    :param eomanager: EOContextManager instance in which to retrieve the different masks to write
    """
    quality_ds_profile = None
    for key in keys_to_write:
        profile = eomanager.get_profile(key)
        # check input mask consistency
        if profile['dtype'] != np.uint8:
            BulldozerLogger.log(f"Mask to write dtype is not set to uint8", logging.ERROR)
        elif profile['count'] != 1:
            BulldozerLogger.log(f"Mask to write has several layers", logging.ERROR)
        elif profile['nodata'] is not None:
            BulldozerLogger.log(f"Mask to write has a nodata value provided", logging.ERROR)

        if quality_ds_profile is None:
            quality_ds_profile = copy(profile)
        else:
            # check consistency with previous mask
            if profile["height"] != quality_ds_profile["height"] or profile["width"] != quality_ds_profile["width"]:
                BulldozerLogger.log(f"Masks to write have different shape", logging.ERROR)
            elif profile["transform"] != quality_ds_profile["transform"]:
                BulldozerLogger.log(f"Masks to write have different transform", logging.ERROR)
            elif profile["crs"] != quality_ds_profile["crs"]:
                BulldozerLogger.log(f"Masks to write have different crs code", logging.ERROR)
            quality_ds_profile["count"] += 1

    # write quality mask
    quality_ds_profile['interleave'] = 'band'
    quality_ds_profile['driver'] = 'GTiff'
    quality_ds_profile['nodata'] = 255

    with rasterio.open(quality_mask_path, 'w', nbits=1, **quality_ds_profile) as q_mask_dataset:
        idx = 1
        for key in keys_to_write:
            q_mask_dataset.write_band(idx, eomanager.shared_resources[key].get_array()[0, :, :])
            idx += 1


@Runtime
def dsm_to_dtm(config_path: str = None, **kwargs) -> None:
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

    logger = BulldozerLogger.getInstance(logger_file_path=os.path.join(params['output_dir'], "trace_" + datetime.now().strftime("%d.%m.%Y_%H:%M:%S") + ".log"))

    BulldozerLogger.log("Bulldozer input parameters: \n" + "".join("\t- " + str(key) + ": " + str(value) + "\n" for key, value in params.items()), logging.DEBUG)

    # Warns the user that he/she provides parameters that are not used
    if params['ignored_params']:
        BulldozerLogger.log("The following input parameters are ignored: {}. \nPlease refer to the documentation for the list of valid parameters.".format(', '.join(params['ignored_params'])), logging.WARNING)

    with eom.EOContextManager(nb_workers=params['nb_max_workers'], tile_mode=True) as eomanager:

        # Open the input dsm that might be noisy and full of nodatas...
        input_dsm_key = eomanager.open_raster(raster_path=params['dsm_path'])
        max_object_size: float = 1.0 / params['min_object_spatial_frequency']

        # Step 1: TODO local denoising
        noisy_mask_key = eomanager.create_image(eomanager.get_profile(input_dsm_key))

        if params["developer_mode"]:
            noisy_mask_path: str = os.path.join(params["output_dir"], "noisy_mask.tif")
            eomanager.write(key=noisy_mask_key, img_path=noisy_mask_path)

        # Step 2: Compute the regular area mask
        BulldozerLogger.log("Regular mask computation: Starting...", logging.INFO)
        regular_slope: float = max(float(params["max_ground_slope"]) * eomanager.get_profile(key=input_dsm_key)["transform"][0] / 100.0, params['dsm_z_precision'])
        regular_outputs = preprocess_regular_detector.run(dsm_key=input_dsm_key,
                                                          noisy_key=noisy_mask_key,
                                                          eomanager=eomanager,
                                                          regular_slope=regular_slope)
        regular_mask_key = regular_outputs["regular_mask"]
        BulldozerLogger.log("Regular mask computation: Done...", logging.INFO)

        if params["developer_mode"]:
            regular_mask_path: str = os.path.join(params["output_dir"], "regular_mask.tif")
            eomanager.write(key=regular_mask_key, img_path=regular_mask_path)

        # Step 2b : Inner and Border no data masks
        BulldozerLogger.log("Starting inner_nodata_mask and border_nodata_mask building", logging.DEBUG)

        nodata_value = eomanager.get_profile(key=input_dsm_key)["nodata"]

        inner_outer_result = border.run(dsm_key=input_dsm_key,
                                        eomanager=eomanager,
                                        nodata=nodata_value)
        
        inner_no_data_mask_key = inner_outer_result["inner_no_data_mask"]
        border_no_data_mask_key = inner_outer_result["border_no_data_mask"]

        if params["developer_mode"]:
            border_no_data_mask_path: str = os.path.join(params["output_dir"], "border_no_data.tif")
            eomanager.write(key=border_no_data_mask_key, img_path=border_no_data_mask_path)

            inner_no_data_mask_path: str = os.path.join(params["output_dir"], "inner_no_data.tif")
            eomanager.write(key=inner_no_data_mask_key, img_path=inner_no_data_mask_path)

        BulldozerLogger.log("inner_nodata_mask and border_nodata_mask generation: Done", logging.INFO)

        # Step 3: Fill the input DSM and compute the uncertainties
        BulldozerLogger.log("Filling the DSM : Starting...", logging.INFO)
        fill_outputs = fill_dsm.run(dsm_key=input_dsm_key,
                                    mask_key=regular_mask_key,
                                    eomanager=eomanager)
        BulldozerLogger.log("Filling the DSM and computing the uncertainties: Done", logging.INFO)

        filled_dsm_key = fill_outputs["filled_dsm"]

        if params["developer_mode"]:
            filled_dsm_path: str = os.path.join(params["output_dir"], "filled_dsm.tif")
            eomanager.write(key=filled_dsm_key, img_path=filled_dsm_path)

        BulldozerLogger.log("End fill dsm", logging.INFO)

        # Step 4 - optional: pre anchor mask computation
        if params['pre_anchor_points_activation']:
            # TODO if COS then preprocess_anchorage_mask_key initialised to the COS
            BulldozerLogger.log("Predicting anchorage points : Starting...", logging.INFO)
            preprocess_anchorage_mask_key = preprocess_anchors_detector.run(filled_dsm_key=filled_dsm_key,
                                                                            regular_mask_key=regular_mask_key,
                                                                            max_object_size=max_object_size,
                                                                            eomanager=eomanager)
            BulldozerLogger.log("Predicting anchorage points : Done", logging.INFO)

            if params["developer_mode"]:
                preprocess_anchorage_mask_path: str = os.path.join(params["output_dir"], "preprocess_anchorage_mask.tif")
                eomanager.write(key=preprocess_anchorage_mask_key, img_path=preprocess_anchorage_mask_path)
        else:
            preprocess_anchorage_mask_key = eomanager.create_image(eomanager.get_profile(regular_mask_key))

        # Step 4.5 - optional: post anchor mask computation (first drape cloth + terrain pixel detection)
        # Brute force post process to minimize a side effect of the drape that often underestimates the terrain height
        # All regular pixels where the diff Z is lower or equal than dtm_max_error meters will be labeled as possible terrain points.
        # Knowing that the drape cloth will be run again.
        if params['post_anchor_points_activation']:
            BulldozerLogger.log("First pass of a drape cloth filter: Starting...", logging.INFO)
            # TODO handle Land use map (convert it to reach: ground=1/else=0)
            cos_mask_key = eomanager.create_image(eomanager.get_profile(regular_mask_key))
            dtm_key = dtm_extraction.drape_cloth(filled_dsm_key=filled_dsm_key,
                                                 predicted_anchorage_mask_key=cos_mask_key,
                                                 eomanager=eomanager,
                                                 max_object_size=max_object_size,
                                                 prevent_unhook_iter=params["prevent_unhook_iter"],
                                                 spring_tension=params["cloth_tension_force"],
                                                 num_outer_iterations=params["num_outer_iter"],
                                                 num_inner_iterations=params["num_inner_iter"])
            BulldozerLogger.log("First pass of a drape cloth filter: Done.", logging.INFO)

            if params["developer_mode"]:
                inter_dtm_path: str = os.path.join(params["output_dir"], "dtm_first_pass.tif")
                eomanager.write(key=dtm_key, img_path=inter_dtm_path)

            # TODO for steps 4.5: add a conditional statement to activate the second pass
            # Attempt to detect terrain pixels
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
            eomanager.release(key=dtm_key)

            if params["developer_mode"]:
                output_post_anchorage_path: str = os.path.join(params["output_dir"], "post_anchorage_mask.tif")
                eomanager.write(key=post_anchorage_mask_key, img_path=output_post_anchorage_path)
        else:
            post_anchorage_mask_key = eomanager.create_image(eomanager.get_profile(regular_mask_key))

        if params['pre_anchor_points_activation']:
            # Union of post_anchorage_mask with pre_process_anchorage_mask
            post_anchors = eomanager.get_array(key=post_anchorage_mask_key)
            pre_anchors = eomanager.get_array(key=preprocess_anchorage_mask_key)
            np.logical_or(pre_anchors[0, :, :], post_anchors[0, :, :], out=post_anchors[0, :, :])

        # Step 5: Compute final DTM with post processed predicted terrain point
        BulldozerLogger.log("Main pass of a drape cloth filter: Starting...", logging.INFO)
        dtm_key = dtm_extraction.drape_cloth(filled_dsm_key=filled_dsm_key,
                                             predicted_anchorage_mask_key=post_anchorage_mask_key,
                                             eomanager=eomanager,
                                             max_object_size=max_object_size,
                                             prevent_unhook_iter=params["prevent_unhook_iter"],
                                             spring_tension=params["cloth_tension_force"],
                                             num_outer_iterations=params["num_outer_iter"],
                                             num_inner_iterations=params["num_inner_iter"])
        BulldozerLogger.log("Main pass of a drape cloth filter: Done.", logging.INFO)

        if params["developer_mode"]:
            eomanager.write(key=dtm_key, img_path=os.path.join(params["output_dir"], "dtm_second_pass.tif"))

        # Step 6: optional - reverse drape cloth
        # Needs: dtm_key and post_anchors to snap and dsm_max_z for int
        if params['reverse_drape_cloth_activation']:
            BulldozerLogger.log("Reverse pass of a drape cloth filter: Starting...", logging.INFO)
            reverse_dtm_key = dtm_extraction.reverse_drape_cloth(filled_dsm_key=filled_dsm_key,
                                                                 first_pass_dtm_key=dtm_key,
                                                                 pre_anchorage_mask_key=preprocess_anchorage_mask_key,
                                                                 post_anchorage_mask_key=post_anchorage_mask_key,
                                                                 eomanager=eomanager,
                                                                 max_object_size=max_object_size,
                                                                 prevent_unhook_iter=params["prevent_unhook_iter"],
                                                                 spring_tension=params["cloth_tension_force"],
                                                                 num_outer_iterations=params["num_outer_iter"],
                                                                 num_inner_iterations=params["num_inner_iter"])

            if params["developer_mode"]:
                eomanager.write(key=reverse_dtm_key, img_path=os.path.join(params["output_dir"], "reverse_dtm.tif"))

            final_dtm = eomanager.get_array(key=dtm_key)
            reverse_dtm = eomanager.get_array(key=reverse_dtm_key)
            final_dtm[0, :, :] += reverse_dtm[0, :, :]
            final_dtm[0, :, :] /= 2.0
            eomanager.release(key=reverse_dtm_key)
            BulldozerLogger.log("Reverse pass of a drape cloth filter: Done...", logging.INFO)

        # Step 8: remove pits
        BulldozerLogger.log("Pits removal: Starting.", logging.INFO)
        dtm_key, pits_mask_key = fill_pits.run(dtm_key, eomanager)
        if params["developer_mode"]:
            eomanager.write(key=pits_mask_key, img_path=os.path.join(params["output_dir"], "fill_pits_mask.tif"))
        BulldozerLogger.log("Pits removal: Done.", logging.INFO)

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
            with rasterio.open(os.path.join(params["output_dir"], "dhm.tif"), "w", **eomanager.get_profile(key=filled_dsm_key)) as dhm_out:
                dhm_out.write(dhm, 1)
            BulldozerLogger.log("Generating DHM: Done.", logging.INFO)

        # write final dtm
        eomanager.write(key=dtm_key, img_path=os.path.join(params["output_dir"], "final_dtm.tif"))

        # write quality mask
        BulldozerLogger.log("Starting write_quality_mask...", logging.INFO)
        write_quality_mask([regular_mask_key, pits_mask_key, inner_no_data_mask_key, border_no_data_mask_key],
                           os.path.join(params["output_dir"], "quality_mask.tif"),
                           eomanager)
        
        BulldozerLogger.log("write_quality_mask generation: Done", logging.INFO)

        # final releases
        eomanager.release(key=dtm_key)
        eomanager.release(key=filled_dsm_key)

        # And finally we are done ! It is exhausting to extract a DTM dont you think ?
        BulldozerLogger.log("And finally we are done ! It is exhausting to extract a DTM don't you think ?", logging.INFO)


def retrieve_params(config_path: str = None, **kwargs):
    """
        Defines the input parameters based on the provided configuration file (if provided), or the kwargs (CLI or Python API).
        For the missing parameters the Bulldozer default values are set.
        
        Args:
            config_path: path to the config file (YAML file expected, refers to the provided template in /conf).
            **kwargs: list of expected arguments if the user don't provide a configuration file:
                - dsm_path: str (required)
                - output_dir: str (required)
                - nb_max_workers: int (optional, 8 by default)
                - dsm_z_precision: float (optional, 1.0 by default)
                - fill_search_radius: int (optional, 100 by default)
                - max_ground_slope: float (optional, 20.0 % by default)
                - min_object_spatial_frequency: float (optional, 0.0625 by default)
                - dtm_max_error: float (optional, 2.0 meters by default)
                - cloth_tension_force: int (optional, 3 by default)
                - prevent_unhook_iter: int (optional, 10 by default)
                - num_outer_iter: int (optional, 100 by default)
                - num_inner_iter: int (optional, 10 by default)
                - output_resolution: float (optional, null by default)
                - generate_dhm: bool (optional, True by default)
                - developer_mode : bool (optional, False by default)
                - pre_anchor_points_activation : bool (optional, False by default)
                - post_anchor_points_activation : bool (optional, False by default)
                - reverse_drape_cloth_activation : bool (optional, False by default)
                refers to the documentation to understand the use of each parameter.
    """
    bulldozer_params = dict()
    # Config path provided case

    input_params = dict()

    if config_path:
        # Configuration file format check
        if not (config_path.endswith('.yaml') or config_path.endswith('.yml')):
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
    else:
        # User directly provides the input parameters (kwargs)
        input_params = kwargs
        if 'dsm_path' not in input_params:
            raise ValueError('No DSM path provided or invalid argument syntax. Expected: \n\t-Python API: dsm_to_dtm(dsm_path="<path>")\n\t-CLI: bulldozer -in <path>')
        bulldozer_params['dsm_path'] = input_params['dsm_path']
        if 'output_dir' not in input_params:
            raise ValueError('No output diectory path provided or invalid argument syntax. Expected: \n\t-Python API:  dsm_to_dtm(dsm_path="<path>, output_dir="<path>")\n\t-CLI: bulldozer -in <path> -out path')
        bulldozer_params['output_dir'] = input_params['output_dir']

    # For each optional parameters of Bulldozer check if the user provide a specific value, otherwise retrieve the default value from DefaultValues enum
    for key, value in DefaultValues.items():
        bulldozer_params[key.lower()] = input_params[key.lower()] if key.lower() in input_params.keys() else value
    
    # Retrieves the number of CPU if the number of available workers if the user didn't provide a specific value
    if bulldozer_params['nb_max_workers'] is None:
        bulldozer_params['nb_max_workers'] = multiprocessing.cpu_count()
    
    # Retrieves ignored provided ipunt parameters
    bulldozer_params['ignored_params'] = set(input_params.keys()).difference(set([key.lower() for key in DefaultValues.keys()] + ['dsm_path', 'output_dir']))

    return bulldozer_params
    

def get_parser():
    """
    Argument parser for Bulldozer (CLI).

    Returns:
        the parser.
    """
    parser = argparse.ArgumentParser(description="Bulldozer")

    parser.add_argument(
        "--conf", required=True, type=str, help="Bulldozer config file"
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    
    # TODO add all the parameters
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
