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
from bulldozer.dtm_extraction.dtm_extraction import ClothSimulation
from bulldozer.preprocessing.dsm_preprocess import preprocess_pipeline
from bulldozer.postprocessing.dtm_postprocess import postprocess_pipeline
from bulldozer.utils.config_parser import ConfigParser
from bulldozer.utils.logging_helper import BulldozerLogger
from bulldozer.utils.helper import Runtime, retrieve_nodata, DefaultValues

__version__ = "1.0.1"

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

    preprocessed_dsm_path, quality_mask_path = preprocess_pipeline(dsm_path = params['dsm_path'], 
                                                                   output_dir = params['output_dir'], 
                                                                   nb_max_workers = params['nb_max_workers'], 
                                                                   nodata = params['nodata'], 
                                                                   slope_threshold = params['slope_threshold'], 
                                                                   is_four_connexity = params['four_connexity'],
                                                                   minValidHeight = params['min_valid_height'])

    clothSimu = ClothSimulation(params['max_object_width'], 
                                params['uniform_filter_size'], 
                                params['prevent_unhook_iter'],
                                params['num_outer_iter'], 
                                params['num_inner_iter'], 
                                params['mp_tile_size'],
                                params['output_resolution'], 
                                params['nb_max_workers'],
                                params['keep_inter_dtm'])

    raw_dtm_path: str = clothSimu.run(preprocessed_dsm_path, 
                                      params['output_dir'], 
                                      params['nodata'])

    postprocess_pipeline(raw_dtm_path =  raw_dtm_path, 
                         output_dir = params['output_dir'],
                         nb_max_workers = params['nb_max_workers'],
                         quality_mask_path =  quality_mask_path, 
                         generate_dhm = params['generate_dhm'], 
                         dsm_path = params['dsm_path'],
                         check_intersection = params['check_intersection'],
                         nodata = params['nodata'])
    
    if not params['developper_mode']:
        # Remove the raw DTM since the postprocess pipeline generates a refined DTM
        preprocessed_dsm_path = os.path.join(params['output_dir'], 'preprocessed_DSM.tif')
        os.remove(raw_dtm_path)
        os.remove(preprocessed_dsm_path)

def retrieve_params(config_path : str = None, **kwargs):
    """
        Defines the input parameters based on the provided configuration file (if provided), or the kwargs (CLI or Python API).
        For the missing parameters the Bullodzer default values are set.
        
        Args:
            config_path: path to the config file (YAML file expected, refers to the provided template in /conf).
            **kwargs: list of expected arguments if the urser don't provide a configuration file:
                - dsm_path: str (required)
                - output_dir: str (required)
                - nodata: float (optionnal)
                - nb_max_workers: int (optionnal)
                - slope_threshold: float (optionnal)
                - four_connexit: bool (optionnal)
                - min_valid_heigh: float (optionnal)
                - max_object_width: float (optionnal)
                - uniform_filter_size: int (optionnal)
                - prevent_unhook_iter: int (optionnal)
                - num_outer_iter: int (optionnal)
                - num_inner_iter: int (optionnal)
                - mp_tile_size: int (optionnal)
                - output_resolution: float (optionnal)
                - generate_dhm: bool (optionnal)
                - check_intersection: bool (optionnal)
                - developper_mode : bool (optionnal)
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
        
    # Retrieves the nodata value from input DSM metadata if the user didn't provides a specific value
    if bulldozer_params['nodata'] is None:
        bulldozer_params['nodata'] = retrieve_nodata(bulldozer_params['dsm_path'])
    
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