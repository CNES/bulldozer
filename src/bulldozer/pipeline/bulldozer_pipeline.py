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
from bulldozer.utils.helper import Runtime, retrieve_nodata

#import threading
import time

__version__ = "1.0.0"


# import psutil

# stop_thread = False

# def memory() :
#     t0 = time.time()
#     with open('memory.txt', 'w') as out :
#         process = psutil.Process(os.getpid())
#         while    :
#             t = time.time()True
#             mem = process.memory_info()
#             print("%.2f" % (t - t0), "%.2f" %  (mem.rss / 1024 / 1024), end='', file=out)
            
#             children = process.children()
#             for child in children:
#                 print('\t', "%.2f" %  (child.memory_info().rss / 1024 / 1024), end='', file=out) 
        
#             print('', file=out)
#             time.sleep(0.5)
#             global stop_thread
#             if stop_thread:
#                 out.close()
#                 break

@Runtime
def dsm_to_dtm(config_path : str = None, **kwargs) -> None:
    """
        Pipeline orchestrator.
        
        Args:
            config_path: path to the config file (YAML file expected, refers to the provided template in /conf).
            **kwargs: you can directly provide the parameters but we expect the following parameters name :
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
                refers to the config file template to understand the use of each parameter.

    """
    # Config path provided case
    if config_path:
        # Configuration file format check
        if not (config_path.endswith('.yaml') or config_path.endswith('.yml')) :
            raise ValueError('Expected yaml configuration file: \'config_path\' argument should be a path to a Yaml file (here: {})'.format(config_path))

        # Configuration file existence check
        if not os.path.isfile(config_path):
            raise FileNotFoundError('The input configuration file \'{}\' doesn\'t exist'.format(config_path))
        
        # Retrieves all the settings
        parser = ConfigParser(False)
        cfg = parser.read(config_path)
        dsm_path = cfg['dsmPath']
        output_dir = cfg['outputDir']
        nodata = cfg['noData']
        nb_max_workers = cfg['nbMaxWorkers']
        slope_threshold = cfg['slopeThreshold']
        four_connexity = cfg['fourConnexity']
        min_valid_height = cfg['minValidHeight']
        max_object_width = cfg['maxObjectWidth']
        uniform_filter_size = cfg['uniformFilterSize']
        prevent_unhook_iter = cfg['preventUnhookIter']
        num_outer_iter = cfg['numOuterIter']
        num_inner_iter = cfg['numInnerIter']
        mp_tile_size = cfg['mpTileSize']
        output_resolution = cfg['outputResolution']
        generate_dhm = cfg['generateDhm']
        check_intersection = cfg['checkIntersection']
        developper_mode = cfg['developperMode']
    # User directly provide the configuration as dsm_to_dtm parameters case
    else :
        if not 'dsm_path' in kwargs:
            raise ValueError('No DSM path provided or invalid argument syntax. Expected dsm_to_dtm(dsm_path="<path>")')
        dsm_path = kwargs['dsm_path']
        if not 'output_dir' in kwargs:
            raise ValueError('No output diectory path provided or invalid argument syntax. Expected dsm_to_dtm(dsm_path="<path>, output_dir="<path>")')
        output_dir = kwargs['output_dir']
        nodata = kwargs['nodata'] if 'nodata' in kwargs else None
        nb_max_workers = kwargs['nb_max_workers'] if 'nb_max_workers' in kwargs else None
        slope_threshold = kwargs['slope_threshold'] if 'slope_threshold' in kwargs else None
        four_connexity = kwargs['four_connexity'] if 'four_connexity' in kwargs else None
        min_valid_height = kwargs['min_valid_height'] if 'min_valid_height' in kwargs else None
        max_object_width = kwargs['max_object_width'] if 'max_object_width' in kwargs else None
        uniform_filter_size = kwargs['uniform_filter_size'] if 'uniform_filter_size' in kwargs else None
        prevent_unhook_iter = kwargs['prevent_unhook_iter'] if 'prevent_unhook_iter' in kwargs else None
        num_outer_iter = kwargs['num_outer_iter'] if 'num_outer_iter' in kwargs else None
        num_inner_iter = kwargs['num_inner_iter'] if 'num_inner_iter' in kwargs else None
        mp_tile_size = kwargs['mp_tile_size'] if 'mp_tile_size' in kwargs else None
        output_resolution = kwargs['output_resolution'] if 'output_resolution' in kwargs else None
        generate_dhm = kwargs['generate_dhm'] if 'generate_dhm' in kwargs else None
        check_intersection = kwargs['check_intersection'] if 'check_intersection' in kwargs else None
        developper_mode = kwargs['developper_mode'] if 'developper_mode' in kwargs else None
    
    # If the target output directory does not exist, creates it
    if not os.path.isdir(output_dir):
            os.mkdir(output_dir) 

    logger = BulldozerLogger.getInstance(logger_file_path=os.path.join(output_dir, "trace_" + datetime.now().strftime("%d.%m.%Y_%H:%M:%S") +".log"))
   
    # x = threading.Thread(target=memory)
    # x.start()

    # Retrieves the nodata value from the config file or the DSM metadata
    nodata = retrieve_nodata(dsm_path, nodata)
    
    # Retrieves the number of CPU if the number of available workers is not set
    if not nb_max_workers:
        nb_max_workers = multiprocessing.cpu_count()
    
    preprocessed_dsm_path, quality_mask_path = preprocess_pipeline(dsm_path = dsm_path, 
                                                                   output_dir = output_dir, 
                                                                   nb_max_workers = nb_max_workers, 
                                                                   nodata = nodata, 
                                                                   slope_threshold = slope_threshold, 
                                                                   is_four_connexity = four_connexity,
                                                                   minValidHeight = min_valid_height)
    

    clothSimu = ClothSimulation(max_object_width, 
                                uniform_filter_size, 
                                prevent_unhook_iter,
                                num_outer_iter, 
                                num_inner_iter, 
                                mp_tile_size,
                                output_resolution, 
                                nb_max_workers)

    raw_dtm_path: str = clothSimu.run(preprocessed_dsm_path, 
                                      output_dir, 
                                      nodata)

    postprocess_pipeline(raw_dtm_path =  raw_dtm_path, 
                         output_dir = output_dir,
                         nb_max_workers = nb_max_workers,
                         quality_mask_path =  quality_mask_path, 
                         generate_dhm = generate_dhm, 
                         dsm_path = dsm_path,
                         check_intersection = check_intersection,
                         nodata = nodata)
    
    if not developper_mode:
        # Remove the raw DTM since the postprocess pipeline generates a refined DTM
        preprocessed_dsm_path = os.path.join(output_dir, 'preprocessed_DSM.tif')
        os.remove(raw_dtm_path)
        os.remove(preprocessed_dsm_path)
        
    # global stop_thread
    # stop_thread = True
    # x.join()

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