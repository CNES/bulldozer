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

__version__ = "0.1.0"

@Runtime
def dsm_to_dtm(cfg: dict) -> None:
    """
        Pipeline orchestrator.
        
        Args:
            cfg: values of the config file.

    """
    logger = BulldozerLogger.getInstance(logger_file_path=os.path.join(cfg['outputDir'], "trace_" + datetime.now().strftime("%d.%m.%Y_%H:%M:%S") +".log"))

    # Retrieves the nodata value from the config file or the DSM metadata
    cfg['noData'] = retrieve_nodata(cfg['dsmPath'], float(cfg['noData']))
    
    # Retrieves the number of CPU if the number of available workers is not set
    if not cfg['nbMaxWorkers']:
        cfg['nbMaxWorkers'] = multiprocessing.cpu_count()
    
    preprocessed_dsm_path, quality_mask_path = preprocess_pipeline(cfg['dsmPath'], 
                                                                   cfg['outputDir'], 
                                                                   cfg['nbMaxWorkers'], 
                                                                   cfg['noData'], 
                                                                   cfg['slopeThreshold'], 
                                                                   cfg['fourConnexity'],
                                                                   cfg['minValidHeight'])

    clothSimu = ClothSimulation(cfg['maxObjectWidth'], 
                                cfg['uniformFilterSize'], 
                                cfg['preventUnhookIter'],
                                cfg['numOuterIter'], 
                                cfg['numInnerIter'], 
                                cfg['mpTileSize'],
                                cfg['outputResolution'], 
                                cfg['nbMaxWorkers'])

    raw_dtm_path: str = clothSimu.run(preprocessed_dsm_path, 
                                      cfg['outputDir'], 
                                      cfg['noData'])

    # raw_dtm_path = os.path.join(cfg['outputDir'], "raw_DTM.tif")
    # quality_mask_path = os.path.join(cfg['outputDir'], "quality_mask.tif")

    postprocess_pipeline(raw_dtm_path =  raw_dtm_path, 
                         output_dir = cfg['outputDir'],
                         nb_max_workers = cfg['nbMaxWorkers'], 
                         quality_mask_path =  quality_mask_path, 
                         generate_dhm = cfg['generateDhm'], 
                         dsm_path = cfg['dsmPath'],
                         nodata = cfg['noData'])
    
    # if not cfg['developperMode']:
    #     # Remove the raw DTM since the postprocess pipeline generates a refined DTM
    #     preprocessed_dsm_path = os.path.join(cfg['outputDir'], 'preprocessed_DSM.tif')
    #     os.remove(raw_dtm_path)
    #     os.remove(preprocessed_dsm_path)

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


def main() -> None:
    """
        Call bulldozer main pipeline.
    """
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # The first parameter should be the path to the configuration file
    config_path = args.conf

    # Configuration file format check
    if not (config_path.endswith('.yaml') or config_path.endswith('.yml')) :
        raise ValueError('Expected yaml configuration file: \'config_path\' argument should be a path to a Yaml file (here: {})'.format(config_path))

    # Configuration file existence check
    if not os.path.isfile(config_path):
        raise FileNotFoundError('The input configuration file \'{}\' doesn\'t exist'.format(config_path))
    
    # Retrieves all the settings
    parser = ConfigParser(False)
    cfg = parser.read(config_path)

    dsm_to_dtm(cfg)

if __name__ == "__main__":
    main()
