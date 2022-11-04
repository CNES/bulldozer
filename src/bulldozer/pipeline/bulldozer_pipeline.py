# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

"""
    This module is used to postprocess the DTM in order to improve its quality. It required a DTM generated from Bulldozer.
"""
import os
import sys
from shutil import copy
from datetime import datetime
from sys import stdout
import argparse
import argcomplete
from bulldozer.dtm_extraction.dtm_extraction import ClothSimulation
from bulldozer.preprocessing.dsm_preprocess import run as run_preprocessing
from bulldozer.postprocessing.dtm_postprocess import run as run_postprocessing
from bulldozer.utils.config_parser import ConfigParser
from bulldozer.utils.logging_helper import BulldozerLogger

__version__ = "0.1.0"


def dsm_to_dtm(cfg: dict) -> None:
    """
        Pipeline constructor.
    """
    bulldoLogger = BulldozerLogger.getInstance(loggerFilePath=os.path.join(cfg['outputDir'], "trace.log"))

    starting_time = datetime.now()
    bulldoLogger.info("Starting time: " + starting_time.strftime("%Y-%m-%d %H:%M:%S"))
    run_preprocessing(cfg['dsmPath'], 
                      cfg['outputDir'], 
                      cfg['nbMaxWorkers'], 
                      cfg['createFilledDsm'], 
                      cfg['noData'], 
                      cfg['slopeThreshold'], 
                      cfg['fourConnexity'],
                      cfg['minValidHeight'])

    dsm_path = os.path.join(cfg['outputDir'], 'filled_DSM.tif')
    clothSimu = ClothSimulation(cfg['maxObjectWidth'], 
                                cfg['uniformFilterSize'], 
                                cfg['preventUnhookIter'],
                                cfg['numOuterIter'], 
                                cfg['numInnerIter'], 
                                cfg['mpTileSize'], 
                                cfg['sequential'], 
                                cfg['nbMaxWorkers'])

    clothSimu.run(dsm_path, cfg['outputDir'])
    
    quality_mask_path = os.path.join(cfg['outputDir'], 'quality_mask.tif')
    dtm_path = os.path.join(cfg['outputDir'], 'DTM.tif')

    run_postprocessing(dtm_path, cfg['outputDir'], quality_mask_path, 
                cfg['createDhm'], cfg['dsmPath'])

    bulldoLogger.info("Ending time: {} (Runtime: {}s)"
                      .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                      datetime.now()-starting_time))


def get_parser():
    """
    ArgumentParser for bulldozer
    :param None
    :return parser
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


def main():
    """
        Call bulldozer main
    """
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # The first parameter should be the path to the configuration file
    config_path = args.conf

    # Configuration file format check
    if not (config_path.endswith('.yaml') or config_path.endswith('.yml')) :
        raise ValueError('Expected yaml configuration file')

    # Configuration file existence check
    if not os.path.isfile(config_path):
        raise FileNotFoundError('Expected yaml configuration file')

    # Retrieves all the settings
    parser = ConfigParser(False)
    cfg = parser.read(config_path)

    dsm_to_dtm(cfg)

if __name__ == "__main__":
    main()
