# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

"""
    This module is used to postprocess the DTM in order to improve its quality. It required a DTM generated from Bulldozer.
"""
import os
import sys
#import logging
#import logging.config
from shutil import copy
from datetime import datetime
from sys import stdout
import argparse
import argcomplete
from bulldozer.dtm_extraction.dtm_extraction import ClothSimulation
from bulldozer.preprocessing.dsm_preprocess import run as run_preprocessing
from bulldozer.postprocessing.dtm_postprocess import PostProcess
from bulldozer.utils.config_parser import ConfigParser
#from bulldozer.utils.helper import init_logger

__version__ = "0.1.0"

# logging.config.fileConfig("logging.ini", 
#                           disable_existing_loggers=False)
# logger = logging.getLogger(__name__)

def dsm_to_dtm(config_path : str) -> None:
    """
        Pipeline constructor.
    """
    #init_logger(logger)
    starting_time = datetime.now()
    #logger.info("Starting time: " + starting_time.strftime("%Y-%m-%d %H:%M:%S"))
    parser = ConfigParser(False)
    # Retrieves all the settings
    cfg = parser.read(config_path)
    run_preprocessing(cfg['dsmPath'], cfg['outputDir'], cfg['nbMaxWorkers'], 
                cfg['createFilledDsm'], cfg['noData'], 
                cfg['slopeThreshold'], cfg['fourConnexity'])

    dsm_path = cfg['outputDir'] + 'filled_DSM.tif'
    # Warning now it is not possible to be robust to nodata because
    # of the upsample step.
    # We will have to use the ancrage drape.
    #dsm_path = cfg['outputDir'] + 'preprocessed_DSM.tif'
    clothSimu = ClothSimulation(cfg['maxObjectWidth'], 
                                cfg['uniformFilterSize'], 
                                cfg['preventUnhookIter'],
                                cfg['numOuterIter'], 
                                cfg['numInnerIter'], 
                                cfg['mpTileSize'], 
                                cfg['sequential'], 
                                cfg['nbMaxWorkers'])

    clothSimu.run(dsm_path, cfg['outputDir'])
    
    quality_mask_path = cfg['outputDir'] + 'quality_mask.tif'
    dtm_path = cfg['outputDir'] + 'DTM.tif'
    postprocess = PostProcess()
    postprocess.run(dtm_path, cfg['outputDir'], quality_mask_path, 
                cfg['dhm'], cfg['dsmPath'])

    # logger.info("Ending time: {} (Runtime: {}s)"
    #             .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
    #             datetime.now()-starting_time))
    
    # if cfg['keepLog']:
    #     try:
    #         copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
    #             "../logs/bulldozer_logfile.log"),
    #             os.path.dirname(os.path.abspath(cfg['outputDir'] + "bulldozer_logfile.log")))
    #     except Exception as e:

    #         logger.warning("Error while writting the logfile: " + str(e), exc_info=True)


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
        #logger.exception('\'config_path\' argument should be a path to a Yaml file (here: {})'.format(config_path))
        raise ValueError('Expected yaml configuration file')

    # Configuration file existence check
    if not os.path.isfile(config_path):
        #logger.exception('The input configuration file \'{}\' doesn\'t exist'.format(config_path))
        raise FileNotFoundError('Expected yaml configuration file')
    #logger.debug('Configuration file existence check: Done')

    dsm_to_dtm(config_path)

if __name__ == "__main__":
    main()
