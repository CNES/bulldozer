# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

"""
    This module is used to postprocess the DTM in order to improve its quality. It required a DTM generated from Bulldozer.
"""
import os
import sys
import logging
import logging.config
from shutil import copy
from datetime import datetime
from sys import stdout
from bulldozer.core.dsm_preprocess import preprocess
from bulldozer.core.dtm_extraction import run
from bulldozer.core.dsm_preprocess import PreProcess
from bulldozer.core.dtm_postprocess import PostProcess
from bulldozer.utils.config_parser import ConfigParser
from bulldozer.utils.helper import init_logger

logging.config.fileConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/logging.ini"), 
                          disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class Pipeline(object):
    """
        Bulldozer main pipeline.
    """
    
    def __init__(self, config_path : str) -> None:
        """
            Pipeline constructor.
        """
        init_logger(logger)
        starting_time = datetime.now()
        logger.info("Starting time: " + starting_time.strftime("%Y-%m-%d %H:%M:%S"))
        parser = ConfigParser(False)
        # Retrieves all the settings
        cfg = parser.read(config_path)

        preprocess(cfg['dsmPath'], cfg['outputDir'], cfg['nbMaxWorkers'], 
                   cfg['createFilledDsm'], cfg['noData'], 
                   cfg['slopeThreshold'], cfg['fourConnexity'])

        dsm_path = cfg['outputDir'] + 'preprocessed_DSM.tif'
        dtm_path = cfg['outputDir'] + 'dtm.tif'
        tmp_dir = cfg['outputDir'] + 'tmp/'
        run(dsm_path, dtm_path, tmp_dir, cfg['maxObjectWidth'], 
            cfg['uniformFilterSize'], cfg['preventUnhookIter'],
            cfg['numOuterIter'], cfg['numInnerIter'], cfg['mpTileSize'], 
            cfg['sequential'], cfg['nbMaxWorkers'] )
        
        quality_mask_path = cfg['outputDir'] + 'quality_mask.tif'
        postprocess = PostProcess()
        postprocess.run(dtm_path, cfg['outputDir'], quality_mask_path, 
                    cfg['dhm'], cfg['dsmPath'])

        logger.info("Ending time: {} (Runtime: {}s)"
                    .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                    datetime.now()-starting_time))
        
        if cfg['keepLog']:
            try:
                copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                    "../logs/bulldozer_logfile.log"),
                    os.path.dirname(os.path.abspath(cfg['outputDir'] + "bulldozer_logfile.log")))
            except Exception as e:
                logger.warning("Error while writting the logfile: " + str(e), exc_info=True)

if __name__ == "__main__":
    assert(len(sys.argv)>=2)

    # The first parameter should be the path to the configuration file
    config_path = sys.argv[1]

    # Configuration file format check
    if not (config_path.endswith('.yaml') or config_path.endswith('.yml')) :
        logger.exception('\'config_path\' argument should be a path to a Yaml file (here: {})'.format(config_path))
        raise ValueError('Expected yaml configuration file')

    # Configuration file existence check
    if not os.path.isfile(config_path):
        logger.exception('The input configuration file \'{}\' doesn\'t exist'.format(config_path))
        raise FileNotFoundError('Expected yaml configuration file')
    logger.debug('Configuration file existence check: Done')

    Pipeline(config_path)
