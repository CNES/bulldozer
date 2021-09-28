import sys
import logging
from bulldozer.core.dsm_preprocess import preprocess
from bulldozer.core.dtm_extraction import run
from bulldozer.core.dtm_postprocess import postprocess
from bulldozer.utils.config_parser import ConfigParser
from bulldozer.utils.helper import init_logger
logger = logging.getLogger(__name__)
# No data value constant used in bulldozer

class Pipeline:
    def __init__(self, config_path : str, verbosity = False):
        """

        """
        init_logger(logger)
        parser = ConfigParser(verbosity)
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
        postprocess(dtm_path, cfg['outputDir'], quality_mask_path, 
                    cfg['dhm'], cfg['dsmPath'])
    

if __name__ == "__main__":
    config_path = sys.argv[1]
    Pipeline(config_path)