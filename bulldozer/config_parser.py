# coding: utf-8

"""
    This module is used to retrieve the bulldozer parameters from an YAML configuration file.
"""

from yaml import safe_load,YAMLError
import os.path
import logging

logger = logging.getLogger(__name__)

class ConfigParser(object):
    """
        Configuration file parser. Used to read the bulldozer parameters.
    """

    def __init__(self, verbose : bool = False) -> None:
        """
            Parser constructor

            @params:
                verbose (bool=False): increase output verbosity if true
        """
        if verbose:
            self.level=logging.DEBUG
        else:
            self.level=logging.INFO


    def read(self, path : str)->dict:
        """
            This method returns the dict containing the bulldozer parameters extracted from 
            the input YAML configuration file.

            @params:
                path (str): path to the configuration file (expected YAML file)

            @returns:
                cfg (dict): configuration parameters for bulldozer

            @raises:
                ValueError: when bad input path is provided

                FileNotFoundError: when the input file doesn't exist
        """
        # input file format check
        if not (isinstance(path, str) and (path.endswith('.yaml') or path.endswith('.yml'))) :
            logger.exception('\'path\' argument should be a path to the YAML config file (here: {})'.format(path))
            raise ValueError()
        # input file existence check
        if not os.path.isfile(path):
            logger.exception('The input config file \'{}\' doesn\'t exist'.format(path))
            raise FileNotFoundError()
        
        if self.level == logging.DEBUG:
            logger.debug('Check input config file => Passed')

        with open(path, 'r') as stream:
            try:
                cfg = safe_load(stream)
                if self.level == logging.DEBUG:
                    logger.debug('Retrieved data: {}'.format(cfg))
            except YAMLError as e:
                logger.exception('Exception occured while reading the configuration file: {}\nException: {}'.format(path, str(e)))
                raise YAMLError()
        return cfg