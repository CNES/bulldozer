# coding: utf-8

"""
    This module is used to retrieve the bulldozer parameters from an YAML configuration file.
"""

from yaml import safe_load,YAMLError
import os.path
import logging

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
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)


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
            raise ValueError('\'path\' argument should be a path to the YAML config file (here: {})'.format(path))
        # input file existence check
        if not os.path.isfile(path):
            raise FileNotFoundError('The input config file \'{}\' doesn\'t exist'.format(path))
        
        logging.debug('{} - [read] : Check input config file => Passed'.format(__class__.__name__))

        with open(path, 'r') as stream:
            try:
                cfg = safe_load(stream)
                logging.debug('{0} - [read] : Retrieved data: {1}'.format(__class__.__name__, cfg))
            except YAMLError:
                raise YAMLError('Exception occured while reading the configuration file: ' + path)
        return cfg