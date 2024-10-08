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
    This module is used to retrieve the bulldozer parameters from an YAML configuration file.
"""

from yaml import safe_load,YAMLError
import os.path
import logging
from bulldozer.utils.bulldozer_logger import BulldozerLogger

class ConfigParser(object):
    """
        Configuration file parser. Used to read the bulldozer parameters.
    """

    def __init__(self, verbose : bool = False) -> None:
        """
            Parser constructor.

            Args:
                verbose (default=False): increase output verbosity if true.
        """
        if verbose:
            self.level=logging.DEBUG
        else:
            self.level=logging.INFO


    def read(self, path : str) -> dict:
        """
            This method returns the dict containing the bulldozer parameters extracted from 
            the input YAML configuration file.

            Args:
                path: path to the configuration file (expected YAML file).

            Returns:
                cfg: configuration parameters for bulldozer.

            Raises:
                ValueError: if bad input path is provided.
                FileNotFoundError: if the input file doesn't exist.
                YAMLError: if an error occured while reading the yaml file.
        """
        # input file format check
        if not (isinstance(path, str) and (path.endswith('.yaml') or path.endswith('.yml'))) :
            BulldozerLogger.log('\'path\' argument should be a path to the YAML config file (here: {})'.format(path), logging.ERROR)
            raise ValueError()
        # input file existence check
        if not os.path.isfile(path):
            BulldozerLogger.log('The input config file \'{}\' doesn\'t exist'.format(path), logging.ERROR)
            raise FileNotFoundError()
        
        if self.level == logging.DEBUG:
            BulldozerLogger.log('Check input config file => Passed', self.level)

        with open(path, 'r') as stream:
            try:
                cfg = safe_load(stream)
                if self.level == logging.DEBUG:
                    BulldozerLogger.log('Retrieved data: {}'.format(cfg), self.level)
            except YAMLError as e:
                BulldozerLogger.log('Exception occured while reading the configuration file: {}\nException: {}'.format(path, str(e)), logging.ERROR)
                raise YAMLError(str(e))
        return cfg