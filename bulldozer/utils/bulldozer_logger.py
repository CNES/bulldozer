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
This module aims to centralize the use of the logger in Bulldozer.
"""
from __future__ import annotations
import sys
import os
import getpass
import platform
import time
import psutil
import multiprocessing
import logging
import logging.config
from bulldozer._version import __version__

class BulldozerLogger:
    """
        Bulldozer logger singleton. Only used in the full pipeline mode (not for the standalone calls).
    """
    __instance = None

    @staticmethod
    def getInstance(logger_file_path: str = None) -> BulldozerLogger:
        """
            Return the logger or create it if the instance does not exist.

            Args:
                logger_file_path: path to the output logfile.

            Returns:
                the Bulldozer logger.
        """
        if BulldozerLogger().__instance is None :

            # Create the Logger
            # Sub folders will inherits from the logger configuration, hence
            # we need to give the root package directory name of Bulldozer
            logger = logging.getLogger("bulldozer")
            logger.setLevel(logging.DEBUG)

            # create file handler which logs even debug messages
            fh = logging.FileHandler(filename=logger_file_path, mode='w')
            fh.setLevel(logging.DEBUG)

            LOG_FORMAT = '%(asctime)s [%(levelname)s] %(module)s - %(funcName)s (line %(lineno)d): %(message)s'
            logger_formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
            fh.setFormatter(logger_formatter)

            logger.addHandler(fh)

            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)

            STREAM_FORMAT = '%(asctime)s [%(levelname)s] - %(message)s'
            logger_formatter = logging.Formatter(STREAM_FORMAT, datefmt="%H:%M:%S")
            sh.setFormatter(logger_formatter)

            logger.addHandler(sh)
            BulldozerLogger.__instance = logger

            BulldozerLogger.init_logger()

        return BulldozerLogger.__instance


    @staticmethod
    def log(msg : str, level : any) -> None:
        """
            Bulldozer logger log function.
            The following logging levels are used:
                DEBUG
                INFO
                WARNING
                ERROR

            Args:
                msg: log message.
                level: crticity level.
        """
        if BulldozerLogger.__instance is not None :
           if level == logging.DEBUG:
                BulldozerLogger.__instance.debug(msg)
           if level == logging.INFO:
                BulldozerLogger.__instance.info(msg)
           if level == logging.WARNING:
                BulldozerLogger.__instance.warning(msg)
           if level == logging.ERROR:
                BulldozerLogger.__instance.error(msg)
    
    @staticmethod
    def init_logger() -> None:
        """
            This method store the environment state in the logfile.
        """
        info={}
        try:  
            # Node info
            try:
                info['user'] = getpass.getuser()
            except:
                info['user'] = 'unknown'
            try:
                info['node'] = platform.node()
            except:
                info['node'] = 'unknown'
            info['processor'] = platform.processor()
            info['cpu_count'] = multiprocessing.cpu_count()
            info['ram'] = str(round(psutil.virtual_memory().total / (1024 **3)))+" GB"
            
            # OS info
            info['system'] = platform.system()
            info['release'] = platform.release()
            info['os_version'] = platform.version()
            
            # Message format
            init = ("\n"+"#"*17+"\n#   BULLDOZER   #\n"+"#"*17+"\n# <Bulldozer info>\n#\t- version: {}"+
                    "\n#\n# <Node info>\n#\t - user: {}\n#\t - node: {}\n#\t - processor: {}\n#\t - CPU count: {}\n#\t - RAM: {}"
                    "\n#\n# <OS info>\n#\t - system: {}\n#\t - release: {}\n#\t - version: {}\n"
                    +"#"*17).format(__version__, info['user'], info['node'], info['processor'], info['cpu_count'], info['ram'], 
                    info['system'], info['release'], info['os_version'])
            BulldozerLogger.log(init, logging.DEBUG)

        except Exception as e:
            BulldozerLogger.log("Error occured during logger init: \n" + str(e), logging.DEBUG)

class Runtime:
    """
    This class is used as decorator to monitor the runtime.
    """
    
    def __init__(self, function) -> None:
        """
            Decorator constructor.

            Args:
                function: the function to call.
        """
        self.function = function

    def __call__(self, *args, **kwargs) -> Any:
        """
            Log the start and end of the function with the associated runtime.

            Args:
                args: function arguments.
                kwargs: function key arguments.

            Returns:
                the function output.
        """
        func_start = time.perf_counter()
        BulldozerLogger.log("{}: Starting...".format(self.function.__name__), logging.DEBUG)
        # Function run
        result = self.function(*args, **kwargs)
        func_end = time.perf_counter()
        BulldozerLogger.log("{}: Done (Runtime: {}s)".format(self.function.__name__, round(func_end-func_start,2)), logging.INFO)
        return result