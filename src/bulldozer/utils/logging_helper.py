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
This module aims to centralize the use of the logger in Bullodzer.
"""
import sys
import os
import getpass
import platform
import psutil
import multiprocessing
import logging
import logging.config
from git import Repo
from git.exc import InvalidGitRepositoryError

class BulldozerLogger:
    """
        Bulldozer logger singleton. Only used in the full pipeline mode (not for the standalone calls).
    """
    __instance = None

    @staticmethod
    def getInstance(logger_file_path: str = None):
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

            LOG_FORMAT = '%(asctime)s [%(levelname)s]: %(message)s'
            logger_formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d,%H:%M:%S")
            fh.setFormatter(logger_formatter)

            logger.addHandler(fh)

            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)

            STREAM_FORMAT = '=%(asctime)s [%(levelname)s] %(module)s - %(message)s'
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
            We use the following levels:
                DEBUG
                INFO
                WARNING

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
    
    @staticmethod
    def init_logger() -> None:
        """
            This method store the environment state in the logfile.
        """
        info={}
        try:
            # Git info
            try :
                repo = Repo(search_parent_directories=True)
                info['commit_sha'] = repo.head.object.hexsha
                info['branch'] = repo.active_branch
            except InvalidGitRepositoryError as e:
                info['commit_sha'] = "No git repo found ({})".format(e)
                info['branch'] = "No git repo found ({})".format(e)
                
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
            init = ("\n"+"#"*17+"\n#   BULLDOZER   #\n"+"#"*17+"\n# <Git info>\n#\t- branch: {}\n#\t- commit SHA: {}"
                    "\n#\n# <Node info>\n#\t - user: {}\n#\t - node: {}\n#\t - processor: {}\n#\t - CPU count: {}\n#\t - RAM: {}"
                    "\n#\n# <OS info>\n#\t - system: {}\n#\t - release: {}\n#\t - version: {}\n"
                    +"#"*17).format(info['branch'], info['commit_sha'], info['user'], info['node'], 
                                    info['processor'], info['cpu_count'], info['ram'], info['system'], info['release'], info['os_version'])
            BulldozerLogger.log(init, logging.DEBUG)

        except Exception as e:
            BulldozerLogger.log("Error occured during logger init: \n" + str(e), logging.DEBUG)