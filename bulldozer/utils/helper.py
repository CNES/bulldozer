# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

"""
    This module groups different generic methods used in Bulldozer.
"""

import platform
import psutil
import os
import getpass
import rasterio
import numpy as np
from git import Repo
from git.exc import InvalidGitRepositoryError
from logging import Logger

def init_logger(logger : Logger) -> None:
    """
        This method initiates the log file in order to store the environment state.

        Args:
           logger: logger used to write the environment state.
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
        info['user']=getpass.getuser()
        info['node']=platform.node()
        info['processor']=platform.processor()
        info['ram']=str(round(psutil.virtual_memory().total / (1024 **3)))+" GB"

        # OS info
        info['system']=platform.system()
        info['release']=platform.release()
        info['os_version']=platform.version()
        
        # Message format
        init = ("\n"+"#"*17+"\n#   BULLDOZER   #\n"+"#"*17+"\n# <Git info>\n#\t- branch: {}\n#\t- commit SHA: {}"
                "\n#\n# <Node info>\n#\t - user: {}\n#\t - node: {}\n#\t - processor: {}\n#\t - RAM: {}"
                "\n#\n# <OS info>\n#\t - system: {}\n#\t - release: {}\n#\t - version: {}\n"
                +"#"*17).format(info['branch'], info['commit_sha'], info['user'], info['node'], 
                                info['processor'], info['ram'], info['system'], info['release'], info['os_version'])
        logger.debug(init)

    except Exception as e:
        logger.error("Error occured during logger init: \n" + e)
  

def write_dataset(buffer_path : str, buffer : np.ndarray, profile : rasterio.profiles.Profile) -> None:
    """
        This method allows to write a TIFF file based on the input buffer.

        Args:
            buffer_path: path to the output file.
            buffer: dataset to write.
            profile: destination dataset profile (driver, crs, transform, etc.).
    """
    profile['driver'] = 'GTiff'
    with rasterio.open(buffer_path, 'w', **profile) as dst_dataset:
        dst_dataset.write(buffer, 1)
        dst_dataset.close()

def npAsContiguousArray(arr : np.array) -> np.array:
    """
    This method checks that the input array is contiguous. 
    If not, returns the contiguous version of the input numpy array.

    Args:
        arr: input array.

    Returns:
        contiguous array usable in C++.
    """
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr
        
