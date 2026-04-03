#!/usr/bin/env python
#
# Copyright (c) 2022-2026 Centre National d'Etudes Spatiales (CNES).
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

import getpass
import logging
import logging.config
import multiprocessing
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from bulldozer._version import __version__

# Global logger instance
logger = logging.getLogger("bulldozer")


def setup_logger(output_dir: str) -> logging.Logger:
    """
    Set up the Bulldozer logger that logs to both console and a logfile.

    Args:
        output_dir: path to the output directory.

    Returns:
        the Bulldozer logger.
    """
    output_path = Path(output_dir)
    if not output_path.exists() or not output_path.is_dir():
        raise ValueError(f"Output directory '{output_dir}' doesn't exist or isn't a directory.")

    logger.setLevel(logging.DEBUG)

    # Clean handlers if logger already exists
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    stream_format = "%(asctime)s [%(levelname)s] - %(message)s"
    console_formatter = logging.Formatter(stream_format, datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file_path = output_path / f"bulldozer_{datetime.now().strftime('%Y-%m-%dT%Hh%Mm%S')}.log"
    try:
        file_handler = logging.FileHandler(log_file_path)
    except (PermissionError, OSError) as error:
        logger.error(f"Failed to create log file '{log_file_path}': {error}")
        raise PermissionError(f"Failed to create log file: {error}") from error
    file_handler.setLevel(logging.DEBUG)
    file_format = "%(asctime)s [%(levelname)s] %(module)s - %(funcName)s (line %(lineno)d): %(message)s"
    file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%dT%H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def init_logfile() -> None:
    """
    This method stores the environment state in the logfile.
    """
    env_info: dict[str, Any] = {}
    try:
        # Node info
        try:
            env_info["user"] = getpass.getuser()
        except (getpass.GetPassWarning, OSError):
            logger.debug("Failed to retrieve user information")
            env_info["user"] = "unknown"
        env_info["node"] = "unknown" if platform.node() == "" else platform.node()
        env_info["processor"] = platform.processor()
        env_info["cpu_count"] = multiprocessing.cpu_count()
        try:
            env_info["ram"] = f"{round(psutil.virtual_memory().total / (1024**3))} GB"
        except NameError:
            logger.debug("psutil not installed; RAM information unavailable")
            env_info["ram"] = "unknown"

        # OS info
        env_info["system"] = platform.system()
        env_info["release"] = platform.release()
        env_info["os_version"] = platform.version()

        # Message format
        init_message = f"""
{"#" * 17}
#   BULLDOZER   #
{"#" * 17}
# <Bulldozer info>
#\t- version: {__version__}
#
# <Node info>
#\t- user: {env_info["user"]}
#\t- node: {env_info["node"]}
#\t- processor: {env_info["processor"]}
#\t- CPU count: {env_info["cpu_count"]}
#\t- RAM: {env_info["ram"]}
#
# <OS info>
#\t- system: {env_info["system"]}
#\t- release: {env_info["release"]}
#\t- version: {env_info["os_version"]}
{"#" * 17}"""
        logger.debug(init_message)
    except Exception as error:
        logger.debug(f"Error occurred during logfile init: \n{error}")


class Runtime:
    """
    This class is used as decorator to monitor the runtime.
    """

    def __init__(self, function: Any) -> None:
        """
        Decorator constructor.

        Args:
            function: the function to call.
        """
        self.function = function

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Log the start and end of the function with the associated runtime.

        Args:
            args: function arguments.
            kwargs: function key arguments.

        Returns:
            the function output.
        """
        func_start = time.perf_counter()
        logger.debug(f"{self.function.__name__}: Starting...")
        # Function run
        result = self.function(*args, **kwargs)
        func_end = time.perf_counter()
        logger.info(f"{self.function.__name__}: Done (Runtime: {round(func_end - func_start, 2)}s)")

        return result
