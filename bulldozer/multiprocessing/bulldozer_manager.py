#!/usr/bin/env python
# coding: utf8
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
This module is used to manage the execution context of bulldozer
"""

import multiprocessing
import os
import shutil
from typing import Any


def extract_param(params: dict, key: str) -> Any:
    if key not in params.keys():
        raise ValueError(f"Input parameters must contain the key '{key}'")
    return params[key]


class BulldozerContextManager:
    """
    Bulldozer Context Manager to manage processing with multiprocessing.
    """

    def __init__(self, params: dict, tile_mode: bool = False):
        self.nb_workers = extract_param(params, "nb_max_workers")
        self.dev_mode = extract_param(params, "developer_mode")
        self.in_memory = extract_param(params, "method") == "mem"
        self.context = extract_param(params, "mp_context")
        self.tile_mode = tile_mode

        # If the target output directories does not exist, creates it
        output_dir = extract_param(params, "output_dir")
        self.out_directories = {"out": output_dir, "mask": os.path.join(output_dir, "masks")}
        os.makedirs(self.out_directories["mask"], exist_ok=True)
        if not self.in_memory:
            self.out_directories["tmp"] = os.path.join(output_dir, "tmp")
            os.makedirs(self.out_directories["tmp"], exist_ok=True)
        if self.dev_mode:
            self.out_directories["dev"] = os.path.join(output_dir, "developer")
            os.makedirs(self.out_directories["dev"], exist_ok=True)

        self.pool = None
        self.lock = None
        self._manager = None

    def __enter__(self):  # type: ignore
        if self.nb_workers > 1:  # multiprocessing
            if self.context not in ["spawn", "fork", "forkserver"]:
                raise ValueError(f"The multiprocessing context must be spawn, fork or forkserver, not '{self.context}'")
            ctx = multiprocessing.get_context(self.context)
            self.pool = ctx.Pool(processes=self.nb_workers)
            if not self.in_memory:
                self._manager = multiprocessing.Manager()
                self.lock = self._manager.Lock()
        else:
            self.in_memory = True  # always in memory if no multiprocessing

        return self

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore
        # Close pool of process
        if self.nb_workers > 1:
            self.pool.close()
            self.pool.join()
            if not self.in_memory:
                self._manager.shutdown()
        # Delete tmp folder
        if "tmp" in self.out_directories:
            shutil.rmtree(self.out_directories["tmp"])

    # Public methods

    def get_path(self, filename: str, key: str = "out") -> str:
        """Get full path knowing the key of the output folder"""
        if key in self.out_directories:
            img_path = os.path.join(self.out_directories[key], filename)
        else:
            raise KeyError(
                f"The key '{key}' is not available. Please choose between {list(self.out_directories.keys())}"
            )

        return img_path

    def add_out_directory(self, folder: str, key: str = "out") -> None:
        """Add an output directory in self.out_directories"""
        if key in self.out_directories:
            self.out_directories[folder] = os.path.join(self.out_directories[key], folder)
            os.makedirs(self.out_directories[folder], exist_ok=True)
        else:
            raise KeyError(
                f"The key '{key}' is not available. Please choose between {list(self.out_directories.keys())}"
            )
