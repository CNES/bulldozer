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
This module is used to manage the execution context of bulldozer
"""

import multiprocessing as mp
import os
import shutil
from typing import Any

import numpy as np

from bulldozer.eomultiprocessing.utils import write


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
        self.in_memory = not extract_param(params, "intermediate_write")
        self.context = extract_param(params, "mp_context")
        self.tile_mode = tile_mode

        # If the target output directories does not exist, creates it
        self.debug_key = "dev"
        self.tmp_key = "tmp"
        output_dir = extract_param(params, "output_dir")
        self.out_directories = self.init_out_directories(output_dir)

        self.pool = None
        self.lock = None
        self._manager = None

    def __enter__(self):  # type: ignore
        if self.nb_workers > 1:  # multiprocessing
            if self.context is None:
                self.context = mp.get_start_method()
            if self.context not in mp.get_all_start_methods():
                raise ValueError(
                    f"The multiprocessing context '{self.context}' is not supported by your OS. "
                    f"Please choose one among {mp.get_all_start_methods()}"
                )
            ctx = mp.get_context(self.context)
            self.pool = ctx.Pool(processes=self.nb_workers)
            if not self.in_memory:
                self._manager = mp.Manager()
                self.lock = self._manager.Lock()
        else:
            self.in_memory = True  # always in memory if no multiprocessing

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        # Close pool of process
        if self.nb_workers > 1:
            self.pool.close()  # type: ignore
            self.pool.join()  # type: ignore
            if not self.in_memory:
                self._manager.shutdown()  # type: ignore
        # Delete tmp folder
        if self.tmp_key in self.out_directories:
            shutil.rmtree(self.out_directories[self.tmp_key])

    # Public methods

    def init_out_directories(self, output_dir: str) -> dict[str, str]:
        """Generates a new out_directories dictionary"""
        out_directories = {"out": output_dir, "mask": os.path.join(output_dir, "masks")}
        os.makedirs(out_directories["mask"], exist_ok=True)
        # Create temporary folder
        if not self.in_memory:
            out_directories[self.tmp_key] = os.path.join(output_dir, "tmp")
            os.makedirs(out_directories[self.tmp_key], exist_ok=True)
        # Create debug folder
        if self.dev_mode:
            out_directories[self.debug_key] = os.path.join(output_dir, "developer")
            os.makedirs(out_directories[self.debug_key], exist_ok=True)

        return out_directories

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

    def write_tif(
        self,
        data: np.ndarray,
        filename: str,
        target_profile: dict[str, Any],
        key: str | None = None,
    ) -> str:
        """Get output path and write the data"""
        if key is None:
            key = self.debug_key if self.dev_mode else self.tmp_key
        full_path = self.get_path(filename, key=key)
        write(data, full_path, target_profile)
        return full_path

    def move_tif(self, input_path: str, filename: str, key: str | None = None) -> str:
        """Get output path and move the input file"""
        if key is None:
            key = self.debug_key if self.dev_mode else self.tmp_key
        full_path = self.get_path(filename, key=key)
        shutil.move(input_path, full_path)
        return full_path
