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
This module contains some utils function for bulldozer processing.
"""

import os
from collections import namedtuple
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

MpTile = namedtuple(
    "MpTile",
    [
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "top_margin",
        "right_margin",
        "left_margin",
        "bottom_margin",
        "height",
        "width",
        "height_margin",
        "width_margin",
    ],
)

MAX_CACHE = 64  # Mb, cache size for rasterio reading and writing operations, by default 5% of the usable physical RAM


def write(data: np.ndarray, img_path: str, target_profile: dict[str, Any]) -> None:
    """Save in file with rasterio."""
    with rasterio.Env(GDAL_CACHEMAX=MAX_CACHE):
        with rasterio.open(img_path, "w", **target_profile) as out_dataset:
            out_dataset.write(data, indexes=1)


def write_window(img_buffer: np.ndarray, img_path: str, target_profile: dict[str, Any], tile: MpTile) -> None:
    """Update window in file."""
    width = tile.end_x - tile.start_x + 1
    height = tile.end_y - tile.start_y + 1
    mode = "r+" if os.path.isfile(img_path) else "w"
    with rasterio.Env(GDAL_CACHEMAX=MAX_CACHE):
        with rasterio.open(img_path, mode, **target_profile) as out_dataset:
            out_dataset.write(img_buffer, window=Window(tile.start_x, tile.start_y, width, height), indexes=1)


def read(img_path: str) -> np.ndarray:
    """Read a file with minimal GDAL cache"""
    with rasterio.Env(GDAL_CACHEMAX=MAX_CACHE):
        with rasterio.open(img_path, "r") as src:
            data = src.read(1)

    return data


def read_and_get_profile(img_path: str) -> tuple[np.ndarray, dict]:
    """Read a file with minimal GDAL cache and also get its profile"""
    with rasterio.Env(GDAL_CACHEMAX=MAX_CACHE):
        with rasterio.open(img_path, "r") as src:
            input_profile = src.profile.copy()
            data = src.read(1)

    return data, input_profile


def read_window(img_path: str, tile: MpTile) -> np.ndarray:
    """Read a window from a file with minimal GDAL cache"""
    col_off = tile.start_x - tile.left_margin
    row_off = tile.start_y - tile.top_margin
    with rasterio.Env(GDAL_CACHEMAX=MAX_CACHE):
        with rasterio.open(img_path, "r") as src:
            data = src.read(1, window=Window(col_off, row_off, tile.width_margin, tile.height_margin))

    return data
