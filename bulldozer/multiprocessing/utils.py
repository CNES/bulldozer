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
This module contains some utils function for bulldozer processing.
"""

import os
from collections import namedtuple
from typing import Any, Dict

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


def write(data: np.ndarray, img_path: str, target_profile: Dict[str, Any], binary: bool = False) -> None:
    """Save in file with rasterio."""
    if binary:
        with rasterio.open(img_path, "w", nbits=1, **target_profile) as out_dataset:
            out_dataset.write(data, indexes=1)
    else:
        with rasterio.open(img_path, "w", **target_profile) as out_dataset:
            out_dataset.write(data, indexes=1)


def write_window(
    img_buffer: np.ndarray, img_path: str, target_profile: Dict[str, Any], tile: MpTile, binary: bool = False
) -> None:
    """Update window in file."""
    width = tile.end_x - tile.start_x + 1
    height = tile.end_y - tile.start_y + 1
    mode = "r+" if os.path.isfile(img_path) else "w"
    if binary:
        with rasterio.open(img_path, mode, nbits=1, **target_profile) as out_dataset:
            out_dataset.write(img_buffer, window=Window(tile.start_x, tile.start_y, width, height), indexes=1)
    else:
        with rasterio.open(img_path, mode, **target_profile) as out_dataset:
            out_dataset.write(img_buffer, window=Window(tile.start_x, tile.start_y, width, height), indexes=1)
