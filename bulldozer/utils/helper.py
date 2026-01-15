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
This module groups different generic methods used in Bulldozer.
"""

from typing import Any, Dict

import numpy as np
from rasterio import Affine


def np_as_contiguous_array(arr: np.ndarray) -> np.ndarray:
    """
    This method checks that the input array is contiguous.
    If not, returns the contiguous version of the input numpy array.

    Args:
        arr: input array.

    Returns:
        contiguous array usable in C++.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def downsample_profile(profile: Dict[str, Any], factor: float) -> Dict[str, Any]:
    """Downsample image profile by affine translation."""
    transform = profile["transform"]

    newprofile = profile.copy()
    dst_transform = Affine.translation(transform[2], transform[5]) * Affine.scale(
        transform[0] * factor, transform[4] * factor
    )

    newprofile.update(
        {
            "transform": dst_transform,
        }
    )

    return newprofile


def ubyte_profile(input_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Convert input profile to ubyte profile."""
    output_profile = input_profile.copy()
    output_profile["dtype"] = np.ubyte
    output_profile["nodata"] = None

    return output_profile
