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

import rasterio
import numpy
from collections import namedtuple

MpTile = namedtuple('MpTile', ["start_x", "start_y", "end_x", "end_y", "top_margin", "right_margin", "left_margin", "bottom_margin"])

JSON_NONE: str = "none"


def rasterio_profile_to_dict(profile: rasterio.DatasetReader.profile) -> dict:
    """
        Convert a rasterio profile to a serializable python dictionnary
        needed for storing in a chunk of memory that will be shared among
        processes 
    """
    metadata = dict()
    for key, value in profile.items():
        if key == "crs":
            metadata['crs'] = profile['crs'].to_wkt()
        elif key == "transform":
            metadata['transform_1'] = profile['transform'][0]
            metadata['transform_2'] = profile['transform'][1]
            metadata['transform_3'] = profile['transform'][2]
            metadata['transform_4'] = profile['transform'][3]
            metadata['transform_5'] = profile['transform'][4]
            metadata['transform_6'] = profile['transform'][5]
        elif key == "nodata":
            if value is None:
                metadata[key] = JSON_NONE
            else:
                metadata[key] = value
        elif key == "dtype":
            if not isinstance(value, str):
                metadata[key] = numpy.dtype(value).name
            else:
                metadata[key] = value
        else:
            metadata[key] = value
    return metadata


def dict_to_rasterio_profile(metadata: dict) -> rasterio.DatasetReader.profile :
    """
        Convert a serializable dictionnary to a rasterio profile
    """
    rasterio_profile = {}
    for key, value in metadata.items():
        if key == "crs":
            rasterio_profile["crs"] = rasterio.crs.CRS.from_string(metadata['crs'])
        elif key == "transform_1":
            rasterio_profile['transform'] = rasterio.Affine(metadata['transform_1'], 
                                                            metadata['transform_2'], 
                                                            metadata['transform_3'], 
                                                            metadata['transform_4'], 
                                                            metadata['transform_5'], 
                                                            metadata['transform_6'])
        elif key.startswith("transform"):
            continue
        elif key == "nodata":
            if value == JSON_NONE:
                rasterio_profile[key] = None
            else:
                rasterio_profile[key] = value
        else:
            rasterio_profile[key] = value

    return rasterio_profile