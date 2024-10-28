# distutils: language = c++
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

import numpy as np
from bulldozer.utils.helper import npAsContiguousArray

# Begin PXD

# Necessary to include the C++ code
cdef extern from "c_regular.cpp":
    pass

# Declare the class with cdef
cdef extern from "c_regular.h" namespace "bulldozer":

        void buildRegularMask(float * ,
							  unsigned char *,
							  unsigned int,
							  unsigned int,
							  float, 
                              float)


# End PXD

cdef class PyRegularAreas:

    def __cinit__(self) -> None:
        """
        Default constructor.
        """
        pass

    def build_regular_mask(self, 
                           dsm_strip : np.array, 
                           slope_threshold : float, 
                           nodata_value : float) -> np.array:
        """ 
        This method detects regular areas using average slope from the 8 neighbors.

        Args:
            dsm_strip: part of the DSM analyzed.
            slope_threshold: if the average slope is lower than this threshold then it's considered as a regular area.
            nodata_value: nodata value.
        Return:
            mask of the regular / disturbed areas.
        """
        cdef float[::1] dsm_memview = npAsContiguousArray(dsm_strip.ravel().astype(np.float32))
        # Ouput mask that will be filled by the C++ part
        cdef unsigned char[::1] regular_mask_memview = npAsContiguousArray(np.zeros((dsm_strip.shape[0] * dsm_strip.shape[1]), dtype=np.uint8))
        # Regular detection
        buildRegularMask(&dsm_memview[0], &regular_mask_memview[0], dsm_strip.shape[0], dsm_strip.shape[1], slope_threshold, nodata_value)
        # Reshape the output mask. From array to matrix corresponding to the input DSM strip shape
        return np.asarray(regular_mask_memview).reshape(dsm_strip.shape[0], dsm_strip.shape[1]).astype(np.uint8)
    
   