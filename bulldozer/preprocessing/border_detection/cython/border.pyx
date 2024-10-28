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
cdef extern from "c_border.cpp":
    pass

# Declare the class with cdef
cdef extern from "c_border.h" namespace "bulldozer":
        
        void buildBorderNodataMask(float *, 
                                   unsigned char *, 
                                   unsigned int, 
                                   unsigned int, 
                                   float)

# End PXD

cdef class PyBorderNodata:


    def __cinit__(self) -> None:
        """
        Default constructor.
        """
        pass


    def build_border_nodata_mask(self, 
                               dsm_strip : np.array, 
                               nodata_value : float,
                               is_transposed: bool) -> np.array:
        """
        This method detects the border nodata areas in the input DSM window.
        For the border nodata along vertical axis, transpose the input DSM window.

        Args:
            dsm_strip: part of the DSM analyzed.
            nodata_value: nodata value used in the input DSM.
            is_transposed: boolean flag indicating if the computation is vertical or horizontal.
        Returns:
            mask of the border nodata areas in the input DSM window.
        """
        if is_transposed:
            first_index = 0
            second_index = 1
        else:
            first_index = 1
            second_index = 2

        cdef float[::1] dsm_memview = npAsContiguousArray(dsm_strip.ravel().astype(np.float32))
        # Ouput mask that will be filled by the C++ part
        cdef unsigned char[::1] border_nodata_mask_memview = npAsContiguousArray(np.zeros((dsm_strip.shape[first_index] * dsm_strip.shape[second_index]), dtype=np.uint8))
        # Border nodata detection
        buildBorderNodataMask(&dsm_memview[0], &border_nodata_mask_memview[0], dsm_strip.shape[first_index], dsm_strip.shape[second_index], nodata_value)
        # Reshape the output mask. From array to matrix corresponding to the input DSM strip shape
        return np.asarray(border_nodata_mask_memview).reshape(dsm_strip.shape[first_index], dsm_strip.shape[second_index]).astype(np.ubyte)