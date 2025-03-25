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
cdef extern from "c_fill.cpp":
    pass

# Declare the class with cdef
cdef extern from "c_fill.h" namespace "bulldozer":

        void iterativeFilling(float *,
                              unsigned char *,
                              int,
                              int,
                              float,
                              int) 


# End PXD

cdef class PyFill:

    def __cinit__(self) -> None:
        """
        Default constructor.
        """
        pass

    def iterative_filling(self, 
                          dsm_strip : np.array, 
                          nodata_value : float,
                          nb_it : int,
                          border_nodata_strip : np.array = None) -> np.array:
        """ 
        This method fills the DSM

        Args:
            dsm_strip: part of the DSM analyzed.
            disturbance_strip: part of the disturbance mask analyzed.
            nodata_value: nodata value.
        Return:
            Filled DSM
        """
        cdef float[::1] dsm_memview = npAsContiguousArray(dsm_strip.ravel().astype(np.float32))
        cdef unsigned char[::1] border_nodata_mask_memview 
        cdef unsigned char* border_nodata_mask_ptr = NULL  # Initialize as NULL

        if border_nodata_strip is not None: # For the level 0
            border_nodata_mask_memview = npAsContiguousArray(border_nodata_strip.ravel().astype(np.uint8))
            border_nodata_mask_ptr = &border_nodata_mask_memview[0]

        # Iterative Filling
        iterativeFilling(&dsm_memview[0], border_nodata_mask_ptr, dsm_strip.shape[0], dsm_strip.shape[1], nodata_value, nb_it)
        # Reshape the output DSM. From array to matrix corresponding to the input DSM strip shape
        return np.asarray(dsm_memview).reshape(dsm_strip.shape[0], dsm_strip.shape[1]).astype(np.float32)
    
