# distutils: language = c++
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
import numpy as np
from bulldozer.utils.helper import npAsContiguousArray

# Begin PXD

# Necessary to include the C++ code
cdef extern from "c_bordernodata.cpp":
    pass

# Declare the class with cdef
cdef extern from "c_bordernodata.h" namespace "bulldozer":

    cdef cppclass BorderNodata:
        
        BorderNodata() except +
        void buildBorderNodataMask(float *, unsigned char *, unsigned int, unsigned int, float)

# End PXD

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class PyBorderNodata:

    cdef BorderNodata border_nodata # Hold a C++ instance wich we're wrapping

    def __cinit__(self):
        """
        Default constructor.
        """
        self.border_nodata = BorderNodata()


    def build_border_nodata_mask(self, 
                               dsm_strip : np.array, 
                               no_data_value : float):
        """
        This method detects the border nodata areas in the input DSM window.
        For the border nodata along vertical axis, transpose the input DSM window.

        Args:
            dsm_strip: part of the DSM analyzed.
            no_data_value: nodata value used in the input DSM.
        
        Returns:
            mask of the border nodata areas in the input DSM window.
        """

        cdef float[::1] dsm_memview = npAsContiguousArray(dsm_strip.flatten().astype(np.float32))
        # Ouput mask that will be filled by the C++ part
        cdef unsigned char[::1] border_nodata_mask_memview = npAsContiguousArray(np.zeros((dsm_strip.shape[0] * dsm_strip.shape[1]), dtype=np.uint8))
        # Border nodata detection
        self.border_nodata.buildBorderNodataMask(&dsm_memview[0], &border_nodata_mask_memview[0], dsm_strip.shape[0], dsm_strip.shape[1], no_data_value)
        # Reshape the output mask. From array to matrix corresponding to the input DSM strip shape
        return np.asarray(border_nodata_mask_memview).reshape(dsm_strip.shape[0], dsm_strip.shape[1]).astype(np.ubyte)