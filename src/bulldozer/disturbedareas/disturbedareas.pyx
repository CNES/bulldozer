# distutils: language = c++

# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

from libcpp cimport bool
import numpy as np
from bulldozer.utils.helper import npAsContiguousArray

# pxd section
cdef extern from "c_disturbedareas.cpp":
    pass

cdef extern from "c_disturbedareas.h" namespace "bulldozer":

    cdef cppclass DisturbedAreas:
        
        DisturbedAreas() except +
        DisturbedAreas(bool) except +
        void build_disturbance_mask(float *, unsigned char *, unsigned int, unsigned int, float, float)
# end pxd section

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class PyDisturbedAreas:

    cdef DisturbedAreas disturbed_areas # Hold a C++ instance wich we're wrapping

    def __cinit__(self, is_four_connexity=True):
        self.disturbed_areas = DisturbedAreas(is_four_connexity)
    
    def build_disturbance_mask(self, 
                               dsm_strip : np.array, 
                               slope_treshold : float,
                               no_data_value : float):
        """
        This method detects the disturbed areas along horizontal axis in the input DSM window.
        For the disturbed areas along vertical axis, transpose the input DSM window.

        Args:
            dsm_strip: part of the DSM analyzed.
            slope_treshold: if the slope is greater than this threshold then we consider it as disturbed variation.
            is_four_connexity: number of evaluated axis. Vertical and horizontal if true else vertical, horizontal and diagonals.
            no_data_value: nodata value used in the input DSM.
        Returns:
            mask of the disturbed areas in the input DSM window.
        """
        cdef float[::1] dsm_memview = npAsContiguousArray(dsm_strip.flatten().astype(np.float32))
        # Ouput mask that will be filled by the C++ part
        cdef unsigned char[::1] disturbance_mask_memview = npAsContiguousArray(np.zeros((dsm_strip.shape[0] * dsm_strip.shape[1]), dtype=np.uint8))
        # Disturbance detection
        self.disturbed_areas.build_disturbance_mask(&dsm_memview[0], &disturbance_mask_memview[0], dsm_strip.shape[0], dsm_strip.shape[1], slope_treshold, no_data_value)
        # Reshape the output mask. From array to matrix corresponding to the input DSM strip shape
        return np.asarray(disturbance_mask_memview).reshape(dsm_strip.shape[0], dsm_strip.shape[1]).astype(np.bool)