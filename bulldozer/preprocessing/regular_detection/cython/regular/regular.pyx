# distutils: language = c++
# coding: utf8

import numpy as np

from bulldozer.utils.helper import npAsContiguousArray

# Begin PXD

# Necessary to include the C++ code
cdef extern from "c_regular.cpp":
    pass

# Declare the class with cdef
cdef extern from "c_regular.h" namespace "bulldoproto":

        void buildRegularMask(float * ,
							  unsigned char *,
							  unsigned int,
							  unsigned int,
							  float, 
                              float)


# End PXD

cdef class PyRegularAreas:

    def __cinit__(self):
        """
        Default constructor.
        """
        pass

    def buildRegularMask(self, 
                         dsm_strip : np.array, 
                         slope_threshold : float, 
                         no_data_value : float):
        """ 
        This method detects regular areas using average slope from the 8 neighbors
        Args:
            dsm_strip: part of the DSM analyzed
            slope_threshold: if the average slope is lower than this threshold then we consider it as a regular area
        Return:
            mask of the regular / disturbed areas
        """
        cdef float[::1] dsm_memview = npAsContiguousArray(dsm_strip.ravel().astype(np.float32))
        # Ouput mask that will be filled by the C++ part
        cdef unsigned char[::1] regular_mask_memview = npAsContiguousArray(np.zeros((dsm_strip.shape[0] * dsm_strip.shape[1]), dtype=np.uint8))
        # Regular detection
        buildRegularMask(&dsm_memview[0], &regular_mask_memview[0], dsm_strip.shape[0], dsm_strip.shape[1], slope_threshold, no_data_value)
        # Reshape the output mask. From array to matrix corresponding to the input DSM strip shape
        return np.asarray(regular_mask_memview).reshape(dsm_strip.shape[0], dsm_strip.shape[1]).astype(np.uint8)
    
   