# distutils: language = c++
# coding: utf8

import numpy as np

from bulldozer.utils.helper import npAsContiguousArray

# Begin PXD

# Necessary to include the C++ code
cdef extern from "c_uncertain.cpp":
    pass

# Declare the class with cdef
cdef extern from "c_uncertain.h" namespace "bulldoproto":

    void prefill_uncertain(float * dsm,
                            unsigned char * uncertain_mask,
                            unsigned char * regular_mask,
                            float * prefilled_dsm,
                            float * uncertain_map,
                            unsigned int rows,
                            unsigned int cols,
                            unsigned int search_radius,
                            float max_slope_percent,
                            float dsm_resolution)


# End PXD

cdef class PyUncertain:

    def __cinit__(self):
        """
        Default constructor.
        """
        pass

    def prefillUncertain(self, 
                         dsm : np.array,
                         uncertain_mask: np.ndarray,
                         regular_mask: np.ndarray,
                         search_radius: int,
                         max_slope_percent: float,
                         dsm_resolution: float) -> list:
        """ 
        This method compute min and max valid height given a nodata value
        The motivation of a cython code is to not modify the input array
        with np.nan values in Python.
        Args:
            dsm_strip: part of the DSM analyzed
            nodata: nodata value (usually -32768) of the input image
        Return:
            an array of two values corresponding to min and max value
        """
        cdef float[::1] dsm_memview = npAsContiguousArray(dsm.flatten().astype(np.float32))
        cdef unsigned char[::1] mask_memview = npAsContiguousArray(uncertain_mask.flatten().astype(np.uint8))
        cdef unsigned char[::1] regular_memview = npAsContiguousArray(regular_mask.flatten().astype(np.uint8))
        # Ouput mask that will be filled by the C++ part
        cdef float [::1] prefill_dsm_memview = npAsContiguousArray(np.zeros((np.prod(dsm.shape)), dtype=np.float32))
        cdef float [::1] uncertain_map_memview = npAsContiguousArray(np.zeros((np.prod(dsm.shape)), dtype=np.float32))
        # Prefill the dsm and compute uncertainties
        prefill_uncertain(&dsm_memview[0], 
                          &mask_memview[0],
                          &regular_memview[0], 
                          &prefill_dsm_memview[0],
                          &uncertain_map_memview[0],
                          dsm.shape[0], dsm.shape[1],
                          search_radius,
                          max_slope_percent,
                          dsm_resolution)

        # Reshape the output mask. From array to matrix corresponding to the input DSM shape
        np_prefill_dsm = np.asarray(prefill_dsm_memview).reshape(dsm.shape[0], dsm.shape[1])
        np_uncertain_map = np.asarray(uncertain_map_memview).reshape(dsm.shape[0], dsm.shape[1])

        return [ np_prefill_dsm, np_uncertain_map ]





