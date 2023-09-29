# distutils: language = c++
# coding: utf8

import numpy as np

from bulldozer.utils.helper import npAsContiguousArray

# Begin PXD

# Necessary to include the C++ code
cdef extern from "c_stats.cpp":
    pass

# Declare the class with cdef
cdef extern from "c_stats.h" namespace "bulldoproto":

    void compute_stats(float * dsm,
                       float * stats,
                       unsigned int nb_rows,
                       unsigned int nb_cols,
                       float nodata)


# End PXD

cdef class PyStats:

    def __cinit__(self):
        """
        Default constructor.
        """
        pass

    def computeStats(self, 
                     dsm_strip : np.array, 
                     nodata : float) -> list:
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
        cdef float[::1] dsm_memview = npAsContiguousArray(dsm_strip.flatten().astype(np.float32))
        # Ouput mask that will be filled by the C++ part
        cdef float[::1] stats_memview = npAsContiguousArray(np.zeros((2), dtype=np.float32))
        # Compute stats
        compute_stats(&dsm_memview[0], &stats_memview[0], dsm_strip.shape[0], dsm_strip.shape[1], nodata)
        # Reshape the output mask. From array to matrix corresponding to the input DSM strip shape
        npstats = np.asarray(stats_memview).astype(np.float32)

        return [ npstats[0], npstats[1] ]





