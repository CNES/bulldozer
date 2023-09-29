# distutils: language = c++
# coding: utf8

import numpy as np

from bulldozer.utils.helper import npAsContiguousArray

# Begin PXD

# Necessary to include the C++ code
cdef extern from "c_hist.cpp":
    pass

# Declare the class with cdef
cdef extern from "c_hist.h" namespace "bulldoproto":

    void compute_hist(float * dsm,
					  unsigned int * hist,
					  float min_z,
					  float bin_width,
					  unsigned int nb_bins,
                      unsigned int nb_rows,
                      unsigned int nb_cols,
                      float nodata)


# End PXD

cdef class PyHist:

    def __cinit__(self):
        """
        Default constructor.
        """
        pass

    def computeHist(self, 
                    dsm_strip : np.array,
                    dsm_min_z: float,
                    nb_bins: int,
                    bin_width: float, 
                    nodata : float) -> np.ndarray:
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
        cdef unsigned int [::1] hist_memview = npAsContiguousArray(np.zeros((nb_bins), dtype=np.uint32))
        # Compute stats
        compute_hist(&dsm_memview[0], &hist_memview[0], dsm_min_z, bin_width, nb_bins, dsm_strip.shape[0], dsm_strip.shape[1], nodata)
        # Reshape the output mask. From array to matrix corresponding to the input DSM strip shape
        nphist = np.asarray(hist_memview).astype(np.uint32)

        return nphist





