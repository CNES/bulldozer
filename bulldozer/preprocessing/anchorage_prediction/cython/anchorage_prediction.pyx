# distutils: language = c++
# coding: utf8

import numpy as np

from bulldozer.utils.helper import npAsContiguousArray
import bulldozer.eoscale.manager as eom
import rasterio

# Begin PXD

# Necessary to include the C++ code
cdef extern from "c_anchorage_prediction.cpp":
    pass

# Declare the class with cdef
cdef extern from "c_anchorage_prediction.h" namespace "bulldoproto":

    void predict_anchorage(float * dsm,
                           float nodata,
                           unsigned char * regular_mask,
                           unsigned char * anchorage_mask,
                           unsigned int nb_rows,
                           unsigned int nb_cols,
                           float max_object_size)


# End PXD

cdef class PyAnchoragePredictor:

    def __cinit__(self):
        """
        Default constructor.
        """
        pass

    def predict(self, 
                dsm : np.array,
                regular_mask : np.array,
                anchors_mask: np.array,
                nodata: float,
                max_object_size: float) -> str:
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
        nb_rows: int = dsm.shape[1]
        nb_cols: int = dsm.shape[2]
        
        cdef float[::1] dsm_memview = dsm.ravel()
        cdef unsigned char[::1] regular_mask_memview = regular_mask.ravel()
        # Output mask that will be filled by the C++ part
        cdef unsigned char [::1] anchors_memview = anchors_mask.ravel()

        # Compute stats
        predict_anchorage(&dsm_memview[0],
                          nodata,
                          &regular_mask_memview[0], 
                          &anchors_memview[0], 
                          nb_rows, 
                          nb_cols,
                          max_object_size)




