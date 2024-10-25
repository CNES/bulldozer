# distutils: language = c++
import numpy as np
from bulldozer.utils.helper import npAsContiguousArray

# pxd section
cdef extern from "c_springforce.cpp":
    pass

# Declare the class with cdef
cdef extern from "c_springforce.h" namespace "bulldozer":

    cdef cppclass BulldozerFilters:
        
        BulldozerFilters() except +

        void applyUniformFilter(float *, float *, unsigned int, unsigned int, float, unsigned int)
# end pxd section

cdef class PyBulldozerFilters:

    cdef BulldozerFilters c_bf # Hold a C++ instance wich we're wrapping

    def __cinit__(self):
        self.c_bf = BulldozerFilters()
    
    def run(self, np_dtm, nb_rows, nb_cols, filter_size, no_data):
        cdef float[::1] dtm_memview = npAsContiguousArray(np_dtm.flatten().astype(np.float32))
        cdef float[::1] filtered_dtm_memview = npAsContiguousArray(np.zeros((nb_rows * nb_cols), dtype=np.float32))
        self.c_bf.applyUniformFilter(&dtm_memview[0], &filtered_dtm_memview[0], nb_rows, nb_cols, no_data, filter_size)
        return np.asarray(filtered_dtm_memview)