# distutils: language = c++

from BulldozerFilters cimport BulldozerFilters
import numpy as np

def npAsContiguousArray(arr):
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr) # Makes a contiguous copy of the numpy array.
    return arr

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methdss
# Python extention type.
cdef class PyBulldozerFilters:

    cdef BulldozerFilters c_bf # Hold a C++ instance wich we're wrapping

    def __cinit__(self):
        self.c_bf = BulldozerFilters()
    
    def run(self, np_dtm, nb_rows, nb_cols, filter_size, no_data):
        cdef float[::1] dtm_memview = npAsContiguousArray(np_dtm.flatten().astype(np.float32))
        cdef float[::1] filtered_dtm_memview = npAsContiguousArray(np.zeros((nb_rows * nb_cols), dtype=np.float32))
        self.c_bf.applyUniformFilter(&dtm_memview[0], &filtered_dtm_memview[0], nb_rows, nb_cols, no_data, filter_size)
        return np.asarray(filtered_dtm_memview)