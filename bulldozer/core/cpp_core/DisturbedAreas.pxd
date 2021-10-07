# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.
from libcpp cimport bool

# Necessary to include the C++ code
cdef extern from "src/DisturbedAreas.cpp":
    pass

# Declare the class with cdef
cdef extern from "src/DisturbedAreas.h" namespace "bulldozer":

    cdef cppclass DisturbedAreas:
        
        DisturbedAreas() except +
        DisturbedAreas(bool) except +
        void build_disturbance_mask(float *, unsigned char *, unsigned int, unsigned int, float, float)