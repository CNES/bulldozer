# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

# Necessary to include the C++ code
cdef extern from "DisturbedAreas.cpp":
    pass

# Declare the class with cdef
cdef extern from "DisturbedAreas.h" namespace "bulldozer":

    cdef cppclass DisturbedAreas:
        
        DisturbedAreas() except +

        void build_disturbance_mask(float *, bool *, unsigned int, unsigned int, float, unsigned int, float, float)