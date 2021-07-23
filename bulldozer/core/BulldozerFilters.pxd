# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer library
#
#    author:     Pierre Lassalle (DNO/OT/IS)
#    contact:    pierre.lassalle@cnes.fr
#
# All rights reserved.

# Necessary to include the C++ code
cdef extern from "cpp_core/BulldozerFilters.cpp":
    pass

# Declare the class with cdef
cdef extern from "cpp_core/BulldozerFilters.h" namespace "bulldozer":

    cdef cppclass BulldozerFilters:
        
        BulldozerFilters() except +

        void applyUniformFilter(float *, float *, unsigned int, unsigned int, float, unsigned int)