# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.
from libcpp cimport bool

# Necessary to include the C++ code
cdef extern from "src/BorderNodata.cpp":
    pass

# Declare the class with cdef
cdef extern from "src/BorderNodata.h" namespace "bulldozer":

    cdef cppclass BorderNodata:
        
        BorderNodata() except +
        void buildBorderNodataMask(float *, unsigned char *, unsigned int, unsigned int, float)