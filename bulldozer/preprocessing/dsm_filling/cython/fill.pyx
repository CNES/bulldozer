# distutils: language = c++
# coding: utf8
#
# Copyright (c) 2022-2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of Bulldozer
# (see https://github.com/CNES/bulldozer).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cimport cython
from cython cimport floating
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
from libcpp cimport bool


from libc.math cimport sqrt

import time

np.import_array()

######################################################################
# Iterative
######################################################################
cpdef iterative_filling(floating[:,:] dsm,
                  unsigned char[:,:] disturbance_mask,
                  float nodata_val,
                  int num_iterations = 10000):

    cdef:
        int dsm_h = dsm.shape[0]
        int dsm_w = dsm.shape[1]
        double diag_weight = 1/sqrt(2)
        int i, j, k, l, m
        int good_neighbor
        bool has_nodata = True
        double total_weight
        int corrected, tocorrect
        int nb_pass = 0

    #TODO: remove switch from print to logger
    print("Iterative filling...")
    #start = time.time()
    
    # v8 neighborhood indexing 
    # 0 1 2
    # 3   4
    # 5 6 7 
    cdef int goods[8]
    cdef double weights[8]
    
    for k in range(num_iterations):
        tocorrect = 0
        corrected = 0
        has_nodata = False
        
        for i in range(1, dsm_h-1):
            for j in range(1, dsm_w-1):
                if disturbance_mask[i,j] == 1 :
                    
                    has_nodata = True
                    tocorrect += 1
                    
                    goods[0] = disturbance_mask[i-1, j-1]==0 and dsm[i-1, j-1] != nodata_val
                    goods[1] = disturbance_mask[i,   j-1]==0 and dsm[i,   j-1] != nodata_val
                    goods[2] = disturbance_mask[i+1, j-1]==0 and dsm[i+1, j-1] != nodata_val
                    
                    goods[3] = disturbance_mask[i-1, j  ]==0 and dsm[i-1, j  ] != nodata_val
                    goods[4] = disturbance_mask[i+1, j  ]==0 and dsm[i+1, j  ] != nodata_val

                    goods[5] = disturbance_mask[i-1, j+1]==0 and dsm[i-1, j+1] != nodata_val
                    goods[6] = disturbance_mask[i,   j+1]==0 and dsm[i,   j+1] != nodata_val
                    goods[7] = disturbance_mask[i+1, j+1]==0 and dsm[i+1, j+1] != nodata_val                 
                    
                    good_neighbor = goods[0] + goods[1] + goods[2] + goods[3] + goods[4] + goods[5] + goods[6] + goods[7]

                    if(good_neighbor >= 3) :
                        
                        corrected += 1
                        
                        weights[0] = goods[0] * diag_weight
                        weights[1] = goods[1]
                        weights[2] = goods[2] * diag_weight
                        weights[3] = goods[3]
                        weights[4] = goods[4]
                        weights[5] = goods[5] * diag_weight
                        weights[6] = goods[6]
                        weights[7] = goods[7] * diag_weight 
                        
                        dsm[i,j] = weights[0] * dsm[i-1,j-1] + weights[1] * dsm[i,j-1]  + weights[2] * dsm[i+1,j-1] \
                                + weights[3] * dsm[i-1,j]                              + weights[4] * dsm[i+1,j] \
                                + weights[5] * dsm[i-1,j+1] + weights[6] * dsm[i,j+1]  + weights[7] * dsm[i+1,j+1]
                                            
                        total_weight = weights[0] + weights[1] + weights[2] + weights[3] + weights[4] + weights[5] + weights[6] + weights[7]
                        
                        dsm[i,j] /= total_weight
                        disturbance_mask[i,j] = 2
        
        for l in prange(1, dsm_h-1, nogil=True):
            for m in range(1, dsm_w-1):
                if disturbance_mask[l,m] == 2 :
                    disturbance_mask[l,m] = 0 
        
        if has_nodata==False or corrected==0 :
            break

        #print(nb_pass, "corrected", corrected, '/', tocorrect)
        nb_pass += 1     
               
    end = time.time()
    #TODO change print to logger and extract .cpp and .h files from .pyx file
    #print("DSM Correction : done in {} s ({} passes performed)".format(end - start, nb_pass))
    
    return dsm
