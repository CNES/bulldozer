# distutils: language = c++
# cython: boundscheck=False

cimport cython
from cython cimport sizeof
from cython cimport floating
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
from scipy import ndimage
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp cimport bool


from libc.math cimport sqrt

import time

import matplotlib.pyplot as plt

np.import_array()

# Compute min/max
cdef (float, float) minmax(vector[float] a):
    cdef:
        float min = np.inf, max = -np.inf
        int i = 0

    for i in range(a.size()):
        if a[i] < min:
            min = a[i]
        if a[i] > max:
            max = a[i]

    return min, max


cdef compute_hist(np.ndarray z, float hist_step):
    cdef:
        float zmin = 0., zmax = 0.
        long bins = 0
        hist = None

    zmin, zmax = minmax(z)
    bins = round((zmax - zmin)/hist_step)
    if bins > 0:
        hist = np.histogram(z, bins)
    
    return hist

# # Method A : Get first 20%
# #-------------------------
# def method_a(x, y, z):
#     coefs = []
#     # 20% of total outline points
#     points_20p = round(len(z)*0.2)
#
#     # 4 points at least are required for linear regression
#     if(points_20p > 3):
#         first_points = np.argsort(z)[:points_20p] 
#         x_fp = x[first_points]
#         y_fp = y[first_points]
#         z_fp = z[first_points]
#
#         # Linear regression
#         Xa = np.stack([x_fp, y_fp, np.ones(len(x_fp))]).T
#         Ya = z_fp.T
#         coefs = np.linalg.solve(np.dot(Xa.T, Xa), np.dot(Xa.T, Ya))
#
#     return coefs


# Method B : Detect first peak and enlarge it up to 20% of outline points
#------------------------------------------------------------------------
cdef method_b(np.ndarray x,
              np.ndarray y,
              np.ndarray z,
              float hist_step,
              float threshold):
    cdef coefs = []
    cdef float mean = 0., first_peak = 0.

    # 20% of total outline points
    # cdef float points_20p = round(len(z)*0.2)
    cdef float points_20p = round(len(z)*0.5)
    cdef np.ndarray first_peak_points
    cdef np.ndarray x_fpp, y_fpp, z_fpp
    cdef np.ndarray Xb, Yb

    # 4 points at least are required for linear regression
    if(points_20p > 3):
        hist = compute_hist(z, hist_step)
        if hist is not None:
            mean = np.mean(hist[0])
            first_peak = hist[1][0]
            for j in range(len(hist[0]) -1):
                if (hist[0][j] > mean and (hist[0][j] - hist[0][j+1]) > threshold ):
                    first_peak = hist[1][j]
                    break
    
            first_peak_points = np.argsort(np.abs(z-first_peak))[:points_20p]
            x_fpp = x[first_peak_points]
            y_fpp = y[first_peak_points]
            z_fpp = z[first_peak_points]
    
            # Linear regression
            Xb = np.stack([x_fpp, y_fpp, np.ones(len(x_fpp))]).T
            Yb = z_fpp.T
            try:
                coefs = np.linalg.solve(np.dot(Xb.T, Xb), np.dot(Xb.T, Yb))
            except:
                coefs = []

    return coefs
#
# # Method C : Get from 2% to 22%
# #------------------------------
# def method_c(x, y, z):
#     coefs = []
#
#     # 02% of total outline points
#     points_02p = round(len(z)*0.02)    
#     # 22% of total outline points
#     points_22p = round(len(z)*0.22)
#
#     # 4 points at least are required for linear regression
#     if(points_22p - points_02p > 3):
#         first_points = np.argsort(z)[points_02p:points_22p] 
#         x_2t20p = x[first_points]
#         y_2t20p = y[first_points]
#         z_2t20p = z[first_points]
#
#         # Linear regression
#         Xc = np.stack([x_2t20p, y_2t20p, np.ones(len(x_2t20p))]).T
#         Yc = z_2t20p.T
#         coefs = np.linalg.solve(np.dot(Xc.T, Xc), np.dot(Xc.T, Yc))
#
#     return coefs

######################################################################
# Label regions
######################################################################
cpdef fill_toground(floating[:,:] dsm,
                  unsigned char[:,:] disturbance_mask,
                  unsigned char[:,:] structure,
                  float nodata_val):

    cdef:
        np.ndarray dilated_mask
        int num_features = 0
        int current_val_inner = 0, current_val_edge = 0
        float current_dsm_val = 0.
        int i, j
        np.ndarray corrected_dsm = np.copy(dsm)

    print("Labelling...")
    start = time.time()
    # 1. Dilatation    
    dilated_mask = ndimage.binary_dilation(disturbance_mask, structure)
    
    # 2 Labellisation
    cdef int[:,:] labeled_array_edges, labeled_array_inner
    labeled_array_edges, num_features = ndimage.label(dilated_mask)
    labeled_array_inner = np.multiply(labeled_array_edges, disturbance_mask)
    labeled_array_edges -= np.asarray(labeled_array_inner)
    end = time.time()
    print("Labellisation : done in {} s".format(end - start))
    print("Number of regions to be corrected : {}".format(num_features))
    
    # Get the edges and inner coordinates for each region
    cdef vector[vector[vector[float]]] edges_index, inner_index
    edges_index.resize(num_features+1)
    inner_index.resize(num_features+1)
    
    
    print("Indexing...")
    
    
    start = time.time()
    cdef int dsm_h = dsm.shape[0], dsm_w = dsm.shape[1]
    for i in range(dsm_h ):
        for j in range(dsm_w):
            current_val_edge = labeled_array_edges[i, j]
            current_val_inner = labeled_array_inner[i, j]
            current_dsm_val = dsm[i,j]
            if current_val_edge != 0 and current_dsm_val != nodata_val:
                edges_index[current_val_edge].push_back([i, j, current_dsm_val])
            elif current_val_inner != 0:
                inner_index[current_val_inner].push_back([i, j])
    end = time.time()
    print("Indexing : done in {} s".format(end - start))
    
    # DSM correction
    print("DSM denoising...")
    start = time.time()
    cdef np.ndarray[float, ndim=1] x_e, y_e, z_e, z_i_c
    cdef np.ndarray[int, ndim=1] x_i, y_i
    cdef hist
    cdef float hist_step = 1.
    cdef float threshold = 3.
    cdef int edge_points_nb = 0, inner_points_nb = 0
    
    for i in range(1, num_features+1):
        # print('Label : {}'.format(i))
        edge_points_nb = edges_index[i].size()
        inner_points_nb = inner_index[i].size()
        x_e = np.empty(edge_points_nb, dtype = np.float32)
        y_e = np.empty(edge_points_nb, dtype = np.float32)
        z_e = np.empty(edge_points_nb, dtype = np.float32)
        x_i = np.empty(inner_points_nb, dtype = np.int32)
        y_i = np.empty(inner_points_nb, dtype = np.int32)
        z_i_c = np.empty(inner_points_nb, dtype = np.float32)
        for j in range(edge_points_nb):
            x_e[j] = edges_index[i][j][0]
            y_e[j] = edges_index[i][j][1]
            z_e[j] = edges_index[i][j][2]
            
        for j in range(inner_points_nb):
            x_i[j] = <int> inner_index[i][j][0]
            y_i[j] = <int> inner_index[i][j][1]
        
        coefs = method_b(x_e, y_e, z_e, hist_step, threshold)

        # Apply correction
        if(len(coefs) == 3):
            corrected_dsm_idx = np.vstack((x_i, y_i, np.ones(len(x_i)))).T
            z_i_c = corrected_dsm_idx.dot(coefs).astype(np.float32)
            corrected_dsm[x_i, y_i] = z_i_c
    
    end = time.time()
    print("DSM Correction : done in {} s".format(end - start))
    
    return corrected_dsm



######################################################################
# Iterative
######################################################################
cpdef iterative_filling(floating[:,:] dsm,
                  unsigned char[:,:] disturbance_mask,
                  float nodata_val,
                  int num_iterations = 50000):

    cdef:
        int dsm_h = dsm.shape[0]
        int dsm_w = dsm.shape[1]
        double diag_weight = 1/sqrt(2)
        int i, j, k, l, m
        int good_neighbor
        bool has_nodata = True
        double total_weight
        int corected, tocorrect
        int nb_pass = 0

    print("Iterative...")
    start = time.time()
    
    # v8 neighborhood indexing 
    # 0 1 2
    # 3   4
    # 5 6 7 
    cdef int goods[8]
    cdef double weights[8]
    
    for k in range(num_iterations):
        tocorrect = 0
        corected = 0
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

                    # print(i, j) 
                    # print (good_neighbor)
                    # print("--------------------------------")
                    # print(dsm[i-1,j-1], dsm[i,j-1], dsm[i+1,j-1])
                    # print(dsm[i-1,j], 0, dsm[i+1,j])
                    # print(dsm[i-1,j+1], dsm[i,j+1], dsm[i+1,j+1])
                
                    # print("--------------------------------")
                    # print(disturbance_mask[i-1,j-1], disturbance_mask[i,j-1], disturbance_mask[i+1,j-1])
                    # print(disturbance_mask[i-1,j], 0, disturbance_mask[i+1,j])
                    # print(disturbance_mask[i-1,j+1], disturbance_mask[i,j+1], disturbance_mask[i+1,j+1])

                    if(good_neighbor >= 3) :
                        
                        corected += 1
                        
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
        
        if has_nodata==False :
            break

        print(nb_pass, "corrected", corected, '/', tocorrect)
        nb_pass += 1     
               
    end = time.time()
    print("DSM Correction : done in {} s".format(end - start))
    
    return dsm




######################################################################
# Label regions
######################################################################
cpdef fill_byidw(floating[:,:] dsm,
                  unsigned char[:,:] disturbance_mask,
                  unsigned char[:,:] structure,
                  float nodata_val):

    cdef:
        np.ndarray dilated_mask
        int num_features = 0
        int current_val_inner = 0, current_val_edge = 0
        float current_dsm_val = 0.
        int i, j

    print("Labelling...")
    start = time.time()
    # 1. Dilatation    
    dilated_mask = ndimage.binary_dilation(disturbance_mask, structure)
    
    # 2 Labellisation
    cdef int[:,:] labeled_array_edges, labeled_array_inner
    labeled_array_edges, num_features = ndimage.label(dilated_mask)
    labeled_array_inner = np.multiply(labeled_array_edges, disturbance_mask)
    labeled_array_edges -= np.asarray(labeled_array_inner)
    end = time.time()
    print("Labellisation : done in {} s".format(end - start))
    print("Number of regions to be corrected : {}".format(num_features))
    
    # Get the edges and inner coordinates for each region
    cdef vector[vector[vector[float]]] edges_index, inner_index
    edges_index.resize(num_features+1)
    inner_index.resize(num_features+1)
    
    
    print("Indexing...")
    
    
    start = time.time()
    cdef int dsm_h = dsm.shape[0], dsm_w = dsm.shape[1]
    for i in range(dsm_h ):
        for j in range(dsm_w):
            current_val_edge = labeled_array_edges[i, j]
            current_val_inner = labeled_array_inner[i, j]
            current_dsm_val = dsm[i,j]
            if current_val_edge != 0 and current_dsm_val != nodata_val:
                edges_index[current_val_edge].push_back([i, j, current_dsm_val])
            elif current_val_inner != 0:
                inner_index[current_val_inner].push_back([i, j])
    end = time.time()
    print("Indexing : done in {} s".format(end - start))
    
    # DSM correction
    print("DSM denoising...")
    start = time.time()
    cdef int x_e, y_e
    cdef int x_i, y_i
    cdef double z_e
    cdef int edge_points_nb = 0, inner_points_nb = 0
    
    cdef double z_interpolated, weight, sum_weights
    
    
    for i in range(1, num_features+1):
        # print('Label : {}'.format(i))
        edge_points_nb = edges_index[i].size()
        inner_points_nb = inner_index[i].size()


            
        for j in range(inner_points_nb):
            
            x_i = <int> inner_index[i][j][0]
            y_i = <int> inner_index[i][j][1]
            
            
            sum_weights = 0.
            z_interpolated = 0
            for j in range(edge_points_nb):
                x_e = <int> edges_index[i][j][0]
                y_e = <int> edges_index[i][j][1]
                z_e = edges_index[i][j][2]
                
                weight = 1. / (sqrt( (x_e-x_i)**2 + (y_e-y_i)**2))
                z_interpolated += weight * z_e
                sum_weights += weight
                
            z_interpolated /= sum_weights
            
            dsm[x_i, y_i] = z_interpolated


    
    end = time.time()
    print("DSM Correction : done in {} s".format(end - start))
    
    return dsm
