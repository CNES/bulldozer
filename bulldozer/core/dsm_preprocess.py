# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

"""
    This module is used to preprocess the DSM in order to improve the DTM computation.
"""

import sys
import logging
import rasterio
import os.path
import concurrent.futures
import numpy as np
import scipy.ndimage as ndimage
from rasterio.windows import Window
from rasterio.fill import fillnodata
from tqdm import tqdm
from os import remove

logger = logging.getLogger(__name__)

def build_nodata_mask(dsm : np.ndarray, nodata_value : float) -> [np.ndarray, np.ndarray]:
    """
    This method builds a mask corresponding to nodata values in a given DSM.
    The mask splits the nodata values into two categories:
        - border nodata values
        - inner nodata values (mainly correlation issues in the DSM)
    because both types of nodata areas will no be similarly corrected.

    Args:
        dsm: array containing DSM values.
        nodata_value: nodata value.

    Returns:
        boolean masks list containing the inner and border nodata areas.
    """
    logger.debug("Starting noDataMask building")
    # Get the global nodata mask
    if np.isnan(nodata_value):
        #TODO discuss about that (copy or not?) + very long process
        nodata_value = -32768
        np.nan_to_num(dsm, False, nodata_value)
    nodata_area = (dsm == nodata_value)

    # Connect the groups of nodata elements into labels (non-zero values)
    labeled_array, _ = ndimage.label(nodata_area)

    # Get the label IDs corresponding to nodata regions touching the edges
    border_region_ident = np.unique(
                            np.concatenate((np.unique(labeled_array[0,:]),
                            np.unique(labeled_array[-1,:]),
                            np.unique(labeled_array[:,0]),
                            np.unique(labeled_array[:,-1])), axis = 0))
    # Remove ID = 0 which correspond to background (not nodata areas)
    border_region_ident = border_region_ident[border_region_ident != 0]

    # Retrieve all the border nodata areas and create the corresponding mask 
    border_nodata_mask = np.isin(labeled_array,border_region_ident)
    logger.debug("border_nodata_mask generation : Done")

    # Retrieve all the nodata areas in the input DSM that aren't border areas and create the corresponding mask 
    inner_nodata_mask = np.logical_and(nodata_area == True, np.isin(labeled_array,border_region_ident) != True)
    logger.debug("inner_nodata_mask generation : Done")

    return [border_nodata_mask, inner_nodata_mask]


def compute_disturbance(dsm_path : rasterio.DatasetReader,
                        window : rasterio.windows.Window,
                        slope_treshold : float,
                        disturbed_treshold : int,
                        disturbed_influence_distance : float,
                        dsm_resolution : float) -> (np.ndarray, rasterio.windows.Window) :
    """
    This method computes the disturbance in a DSM window through the horizontal axis.
    It returns the corresping disturbance mask.

    Args:
        dsm_path: path to the input DSM.
        window: coordinates of the concerned window.
        slope_treshold: if the slope is greater than this threshold then we consider it as disturbed variation.
        disturbed_treshold: if the number of successive disturbed pixels along a row is lower than this threshold 
                            then this sequence of pixels is considered as a disturbed area.
        disturbed_influence_distance: if the distance between 2 lists of disturbed cols is lower than this threshold 
                                        expressed in meters then they are merged.
        dsm_resolution: input DSM resolution (in meters).

    Returns:
        mask flagging the disturbed area and its associated window location in the input DSM.
    """
    with rasterio.open(dsm_path, 'r') as dataset:
        dsm_strip = dataset.read(1, window=window).astype(np.float32)
        disturbation_mask = np.zeros(np.shape(dsm_strip), dtype=np.ubyte)
        
        # disturbed_areas will contain row lists and each list will contains k_row lists of disturbed columns
        disturbed_areas = []
        # Loop over each row
        for row in range(dsm_strip.shape[0]):
            disturbed_areas.append([])
            col = 1
            nb_disturbed_areas = 0
            while col < dsm_strip.shape[1]:
                slope = np.abs(dsm_strip[row, col] - dsm_strip[row, col-1])
                if slope >= slope_treshold:
                    nb_disturbed_areas += 1
                    disturbed_areas[row].append([])
                    disturbed_areas[row][nb_disturbed_areas-1].append(col-1)
                    disturbed_areas[row][nb_disturbed_areas-1].append(col)
                    col += 1
                    while col < dsm_strip.shape[1]:
                        slope = np.abs(dsm_strip[row, col] - dsm_strip[row, col-1])
                        if slope >= slope_treshold:
                            disturbed_areas[row][nb_disturbed_areas-1].append(col)
                            col += 1
                        else:
                            col += 1
                            break
                else:
                    col += 1
            
        # merged_disturbed_areas will contain row lists which will contain p_row lists with p_row <= k_row. The p_row lists
        # contain one or more disturbed cols from the k_row lists 
        merged_disturbed_areas = []
        row = 0
        # Retrieve the distance influence in pixel unit
        dist_treshold = disturbed_influence_distance  / dsm_resolution

        # Keep track of the maximum distance of a merged disturbed area
        max_distance = 0

        for row_list in disturbed_areas:
            nb_areas = len(row_list)
            nbMerged = 0
            j = 0
            merged_disturbed_areas.append([])
            while j < nb_areas:
                if len(row_list[j]) >= disturbed_treshold:
                    # We analyse if we found a group of disturbed areas
                    k = j + 1
                    nb_connected = 0
                    while k < nb_areas:
                        if row_list[k][0] - row_list[k-1][len(row_list[k-1])-1] <= dist_treshold:
                            nb_connected += 1
                            k += 1
                        else:
                            break
                    
                    startCol = row_list[j][0]
                    endCol = row_list[j+nb_connected][len(row_list[j+nb_connected])-1]
                    max_distance = max(max_distance, endCol - startCol + 1)
                    merged_disturbed_areas[row].append([])
                    for c in range(startCol, endCol + 1):
                        merged_disturbed_areas[row][nbMerged].append(c)
                    nbMerged += 1
                    j += nb_connected + 1
                else:
                    j += 1

            row += 1
            
        for row in range(disturbation_mask.shape[0]):
            if len(merged_disturbed_areas[row]) > 0:
                for disturbed_area in merged_disturbed_areas[row]:
                    for col in disturbed_area:
                        disturbation_mask[row][col] = 1
        return disturbation_mask, window

###########
#TODO WIP #
###########
def build_disturbance_mask(dsm_path: str,
                        outputMaskPath : str,
                        nb_max_workers : int,
                        disturbance_nodata : np.uint8,
                        slope_treshold: float = 2.0,
                        disturbed_treshold: int = 3,
                        disturbed_influence_distance : float = 5.0,
                        dsm_resolution: float = 0.5):
    """
    This method builds a mask corresponding to the disturbed areas in a given DSM.
    Most of those areas correspond to water or correlation issues during the DSM generation (obstruction, etc.).

    Args:
        dsm_path: path to the input DSM.
        window: coordinates of the concerned window.
        slope_treshold: if the slope is greater than this threshold then we consider it as disturbed variation.
        disturbed_treshold: if the number of successive disturbed pixels along a row is lower than this threshold 
                            then this sequence of pixels is considered as a disturbed area.
        disturbed_influence_distance: if the distance between 2 lists of disturbed cols is lower than this threshold 
                                        expressed in meters then they are merged.

    Returns:
        masks containing the inner and border nodata areas.
    """
    # Determine the number of strip and their height
    with rasterio.open(dsm_path, 'r') as dataset:
        strip_height = dataset.height // nb_max_workers
        strips = [[i*strip_height, (i+1)*strip_height] for i in range(nb_max_workers)]
        strips[-1][1] = dataset.height

        mask_strips = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=nb_max_workers) as executor :
            futures = {executor.submit(compute_disturbance, dsm_path, Window(0,strip[0],dataset.width,strip[1]-strip[0]), 
            slope_treshold, disturbed_treshold, disturbed_influence_distance , dsm_resolution) for strip in strips}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Build Disturbance Mask") :
                mask_strips.append(future.result())

        #TODO remove this part and return a boolean np.array 
        meta = dataset.profile
        meta['nodata'] = 0
        meta['dtype'] = np.ubyte

        with rasterio.open(outputMaskPath, 'w', **meta) as dst:
            for mask in mask_strips :
                dst.write(mask[0], window=mask[1], indexes=1)

###########
# #TODO WIP #
# ###########
# def preprocess(dsm_path : str, 
#                 output_dir : str,
#                 nodata : float,
#                 nb_max_workers : int,
#                 keep_filled_dsm : bool,
#                 keep_corrected_dsm : bool,
#                 slope_treshold, disturbed_treshold,
#                  disturbed_influence_distance, 
#                  dsm_resolution) -> (np.ndarray, list):
#     logger.debug("Starting preprocess")
#     with rasterio.open(dsm_path) as src_dataset:
#         if not nodata:
#             # If nodata is not specified in the config file, retrieve the value from the dsm metadata
#             nodata = src_dataset.nodata
        
#         dsm = src_dataset.read(1)
#         # Generates border and inner nodata mask
#         border_nodata_mask, inner_nodata_mask = build_nodata_mask(dsm, nodata)
#         # Interpolates the inner nodata points (mainly correlation issues in the DSM computation)
#         filled_dsm = fillnodata(dsm, mask=np.invert(inner_nodata_mask))
#         filled_dsm_path = output_dir + 'filled_DSM.tif'
        
#         # Write the filled DSM raster in a new file in order to provide preprocessed DSM to the disturbance detection
#         try:
#             write_dataset(filled_dsm_path, filled_dsm, src_dataset.profile)
#         except (FileNotFoundError, rasterio.RasterioIOError) as e:
#             logger.error('Invalid filled DSM path provided ({})\nError: {}'.format(filled_dsm_path, e))

#         #TODO change disturbance nodata since we want to retrieve boolean mask
#         build_disturbance_mask(filled_dsm_path, output_dir + 'corrected_DSM.tif', nb_max_workers, 1, slope_treshold, 
#                                 disturbed_treshold, disturbed_influence_distance, dsm_resolution)

#         #TODO change disturbance mask
#         #corrected_dsm = rasterio.fill(dsm, mask=np.invert(disturbed_area_mask))

#         # #TODO create preprocessed file
#         # if keep_corrected_dsm:
#         #     corrected_dsm_path = output_dir + "corrected_DSM.tif"
#         #     try:
#         #         write_dataset(corrected_dsm_path, corrected_dsm, src_dataset.profile)
#         #     except (FileNotFoundError, rasterio.RasterioIOError) as e:
#         #         logger.error('Invalid corrected DSM path provided ({})\nError: {}'.format(corrected_dsm_path, e))

#         # # Remove the filled DSM if the option is not True
#         # if not keep_filled_dsm:
#         #     try:
#         #         remove(filled_dsm_path)
#         #     except OSError:
#         #         logger.warning("Error occured during the filled DSM deletion. Path : {}".format(filled_dsm_path))
        
#         src_dataset.close()
#         logger.info("preprocess : Done")
#         #TODO discuss about quality mask: does the allocation make a copy ?
#         # quality_masks = [border_nodata_mask, inner_nodata_mask, disturbed_area_mask]
#         # return corrected_dsm, quality_masks


# if __name__ == "__main__":
#     #TODO remove:
#     with rasterio.open('/home/il/lallemd/scratch/test_preprocess/Toulouse_cropped_2.tif') as t:
#         nodata = t.nodata
#         print(type(t.profile))
#         # preprocess('/home/il/lallemd/scratch/test_preprocess/Toulouse_cropped_2.tif','/home/il/lallemd/scratch/test_preprocess/test_new/', nodata, True, True)
    
#     #TODO discuss about argv vs argparse in this case (standalone call)
#     input_dsm_path = sys.argv[1]
    
#      # input file format check
#     if not (isinstance(input_dsm_path, str) and (input_dsm_path.endswith('.tiff') or input_dsm_path.endswith('.tif'))) :
#         logger.exception('\'path\' argument should be a path to the TIFF config file (here: {})'.format(input_dsm_path))
#         raise ValueError()
#     # input file existence check
#     if not os.path.isfile(input_dsm_path):
#         logger.exception('The input DSM file \'{}\' doesn\'t exist'.format(input_dsm_path))
#         raise FileNotFoundError()
#     #TODO Create a class/module?
#     #TODO which arguments are required for the command line usage
#     # preprocessed_dsm, quality_masks = preprocess() 
#     # try:
#     #     write_dataset(XXX, preprocessed_dsm, XXX)
#     # except (FileNotFoundError, rasterio.RasterioIOError) as err:
#     #     logger.error('Invalid filled DSM path provided ({})\nError: {}'.format(XXX, err))
#     # Merges quality masks
#     #TODO merging
#     #TODO discuss about this
#     # If this module is called in standalone, the user doesn't required the preprocessed DSM, it will produces the filled and corrected DSM
#     # try:
#     #     remove(filled_dsm_path)
#     # except OSError:
#     #     logger.warning("Error occured during the filled DSM deletion. Path : {}".format(filled_dsm_path))