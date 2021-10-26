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
from bulldozer.core.cpp_core import DisturbedAreas as da
from rasterio.windows import Window
from rasterio.fill import fillnodata
from bulldozer.utils.helper import write_dataset
from tqdm import tqdm
from os import remove
from bulldozer.core.cpp_core import BulldozerIDW as bi

logger = logging.getLogger(__name__)
# No data value constant used in bulldozer
NO_DATA_VALUE = -32768
class PreProcess(object):
    """
        Bulldozer pre processing set of functions.
    """

    def __init__(self) -> None:
            pass
    
        
    def build_inner_nodata_mask(self, dsm : np.ndarray) -> np.ndarray:
        """
        This method builds a mask corresponding to inner nodata values in a given DSM.
        (mainly correlation issues in the DSM)

        Args:
            dsm: array containing DSM values.

        Returns:
            boolean mask corresponding to the inner nodata areas.
        """
        logger.debug("Starting inner_nodata_mask building")
        # Get the global nodata mask
        nodata_area = (dsm == NO_DATA_VALUE)

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
        logger.debug("border_nodata_mask generation: Done")

        # Retrieve all the nodata areas in the input DSM that aren't border areas and create the corresponding mask 
        inner_nodata_mask = np.logical_and(nodata_area == True, np.isin(labeled_array,border_region_ident) != True)
        logger.debug("inner_nodata_mask generation: Done")

        return [border_nodata_mask, inner_nodata_mask]


    def compute_disturbance(self, 
                            dsm_path : rasterio.DatasetReader,
                            window : rasterio.windows.Window,
                            slope_treshold : float,
                            is_four_connexity : bool) -> (np.ndarray, rasterio.windows.Window) :
        """
        This method computes the disturbance in a DSM window through the horizontal axis.
        It returns the corresping disturbance mask.

        Args:
            dsm_path: path to the input DSM.
            window: coordinates of the concerned window.
            slope_treshold: if the slope is greater than this threshold then we consider it as disturbed variation.
            is_four_connexity: number of evaluated axis. 
                            Vertical and horizontal if true else vertical, horizontal and diagonals.

        Returns:
            mask flagging the disturbed area and its associated window location in the input DSM.
        """
        logger.debug("Starting disturbed area analysis. Window strip: {}".format(window))
        with rasterio.open(dsm_path, 'r') as dataset:
            dsm_strip = dataset.read(1, window=window).astype(np.float32)
            disturbed_areas = da.PyDisturbedAreas(is_four_connexity)
            disturbance_mask = disturbed_areas.build_disturbance_mask(dsm_strip, slope_treshold, NO_DATA_VALUE).astype(np.ubyte)
            logger.debug("Disturbance mask computation: Done (Window strip: {}".format(window))
            return disturbance_mask, window


    def build_disturbance_mask(self, 
                               dsm_path: str,
                               nb_max_workers : int,
                               slope_treshold: float = 2.0,
                               is_four_connexity : bool = True) -> np.array:
        """
        This method builds a mask corresponding to the disturbed areas in a given DSM.
        Most of those areas correspond to water or correlation issues during the DSM generation (obstruction, etc.).

        Args:
            dsm_path: path to the input DSM.
            nb_max_workers: number of availables workers (multiprocessing).
            slope_treshold: if the slope is greater than this threshold then we consider it as disturbed variation.
            is_four_connexity: number of evaluated axis. 
                            Vertical and horizontal if true else vertical, horizontal and diagonals.

        Returns:
            masks containing the disturbed areas.
        """
        logger.debug("disturbance mask: Done")
        # Determine the number of strip and their height
        with rasterio.open(dsm_path, 'r') as dataset:
            strip_height = dataset.height // nb_max_workers
            strips = [[i*strip_height-1, (i+1)*strip_height] for i in range(nb_max_workers)]
            # Borders handling
            strips[0][0] = 0
            strips[-1][1] = dataset.height - 1

            # Output binary mask initialization
            disturbance_mask = np.zeros((dataset.height, dataset.width), dtype = np.ubyte)

            # Launching parallel processing: each worker computes the disturbance mask for a given DSM strip
            with concurrent.futures.ProcessPoolExecutor(max_workers=nb_max_workers) as executor :
                futures = {executor.submit(self.compute_disturbance, dsm_path, Window(0,strip[0],dataset.width,strip[1]-strip[0]+1), 
                slope_treshold, is_four_connexity) for strip in strips}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Build Disturbance Mask") :
                    mask, window = future.result()
                    window_shape = window.flatten()
                    start_row = window_shape[1]
                    end_row = start_row + window_shape[3] - 1
                    start_row_mask = 0
                    end_row_mask = mask.shape[0] - 1 
                    if start_row > 0:
                        start_row_mask = 1
                        start_row = start_row + 1
                    
                    if end_row < dataset.height - 1:
                        end_row_mask = end_row_mask - 1
                        end_row = end_row - 1

                    disturbance_mask[start_row:end_row+1,:] = mask[start_row_mask:end_row_mask+1, :]
            logger.info("disturbance mask: Done")
            return disturbance_mask != 0

    def get_anchors(self,
                    dsm_path: str,
                    stride: int,
                    search_radius: int,
                    start_row: int, # // 2
                    end_row: int, # // 3
                    start_stable_row: int, # // 0
                    end_stable_row: int,
                    nbcols: int):
        """ """
        idwFilter = bi.PyBulldozerIDW()
        
        window = Window(0, start_row, nbcols, end_row - start_row)
        
        dsm_strip = rasterio.open(dsm_path).read(1, window=window).astype(np.float32)
        
        chunk_start_row = start_stable_row - start_row
        chunk_end_row = end_stable_row - start_row
 
        return start_row, end_row, idwFilter.get_anchors_points(dsm_strip, stride, search_radius, chunk_start_row, chunk_end_row, NO_DATA_VALUE)
    
    def compute_idw_chunk_dtm(self,
                              anchors_heights: np.ndarray,
                              anchors_indices: np.ndarray,
                              start_row: int,
                              end_row: int,
                              nbcols: int):

        """ """
        idwFilter = bi.PyBulldozerIDW()
        return start_row, end_row, idwFilter.build_idw_dtm(start_row, end_row, nbcols, anchors_indices, anchors_heights)



    def compute_idw_dtm(self,
                        preprocessed_dsm_path: str,
                        output_anchors_path: str,
                        output_idw_path: str):
        """
            This methods computes the anchor points and then compute a full DTM using IDW interpolation.
        """
        

        dsmDataset = rasterio.open(preprocessed_dsm_path)
        profile = dsmDataset.profile
        profile["dtype"] = np.uint32
        profile["nodata"] = 0

        nb_max_workers: int = 24
        stride: int = 10
        search_radius: int = 100
        
        # 13 // 4 = 3 (au lieu de 3.25) c'est le floor
        strip_height = dsmDataset.height // nb_max_workers
        strips = [ [i*strip_height, (i+1) * strip_height, 0, 0] for i in range(nb_max_workers)]
        strips[-1][1] = dsmDataset.height

        # Compute the margins for each strip
        for strip in strips:
            strip[2] = max(0, strip[0] - search_radius)
            strip[3] = min(dsmDataset.height, strip[1] + search_radius)
        
        anchorsImg = np.zeros((dsmDataset.height, dsmDataset.width), dtype=np.uint32)

        with concurrent.futures.ProcessPoolExecutor(max_workers=nb_max_workers) as executor :

            futures = { executor.submit(self.get_anchors, preprocessed_dsm_path, stride, search_radius, strip[2], strip[3], strip[0], strip[1], dsmDataset.width) for strip in strips }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Compute anchor image...") :
                start_row, end_row, chunk_anchors = future.result()
                anchorsImg[start_row:end_row, :] = np.add(chunk_anchors, anchorsImg[start_row:end_row, :])
        
        write_dataset(buffer_path = output_anchors_path, buffer = anchorsImg, profile = profile)

        anchor_indices = (((anchorsImg.flatten() > 0).nonzero())[0]).astype(np.uint32)
        anchorsImg = None

        # retrieve altitudes of anchor points
        heights = ((dsmDataset.read(1).astype(np.float32)).flatten())[anchor_indices]

        idw_dtm = np.zeros((dsmDataset.height, dsmDataset.width), dtype=np.float32)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=nb_max_workers) as executor :

            futures = { executor.submit(self.compute_idw_chunk_dtm, heights, anchor_indices, strip[0], strip[1], dsmDataset.width) for strip in strips }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Compute idw dtm...") :
                start_row, end_row, idw_chunk_dtm = future.result()
                idw_dtm[start_row:end_row, :] = idw_chunk_dtm
                
        
        profile["dtype"] = np.float32
        profile["nodata"] = NO_DATA_VALUE
        write_dataset(buffer_path = output_idw_path, buffer = idw_dtm, profile = profile)

        

    def write_quality_mask(self,
                        border_nodata_mask: np.ndarray,
                        inner_nodata_mask : np.ndarray, 
                        disturbed_area_mask : np.ndarray,
                        output_dir : str,
                        profile : rasterio.profiles.Profile) -> None:
        """
        This method merges the nodata masks generated during the DSM preprocessing into a single quality mask.
        There is a priority order: inner_nodata > disturbance
        (e.g. if a pixel is tagged as disturbed and inner_nodata, the output value will correspond to inner_nodata).

        Args:
            inner_nodata_mask: nodata areas in the input DSM.
            disturbed_area_mask: areas flagged as nodata due to their aspect (mainly correlation issue).
            output_dir: bulldozer output directory. The quality mask will be written in this folder.
            profile: DSM profile (TIF metadata).
        """     
        quality_mask = np.zeros(np.shape(inner_nodata_mask), dtype=np.uint8)
        quality_mask_path = output_dir + "quality_mask.tif"

        # Metadata update
        profile['dtype'] = np.uint8
        profile['count'] = 1
        # We don't except nodata value in this mask
        profile['nodata'] = 0
        quality_mask[disturbed_area_mask] = 2
        quality_mask[inner_nodata_mask] = 1
        quality_mask[border_nodata_mask] = 3
        write_dataset(quality_mask_path, quality_mask, profile)



    def run(self,
            dsm_path : str, 
            output_dir : str,
            nb_max_workers : int,
            create_filled_dsm : bool = False,
            nodata : float = None,
            slope_treshold : float = 2.0, 
            is_four_connexity : bool = True) -> None:
        """
        This method merges the nodata masks generated during the DSM preprocessing into a single quality mask.
        There is a priority order: inner_nodata > disturbance
        (e.g. if a pixel is tagged as disturbed and inner_nodata, the output value will correspond to inner_nodata).

        Args:
            dsm_path: path to the input DSM.
            output_dir: bulldozer output directory. The quality mask will be written in this folder.
            nb_max_workers: number of availables workers (for multiprocessing purpose).
            nodata: nodata value of the input DSM. If None, retrieve this value from the input DSM metadata.
            create_filled_dsm: flag to indicate if bulldozer has to generate a filled DSM (without nodata or disturbed areas).
            slope_treshold: if the slope is greater than this threshold then we consider it as disturbed variation.
            is_four_connexity: number of evaluated axis. 
                            Vertical and horizontal if true else vertical, horizontal and diagonals.
        """   


        logger.debug("Starting preprocess")
        with rasterio.open(dsm_path) as dsm_dataset:
            if not nodata:
                # If nodata is not specified in the config file, retrieve the value from the dsm metadata
                nodata = dsm_dataset.nodata
                if nodata is None:
                    # We assume that is nodata is not given then nodata = NODATA_VALUE
                    nodata = NO_DATA_VALUE
            
            dsm = dsm_dataset.read(1)

            # Converts no data
            if np.isnan(nodata):
                dsm = np.nan_to_num(dsm, False, nan=NO_DATA_VALUE)
            else:
                if nodata != NO_DATA_VALUE:
                    dsm = np.where(dsm == nodata, NO_DATA_VALUE, dsm)

            filledDSMProfile = dsm_dataset.profile
            filledDSMProfile['nodata'] = NO_DATA_VALUE
            
            # Generates inner nodata mask
            border_nodata_mask, inner_nodata_mask = self.build_inner_nodata_mask(dsm)
            dsm[border_nodata_mask] = np.max(dsm)
                    
            # Retrieves the disturbed area mask (mainly correlation issues: occlusion, water, etc.)
            disturbed_area_mask = self.build_disturbance_mask(dsm_path, nb_max_workers, slope_treshold, is_four_connexity)
            
            # Merges and writes the quality mask
            self.write_quality_mask(border_nodata_mask, inner_nodata_mask, disturbed_area_mask, output_dir, dsm_dataset.profile)

            # Generates filled DSM if the user provides a valid filled_dsm_path
            if create_filled_dsm:
                filled_dsm = fillnodata(dsm, mask=np.invert(inner_nodata_mask))
                filled_dsm = fillnodata(filled_dsm, mask=np.invert(disturbed_area_mask))

                filled_dsm_path = output_dir + 'filled_DSM.tif'

                # Generates the filled DSM file (DSM without inner nodata nor disturbed areas)
                write_dataset(filled_dsm_path, filled_dsm, filledDSMProfile)
            
            dsm[disturbed_area_mask] = nodata


            # Creates the preprocessed DSM. This DSM is only intended for bulldozer DTM extraction function.
            preprocessed_dsm_path = output_dir + 'preprocessed_DSM.tif'
            write_dataset(preprocessed_dsm_path, dsm, filledDSMProfile)

            # Create the IDW DTM to fill hole by a smooth interpolation
            #output_anchor_path  = output_dir + "/anchors.tif"
            # The idea is to do an approximation of the idw, for the computation of idw, i propose to divide into tiles of 500 * 500 or lower and for all the point of the tile, just use
            # the anchor points 
            #self.compute_idw_dtm(preprocessed_dsm_path=preprocessed_dsm_path, output_anchors_path= output_dir + "/anchors.tif", output_idw_path=output_dir + "/idw_dtm.tif")

            #anchorBuffer = rasterio.open(output_anchor_path).read(1)
            #print("Nb anchors ", (anchorBuffer>0).sum())

            dsm_dataset.close()
            logger.info("preprocess: Done")


if __name__ == "__main__":
    # assert(len(sys.argv)>=X)#X = nb arguments obligatoires +1 car il y a sys.argv[0] qui vaut le nom de la fonction
    # argv[1] should be the path to the input DSM
    input_dsm_path = sys.argv[1]

    # input file format check
    if not (input_dsm_path.endswith('.tiff') or input_dsm_path.endswith('.tif')) :
        logger.exception('\'dsm_path\' argument should be a path to a TIF file (here: {})'.format(input_dsm_path))
        raise ValueError()
    # input file existence check
    if not os.path.isfile(input_dsm_path):
        logger.exception('The input DSM file \'{}\' doesn\'t exist'.format(input_dsm_path))
        raise FileNotFoundError()
    
    output_dir = sys.argv[2]

    # preprocess(input_dsm_path, output_dir,XXX)

    # If this module is called in standalone, the user doesn't required the preprocessed DSM. We remove it
    preprocessed_dsm_path = output_dir + 'preprocessed_DSM.tif' 
    try:
        remove(preprocessed_dsm_path)
    except OSError:
        logger.warning("Error occured during the preprocessed DSM file deletion. Path: {}".format(preprocessed_dsm_path))
