#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022 Centre National d'Etudes Spatiales (CNES).
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

"""
This module is used to postprocess the DTM in order to improve its quality. It required a DTM generated from Bulldozer.
"""
import time
import logging
import rasterio
import numpy as np
import scipy.ndimage as ndimage
from gdal import Warp
from rasterio.fill import fillnodata
from bulldozer.utils.helper import write_dataset, Runtime
from bulldozer.utils.logging_helper import BulldozerLogger

@Runtime
def build_pits_mask(dtm_path : np.ndarray, 
                    nb_max_workers : int = 1) -> np.ndarray:
    """
    This method detects pits in the input DTM.
    Those pits are generated during the DTM extraction by remaining artefacts.

    Args:
        dtm_path: path to the DTM extracted with bulldozer.
        nb_max_workers: number of availables workers (multiprocessing requirement).

    Returns:
        mask flagging the pits in the input DTM.
    """
    BulldozerLogger.log("Pits mask generation: Start", logging.DEBUG)

    with rasterio.open(dtm_path, 'r') as dataset:
        pits_mask = np.zeros([dataset.profile['height'], dataset.profile['width']], dtype=np.bool)
        resolution = dataset.profile['transform'][0]
        # Empirically defined constant. To be improved
        filter_size = 35.5/resolution

        # Compute the strip for multi-processing
        strip_height = dataset.height // nb_max_workers
        strips = [[i*strip_height, (i+1)*strip_height-1] for i in range(nb_max_workers)]
        strips[-1][1] = dataset.height - 1

        #TODO add mp func
        #detect_pits(dtm_path, filter_size, strip)

        return pits_mask

def detect_pits(dtm_path : str, 
                filter_size : int, 
                window : rasterio.windows.Window) -> (np.ndarray, rasterio.windows.Window)  :
    """
    This method detect the pits in the input DTM for a given window.

    Args:
        dtm_path: path to the input DTM.
        filter_size: uniform filter size (35.5/dtm_resolution suggested).
        window: coordinates of the concerned window.

    Returns:
        the pits mask and the corresponding window.
    """
    with rasterio.open(dtm_path, 'r') as dataset:
        strip = dataset.read(1, window=window).astype(np.float32)
        pits_mask = np.zeros(np.shape(strip), dtype=bool)

        # Generates the low frenquency DTM
        dtm_LF = ndimage.uniform_filter(strip, size = filter_size)
        # Retrieves the high frenquencies in the input DTM
        dtm_HF = strip - dtm_LF

        # Tags the pits
        pits_mask[dtm_HF < 0.] = 1

        return(pits_mask, window)

@Runtime
def fill_pits(raw_dtm_path : str,
              pits_mask : np.ndarray, 
              out_dtm_path : str = None, 
              nb_max_workers : int = 1) -> None:
    """
    This method fills the pits of the input raw DTM and writes the result in out_dtm_path.

    Args:
        raw_dtm_path: path to the input raw DTM.
        pits_mask: corresponding pits mask.
        out_dtm_path: path to the output filled DTM. If None, overrides the raw_dtm_path raster.
        nb_max_workers: number of availables workers (multiprocessing requirement).
    """
    BulldozerLogger.log("Pits filling: Start", logging.DEBUG)

    if not out_dtm_path:
        out_dtm_path = raw_dtm_path
    with rasterio.open(raw_dtm_path, 'r') as dataset:

        resolution = dataset.profile['transform'][0]
        # Empirically defined constant. To be improved
        filter_size = 35.5/resolution

        # Compute the strip for multi-processing
        strip_height = dataset.height // nb_max_workers
        strips = [[i*strip_height, (i+1)*strip_height-1] for i in range(nb_max_workers)]
        strips[-1][1] = dataset.height - 1

        #TODO add mp func
        #fill_pits_wd(out_dtm_path, pits_mask[strip], filter_size, strip)

def fill_pits_wd(raw_dtm_path : str,
                 dtm_path : str, 
                 pits_mask : np.array, 
                 filter_size : int, 
                 window : rasterio.windows.Window) -> (np.ndarray, rasterio.windows.Window)  :
    """
    This method fill the pits in the input DTM for a given window.

    Args:
        raw_dtm_path: path to the input raw DTM.
        dtm_path: path to the output filled DTM (can be the same path as raw_dtm_path).
        pits_mask: the corresponding pits mask.
        filter_size: uniform filter size (35.5/dtm_resolution suggested).
        window: coordinates of the concerned window.
    """
    with rasterio.open(raw_dtm_path, 'r') as dataset:
        strip = dataset.read(1, window=window).astype(np.float32)
        
        #TODO handle boundaries
        # Generates the low frenquency DTM
        dtm_LF = ndimage.uniform_filter(strip, size = filter_size)
        strip[pits_mask] = dtm_LF

        write_dataset(dtm_path, strip, dataset.profile, window=window)

@Runtime
def build_dhm(dsm_path : str, 
              dtm_path : str, 
              output_dir : str, 
              nb_max_workers : int = 1,
              nodata : float = None) -> None:
    """
    This method generates a Digital Height Model DHM (DSM-DTM) in the provided directory.
    The DSM and DTM must have the same metadata (shape, type, etc.).

    Args:
        dsm_path: path to the input DSM.
        dtm_path: path to the input DTM.
        output_dir: path to the output DHM file.
        nb_max_workers: number of availables workers (multiprocessing requirement).
    """
    BulldozerLogger.log("DHM generation: Start", logging.DEBUG)
    with rasterio.open(dsm_path, 'r') as dataset:
        # Compute the strip for multi-processing
        strip_height = dataset.height // nb_max_workers
        strips = [[i*strip_height, (i+1)*strip_height-1] for i in range(nb_max_workers)]
        strips[-1][1] = dataset.height - 1

        dhm_path = output_dir + '/DHM.tif'
        dhm_profile = dataset.profile
        if nodata:
            dhm_profile['nodata'] = nodata
        dhm_dataset = rasterio.open(dhm_path, 'w', **dhm_profile)
        # Launching parallel processing: each worker computes DHM for a given DSM and DTM strip
        with concurrent.futures.ProcessPoolExecutor(max_workers=nb_max_workers) as executor :
            futures = {executor.submit(self.compute_dhm, dsm_path, dtm_path, dhm_path, Window(0,strip[0],dataset.width,strip[1]-strip[0]+1), nodata) for strip in strips}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Build DHM") :
                    dhm_chunk, window = future.result()
                    dhm_dataset.write(dhm_chunk)

def compute_dhm(dsm_path : str, 
                dtm_path : str, 
                dhm_path : str, 
                window : rasterio.windows.Window,
                nodata : float = None) -> (np.ndarray, rasterio.windows.Window)  :
    """
    This method computes the Digital Height Model DHM (DSM-DTM) for a given window.

    Args:
        dsm_path: path to the input DSM.
        dtm_path: path to the input DTM.
        dhm_path: path to the output DHM.
        window: coordinates of the concerned window.

    Returns:
        the DHM values and the corresponding window.
    """
    with rasterio.open(dsm_path, 'r') as dsm_dataset:
        dsm_strip = dsm_dataset.read(1, window=window).astype(np.float32)
        with rasterio.open(dtm_path, 'r') as dtm_dataset:
            dtm_strip = dtm_dataset.read(1, window=window).astype(np.float32)
            dhm_strip = dsm_strip - dtm_strip
            # Retrieve the nodata from the input DSM
            if not nodata :
                nodata = dsm_dataset.nodata
            dhm_strip[dsm_strip == nodata] == nodata
            return(dhm_strip, window)

@Runtime
def reproject(in_raster_path : str, 
              target_CRS : str, 
              out_raster_path : str = None) -> None :
    """
    This function allows to reproject a raster to a target Coordinate Reference System (CRS).
    It is based on the gdalwarp function.

    Args:
        in_raster_path: path to the input raster.
        target_CRS: name of the target CRS. it must be compatible with the gdal format (ex:"EPSG:4326").
        out_raster_path:  path to the output raster. If no output raster path is provided, then the input raster path is used (overrides).  
    """
    BulldozerLogger.log("Reprojection: Start", logging.DEBUG)
    # If no output path is provided, then overrides the input raster
    if not out_raster_path:
        out_raster_path = in_raster_path
    Warp(destNameOrDestDS=out_raster_path, srcDSOrSrcDSTab=in_raster_path, dstSRS=target_CRS)

@Runtime
def postprocess_pipeline(raw_dtm_path : str, 
        output_dir : str,
        nb_max_workers : int = 1,
        quality_mask_path : str = None, 
        generate_dhm : bool = False,
        dsm_path : str = None,
        output_CRS : str = None) -> None:
    """
    Bulldozer postprocess pipeline. It removes remaining pits.
    It also generates the DHM and reproject the products if the options are set.

    Args:
        raw_dtm_path: path to the DTM generated with bulldozer (raw).
        output_dir: path to the output directory.
        nb_max_workers: number of availables workers (multiprocessing requirement).
        quality_mask_path: path to the quality mask associated with the DTM.
        generate_dhm: option that indicates if bulldozer has to generate the DHM (DSM-DTM).
        dsm_path: path to the input DSM. This argument is required for the DHM generation.
        output_CRS: if a CRS (different from the input DSM) is provided, reproject the DTM to the new CRS.
    """
    BulldozerLogger.log("Starting postprocess", logging.DEBUG)
    # Detects the pits in the DTM (due to correlation issue)
    pits_mask = build_pits_mask(raw_dtm_path, nb_max_workers)

    dtm_path = os.path.join(cfg['outputDir'], 'DTM.tif')
    # Fills the detected pits
    fill_pits(raw_dtm_path, dtm_path, pits_mask, nb_max_workers)
            
    with rasterio.open(dtm_path, 'r') as dtm_dataset:
        # Updates the quality mask if it's provided
        if quality_mask_path:
            # Add a new band for the pits mask
            with rasterio.open(quality_mask_path, 'r') as q_mask_dataset:
                q_mask_profile = q_mask_dataset.profile
                q_mask_profile['count'] = q_mask_dataset['count'] + 1
                write_dataset(quality_mask_path, pits_mask, q_mask_dataset.profile, band=q_mask_profile['count'])
        else:
            BulldozerLogger.log("No quality mask provided", logging.WARNING)
        # Generates the DHM (DSM - DTM) if the option is activated
        if generate_dhm and dsm_path:
            build_dhm(dsm_path, dtm_path, output_dir, dtm_dataset.profile, nb_max_workers)

        # Check if the output CRS is different from the input. If it's different, reproject
        if (output_CRS and output_CRS!=dtm_dataset['crs']):
            reproject(dtm_path, output_CRS)
            # If a quality mask is provided, reproject it
            if quality_mask_path:
                #TODO : reproject correctly the quality mask (avoid interpolation)
                reproject(quality_mask, output_CRS)
            # If the DHM option is set, reproject the DHM
            if dhm:   
                reproject(output_dir+"/DHM.tif", output_CRS)