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

import os
from collections import namedtuple
import concurrent.futures
import rasterio
import logging
import numpy as np
from tqdm import tqdm
import bulldozer.springforce as sf
from bulldozer.utils.logging_helper import BulldozerLogger

# Lorsqu'on upsample on a potentiellement upsamplé du nodata et il faut faire l'intersection avec le DSM au niveau donné pour le retirer
# et le remplir (fillnodata, idw)
# Mais attention à chaque niveau il faudra aussi rajouter le nodata du MNS.

Tile = namedtuple('Tile', ['start_y', 'start_x', 'end_y', 'end_x', 'margin_top', 'margin_right', 'margin_bottom', 'margin_left', 'path'])

class ClothSimulation(object):

    def __init__(self,            
                 max_object_size: int = 16,
                 uniform_filter_size: int = 3,
                 prevent_unhook_iter: int = 100,
                 num_outer_iterations: int = 100,
                 num_inner_iterations: int= 10,
                 mp_tile_size: int = 1500,
                 mp_nb_procs: int = 16):
        """ """
        self.max_object_size: int = max_object_size
        self.uniform_filter_size: int = uniform_filter_size
        self.prevent_unhook_iter: int = prevent_unhook_iter
        self.num_outer_iterations: int = num_outer_iterations
        self.num_inner_iterations: int= num_inner_iterations
        self.mp_tile_size: int = mp_tile_size
        self.mp_nb_procs: int = mp_nb_procs


    def retrieve_dsm_resolution(self,
                                dsm_dataset: rasterio.DatasetReader) -> float:
        """ """
        # We assume that resolution is the same wrt to both image axis
        res_x: float =  dsm_dataset.transform[0]
        res_y: float = dsm_dataset.transform[4]
        if abs(res_x) != abs(res_y):
            raise ValueError("DSM GSD must be the same wrt to the rows and columns.")
        return abs(res_x)

    def next_power_of_2(self, x : int) -> int:
        """
        This function returns the smallest power of 2 that is greater than or equal to a given non-negative integer x.

        Args:
            x : non negative integer.

        Returns:
            the corresponding power index power (2**index >= x).
        """
        return 0 if x==0 else (1<<(x-1).bit_length()).bit_length() - 1
        
    def get_max_pyramid_level(self, max_object_size_pixels: float) -> int :
        """ 
            Given the max size of an object on the ground,
            this methods compute the max level of the pyramid
            for drap cloth algorithm
        """
        power = self.next_power_of_2(int(max_object_size_pixels))
        
        # Take the closest power to the max object size
        if abs(2**(power-1) - max_object_size_pixels) <  abs(2**(power) - max_object_size_pixels):
            power -= 1

        return power
        
    def downsample(self, buffer: np.ndarray) -> np.ndarray:
        """
            Simple factor X downsample
        """
        return buffer[::2, ::2]

    def upsample(self,
                 buffer: np.ndarray, 
                 shape : tuple):
        """
            Simple 2X upsampling, duplicate pixels (nearest neighbor interpolation).

            Args :
                buffer : input dtm.
                shape : output dimension of the next dtm.
            
            Returns :
                the input dtm upsampled with the NN-interpolation with the input shape dimension.
        """
        next_dtm = np.zeros(shape, dtype = np.float32)

        # Adjust the slicing for odd row count
        if next_dtm.shape[0] % 2 == 1:
            s0 = np.s_[:-1]
        else:
            s0 = np.s_[:]

        # Adjust the slicing for odd column count
        if next_dtm.shape[1] % 2 == 1:
            s1 = np.s_[:-1]
        else:
            s1 = np.s_[:]

        # copy in duplicate values for blocks of 2x2 pixels
        next_dtm[::2, ::2] = buffer
        next_dtm[1::2, ::2] = buffer[s0, :]
        next_dtm[::2, 1::2] = buffer[:, s1]
        next_dtm[1::2, 1::2] = buffer[s0, s1]

        return next_dtm

    def write_tiles(self,
                    tile_buffer: np.ndarray, 
                    tile_path: str,
                    original_profile: dict) -> None:
        """
        """
        tile_profile = original_profile
        tile_profile["count"] = 1
        tile_profile["width"] = tile_buffer.shape[1]
        tile_profile["height"] = tile_buffer.shape[0]
        tile_profile["dtype"] = np.float32
        with rasterio.open(tile_path, 'w', **tile_profile) as dst:
            dst.write(tile_buffer, 1)

    def build_tiles(self,
                    dtm: np.ndarray,
                    dsm: np.ndarray,
                    margin: int,
                    tmp_dir: str,
                    dsm_profile: dict) -> list:
        """
            Write dsm and dtm tiles to disk
            returns a list of tiles
        """
        nb_tiles_y = int(dsm.shape[0] / self.mp_tile_size) # 1
        if float(dsm.shape[0]) / self.mp_tile_size - int(dsm.shape[0] / self.mp_tile_size) > 0:
            nb_tiles_y+=1
        
        nb_tiles_x = int(dsm.shape[1] / self.mp_tile_size)
        if float(dsm.shape[1]) / self.mp_tile_size - int(dsm.shape[1] / self.mp_tile_size) > 0:
            nb_tiles_x+=1

        tile_pair_list = []
        
        # TODO handle cases where nb_tiles = 0
        for ty in range(nb_tiles_y):
            for tx in range(nb_tiles_x):
                start_y = ty * self.mp_tile_size
                start_x = tx * self.mp_tile_size
                end_y = min(dsm.shape[0] - 1, (ty+1)*self.mp_tile_size - 1)
                end_x = min(dsm.shape[1] - 1, (tx+1)*self.mp_tile_size - 1)
                margin_top = margin if start_y - margin > -1 else start_y
                margin_right = margin if end_x + margin < dsm.shape[1] else dsm.shape[1] - 1 - end_x
                margin_bottom = margin if end_y + margin < dsm.shape[0] else dsm.shape[0] - 1 - end_y
                margin_left = margin if start_x - margin > 0 else start_x

                # Extract the tiles
                tile_start_y = start_y - margin_top
                tile_start_x = start_x - margin_left
                tile_end_y = end_y + margin_bottom
                tile_end_x = end_x + margin_right

                tile_dsm_buffer = dsm[tile_start_y:tile_end_y+1, tile_start_x:tile_end_x+1]
                tile_dtm_buffer = dtm[tile_start_y:tile_end_y+1, tile_start_x:tile_end_x+1]
                tile_dsm_path = os.path.join(tmp_dir, "dsm_" + str(ty) + "_" + str(tx) + ".tif")
                tile_dtm_path = os.path.join(tmp_dir, "dtm_" + str(ty) + "_" + str(tx) + ".tif")

                self.write_tiles(tile_buffer = tile_dsm_buffer, 
                                 tile_path=tile_dsm_path, 
                                 original_profile=dsm_profile)
                
                self.write_tiles(tile_buffer = tile_dtm_buffer, 
                                 tile_path=tile_dtm_path, 
                                 original_profile=dsm_profile)

                tile_dsm = Tile(start_y, start_x, end_y, end_x, 
                                margin_top, margin_right, margin_bottom, margin_left, 
                                tile_dsm_path)
                
                tile_dtm = Tile(start_y, start_x, end_y, end_x, 
                                margin_top, margin_right, margin_bottom, margin_left, 
                                tile_dtm_path)

                tile_pair_list.append((tile_dsm, tile_dtm))
        
        
        return tile_pair_list


    def sequential_drape_cloth(self,
                               dtm: np.ndarray,
                               dsm: np.ndarray,
                               num_outer_iterations: int,
                               step: float,
                               nodata_val: float) -> None:

        bfilters = sf.PyBulldozerFilters()

        valid = dtm != nodata_val

        for i in range(num_outer_iterations):
            
            dtm[valid] += step
            
            for j in range(self.num_inner_iterations):

                # handle DSM intersections, snap back to below DSM
                np.minimum(dtm, dsm, out=dtm, where=valid)

                # apply spring tension forces (blur the DTM)
                nb_rows = dtm.shape[0]
                nb_cols = dtm.shape[1]
                dtm = bfilters.run(dtm, nb_rows, nb_cols, self.uniform_filter_size, nodata_val)
                dtm = dtm.reshape((nb_rows, nb_cols))
                
        # One final intersection check
        np.minimum(dtm, dsm, out=dtm, where=valid)

        return dtm

    def tiled_drape_cloth(self,
                          tile_pair: tuple,
                          num_outer_iterations: int,
                          step: float,
                          nodata_val: float,
                          tmp_dir: str) -> tuple:
        """
        """
        dsm_dataset = rasterio.open(tile_pair[0].path)
        dsm_profile = dsm_dataset.profile
        dsm = dsm_dataset.read()[0]
        dtm = rasterio.open(tile_pair[1].path).read()[0]

        dtm = self.sequential_drape_cloth(dtm = dtm,
                                          dsm = dsm,
                                          num_outer_iterations = num_outer_iterations,
                                          step = step,
                                          nodata_val = nodata_val)

        item = tile_pair[1].path.split("/")[-1]
        output_tile_dtm_path = os.path.join(tmp_dir, "output_" + item)
        self.write_tiles(dtm, output_tile_dtm_path, dsm_profile)

        return (tile_pair, output_tile_dtm_path)

    def run(self,
            dsm_path: str,
            output_dir: str,
            nodata_val : float):

        BulldozerLogger.log("Starting dtm extraction by Cloth simulation", logging.INFO)

        # Open the dsm dataset
        in_dsm_dataset = rasterio.open(dsm_path)
        in_dsm_profile = in_dsm_dataset.profile
        dtm_path = os.path.join(output_dir, "raw_DTM.tif")
        # Initial dsm
        dsm_pyramid = []
        dsm_pyramid.append(in_dsm_dataset.read().astype(np.float32)[0])

        # Retrieve dsm resolution
        dsm_res = self.retrieve_dsm_resolution(dsm_dataset=in_dsm_dataset)

        # Determine max object size in pixels
        max_object_size_pixels = self.max_object_size / dsm_res

        # Determine the dezoom factor wrt to max size of an object
        # on the ground.
        nb_levels = self.get_max_pyramid_level(max_object_size_pixels/2) + 1

        # Build the pyramid
        for j in tqdm(range(1, nb_levels), desc="Building dsm pyramid..."):
            dsm_pyramid.append(self.downsample(dsm_pyramid[j-1]))
        
        
        # Initialize the dtm at this current dsm.
        dtm = (dsm_pyramid[nb_levels-1]).copy()
        

        # Prevent unhook from hills
        bfilters = sf.PyBulldozerFilters()
        for i in tqdm(range(self.prevent_unhook_iter), desc="Prevent unhook from hills..."):
            nb_rows = dtm.shape[0]
            nb_cols = dtm.shape[1]
            dtm = bfilters.run(dtm, nb_rows, nb_cols, self.uniform_filter_size, nodata_val)
            dtm = dtm.reshape((nb_rows, nb_cols))
        
        
        # Get min and max valid height from dsm
        valid_data = dsm_pyramid[0][dsm_pyramid[0] != nodata_val]
        min_alt = np.min(valid_data)
        max_alt = np.max(valid_data)

        # We deduce the initial step
        step = (max_alt - min_alt) / self.num_outer_iterations

        # Init classical parameters of drap cloth
        level = nb_levels - 1
        max_level = nb_levels - 1
        current_num_outer_iterations = self.num_outer_iterations

        while level > -1:

            BulldozerLogger.log("Process level " + str(level) + "...", logging.INFO)

            if level < max_level:
                # Upsample current dtm to the next level
                dtm = self.upsample(dtm, dsm_pyramid[level].shape)

            # Check if we need to tile for multi processing execution
            # We tile only if tile_size + 2 * margin < max(dsm_pyramid[level].shape[0], dsm_pyramid[level].shape[1])
            margin = current_num_outer_iterations * self.num_inner_iterations * self.uniform_filter_size
            if self.mp_tile_size + 2 * margin < max(dsm_pyramid[level].shape[0], dsm_pyramid[level].shape[1]):
                # Build tiles and save them to disk
                tile_pair_list = self.build_tiles(dtm= dtm,
                                                  dsm=dsm_pyramid[level],
                                                  margin=margin,
                                                  tmp_dir=output_dir,
                                                  dsm_profile=in_dsm_profile)

                output_tiles = []

                # process each tile independently (results are flushed on disk, today disk accees is fast (SSD))
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.mp_nb_procs) as executor:
                    futures = {executor.submit(self.tiled_drape_cloth,
                                               tile_pair,
                                               current_num_outer_iterations,
                                               step,
                                               nodata_val,
                                               output_dir) for tile_pair in tile_pair_list}
                    
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parallel Drape Cloth execution..."):
                    tile_pair, output_path = future.result()
                    output_tiles.append((tile_pair, output_path))


                # concatenate in the output dtm
                for res in output_tiles:
                    input_tile_pair = res[0]
                    output_tile = res[1]
                    tile_dtm = rasterio.open(output_tile).read()[0]
                    
                    # Stable interval in input referential
                    start_y = input_tile_pair[0].start_y
                    start_x = input_tile_pair[0].start_x
                    end_y = input_tile_pair[0].end_y
                    end_x = input_tile_pair[0].end_x

                    # Stable area in tile
                    tstart_y = input_tile_pair[0].margin_top
                    tend_y = tstart_y + end_y - start_y
                    tstart_x = input_tile_pair[0].margin_left 
                    tend_x = tstart_x + end_x - start_x

                    # Remove the temp tile files
                    os.remove(output_tile)
                    os.remove(input_tile_pair[0].path)
                    os.remove(input_tile_pair[1].path)

                    dtm[start_y:end_y+1, start_x:end_x+1] = tile_dtm[tstart_y: tend_y + 1, tstart_x:tend_x+1]
            
            else:
                dtm = self.sequential_drape_cloth(dtm = dtm,
                                                  dsm = dsm_pyramid[level],
                                                  num_outer_iterations = current_num_outer_iterations,
                                                  step = step,
                                                  nodata_val = nodata_val)

            # Decrease level
            level -= 1

            # Decrease step and number of outer iterations
            step = step / (2 * 2 ** (max_level - level))
            current_num_outer_iterations = max(1, int(self.num_outer_iterations / 2**(max_level - level)))

        self.write_tiles(tile_buffer= dtm, 
                         tile_path = dtm_path,
                         original_profile = in_dsm_profile)
        
        BulldozerLogger.log("Dtm extraction done", logging.INFO)

        return dtm_path