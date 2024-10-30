#!/usr/bin/env python
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

from multiprocessing import shared_memory
import rasterio
import numpy
import uuid
import os
import json
import copy
import bulldozer.eoscale.utils as eoutils

EOSHARED_PREFIX: str = "eoshared"
EOSHARED_MTD: str = "metadata"

class EOShared:

    def __init__(self, virtual_path: str = None):
        """ """
        self.shared_array_memory = None
        self.shared_metadata_memory = None
        self.virtual_path: str = None

        if virtual_path is not None:
            self._open_from_virtual_path(virtual_path = virtual_path)
    
    def _extract_from_vpath(self) -> tuple:
        """
            Extract resource key and metada length from the virtual path
        """
        split_v_path: list = self.virtual_path.split("/")
        resource_key: str = split_v_path[2]
        mtd_len: str = split_v_path[1]
        return resource_key, mtd_len    

    def _open_from_virtual_path(self, virtual_path: str):
        """ """

        self.virtual_path = virtual_path
        
        resource_key, mtd_len = self._extract_from_vpath()

        self.shared_array_memory = shared_memory.SharedMemory(name=resource_key, 
                                                              create=False)

        self.shared_metadata_memory = shared_memory.SharedMemory(name=resource_key + EOSHARED_MTD, 
                                                                 create=False)

    def _build_virtual_path(self, key: str, mtd_len: str) -> None:
        """ """
        self.virtual_path = EOSHARED_PREFIX + "/" + mtd_len + "/" + key
    
    def _create_shared_metadata(self, profile: dict, key: str):
        """ """
        # Encode and compute the number of bytes of the metadata
        encoded_metadata = json.dumps(eoutils.rasterio_profile_to_dict(profile)).encode()
        mtd_size: int = len(encoded_metadata)
        self.shared_metadata_memory = shared_memory.SharedMemory(create=True, 
                                                                 size=mtd_size, 
                                                                 name=key + EOSHARED_MTD)
        self.shared_metadata_memory.buf[:] = encoded_metadata[:]

        # Create the virtual path to these shared resources
        self._build_virtual_path(mtd_len=str(mtd_size), key = key)

    def create_array(self, profile: dict):
        """
            Allocate array 
        """
        # Shared key is made unique
        # this property is awesome since it allows the communication between parallel tasks
        resource_key: str = str(uuid.uuid4())

        # Compute the number of bytes of this array
        d_size = numpy.dtype(profile["dtype"]).itemsize * profile["count"] * profile["height"] * profile["width"]

        # Create a shared memory instance of it
        # shared memory must remain open to keep the memory view
        self.shared_array_memory = shared_memory.SharedMemory(create=True, 
                                                              size=d_size, 
                                                              name=resource_key)
        
        big_array = numpy.ndarray(shape=(profile["count"]  * profile["height"] * profile["width"]), 
                                         dtype= numpy.dtype(profile["dtype"]), 
                                         buffer=self.shared_array_memory.buf)
                               
        big_array[:] = 0

        self._create_shared_metadata(profile = profile, key = resource_key)

    def create_from_raster_path(self,
                                raster_path: str) -> str :
        
        """ Create a shared memory numpy array from a raster image """
        with rasterio.open(raster_path, "r") as raster_dataset:

            # Shared key is made unique
            # this property is awesome since it allows the communication between parallel tasks
            resource_key: str = str(uuid.uuid4())

            # Compute the number of bytes of this array
            d_size = numpy.dtype(raster_dataset.dtypes[0]).itemsize * raster_dataset.count * raster_dataset.height * raster_dataset.width

            # Create a shared memory instance of it
            # shared memory must remain open to keep the memory view
            self.shared_array_memory = shared_memory.SharedMemory(create=True, 
                                                                  size=d_size, 
                                                                  name=resource_key)
            

            big_array = numpy.ndarray(shape=(raster_dataset.count  * raster_dataset.height * raster_dataset.width), 
                                      dtype=raster_dataset.dtypes[0], 
                                      buffer=self.shared_array_memory.buf)

            big_array[:] = raster_dataset.read().flatten()[:]

            self._create_shared_metadata(profile = raster_dataset.profile, key = resource_key)

    def get_profile(self) -> rasterio.DatasetReader.profile:
        """
            Return a copy of the rasterio profile
        """
        resource_key, mtd_len = self._extract_from_vpath()
        encoded_mtd = bytearray(int(mtd_len))
        encoded_mtd[:] = self.shared_metadata_memory.buf[:]
        return copy.deepcopy(eoutils.dict_to_rasterio_profile(json.loads(encoded_mtd.decode())))

    def get_array(self, 
                  tile: eoutils.MpTile = None) -> numpy.ndarray:

        """            
            Return a memory view of the array or a subset of it if a tile is given
            This has be done to be respect the dimension condition of the n_images_to_m_images filter.
        """
        profile = self.get_profile()
        array_shape = (profile['count'], profile['height'], profile['width'])
        
        arr = numpy.ndarray(array_shape,
                            dtype=profile['dtype'],
                            buffer=self.shared_array_memory.buf)

        if tile is None:
            return arr
        else:
            start_y = tile.start_y - tile.top_margin
            end_y = tile.end_y + tile.bottom_margin + 1
            start_x = tile.start_x - tile.left_margin
            end_x = tile.end_x + tile.right_margin + 1
            return arr[:, start_y:end_y, start_x:end_x]
    
    def _update_profile(self, profile: dict) -> None:
        """ 
            Update a shared metadata memory of an existing eoshared resource, 
            normally it might not be called directly by the user because it is a dangerous operation...
        """

        resource_key, mtd_len = self._extract_from_vpath()

        self._create_shared_metadata(profile = profile, key = resource_key)

        return self.virtual_path

    
    def _release_profile(self):
        """
            Method used by EOScale only, normally it might not be called directly by the user
        """
        if self.shared_metadata_memory is not None:
            self.shared_metadata_memory.close()
            self.shared_metadata_memory.unlink()
            self.shared_metadata_memory = None
    
    def close(self):
        """ 
            A close does not mean release from memory. Must be called by a process once it has finished
            with this resource.
        """
        if self.shared_array_memory is not None:
            self.shared_array_memory.close()
        
        if self.shared_metadata_memory is not None:
            self.shared_metadata_memory.close()
    
    def release(self):
    
        """ 
            Definitely release the shared memory.
        """
        if self.shared_array_memory is not None:
            self.shared_array_memory.close()
            self.shared_array_memory.unlink()
            self.shared_array_memory = None
        
        self._release_profile()

