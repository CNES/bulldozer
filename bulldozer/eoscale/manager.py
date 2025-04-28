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

import rasterio
import uuid
import numpy
import copy

import bulldozer.eoscale.utils as eoutils
import bulldozer.eoscale.shared as eosh

class EOContextManager:

    def __init__(self, 
                 nb_workers:int,
                 tile_mode: bool = False):

        self.nb_workers = nb_workers
        self.tile_mode = tile_mode
        self.shared_resources: dict = dict()
        # Key is a unique memview key and value is a tuple (shared_resource_key, array subset, profile_subset)
        self.shared_mem_views: dict = dict()
    
    def __enter__(self):
        self.start()
        return self
    
    def  __exit__(self, exc_type, exc_value, traceback):
        self.end()

    # Private methods

    def _release_all(self):

        self.shared_mem_views = dict()

        for key in self.shared_resources:
            self.shared_resources[key].release()
        
        self.shared_resources = dict()
    
    # Public methods

    def open_raster(self,
                    raster_path: str) -> str:
        """
        Create a new shared instance from file and return its virtual path
        """
        
        new_shared_resource = eosh.EOShared()
        new_shared_resource.create_from_raster_path(raster_path=raster_path)
        self.shared_resources[new_shared_resource.virtual_path] = new_shared_resource
        return new_shared_resource.virtual_path
    
    def create_image(self, profile: dict) -> str:
        """
            Given a profile with at least the following keys:
            count
            height
            width
            dtype
            this method allocates a shared image and its metadata 
        """
        eoshared_instance = eosh.EOShared()
        eoshared_instance.create_array(profile = profile)
        self.shared_resources[eoshared_instance.virtual_path] = eoshared_instance
        return eoshared_instance.virtual_path
    
    def create_memview(self, key: str, arr_subset: numpy.ndarray, arr_subset_profile: dict) -> str:
        """
            This method allows the developper to indicate a subset memory view of a shared resource he wants to use as input
            of an executor.
        """
        mem_view_key: str = str(uuid.uuid4())
        self.shared_mem_views[mem_view_key] = (key, arr_subset, arr_subset_profile)
        return mem_view_key
    
    def get_array(self, key: str, tile: eoutils.MpTile = None) -> numpy.ndarray:
        """
            This method returns a memory view from the key given by the user.
            This key can be a shared resource key or a memory view key 
        """
        if key in self.shared_mem_views:
            if tile is None:
                return self.shared_mem_views[key][1]
            else:
                start_y = tile.start_y - tile.top_margin
                end_y = tile.end_y + tile.bottom_margin + 1
                start_x = tile.start_x - tile.left_margin
                end_x = tile.end_x + tile.right_margin + 1
                return self.shared_mem_views[key][1][:, start_y:end_y, start_x:end_x]
        else:
            return self.shared_resources[key].get_array(tile = tile)
    
    def get_profile(self, key: str) -> dict:
        """
            This method returns a profile from the key given by the user.
            This key can be a shared resource key or a memory view key
        """
        if key in self.shared_mem_views:
            return copy.deepcopy(self.shared_mem_views[key][2])
        else:
            return self.shared_resources[key].get_profile()
    
    def release(self, key: str):
        """
            Release definitely the corresponding shared resource
        """

        mem_view_keys_to_remove: list = []
        # Remove from the mem view dictionnary all the key related to the share resource key
        for k in self.shared_mem_views:
            if self.shared_mem_views[k][0] == key:
                mem_view_keys_to_remove.append(k)
        for k in mem_view_keys_to_remove:
            del self.shared_mem_views[k]

        if key in self.shared_resources:
            self.shared_resources[key].release()
            del self.shared_resources[key]
    
    def write(self, key: str, img_path: str, binary: bool = False, profile: dict = None):
        """
            Write the corresponding shared resource to disk
        """
        if key in self.shared_resources:
            if profile:
                target_profile = profile
            else:
                target_profile = self.shared_resources[key].get_profile()
                target_profile['driver'] = 'GTiff'
                target_profile['interleave'] = 'band'
            img_buffer = self.shared_resources[key].get_array()
            if binary:
                with rasterio.open(img_path, "w", nbits=1,**target_profile) as out_dataset:
                    out_dataset.write(img_buffer)             
            else:
                with rasterio.open(img_path, "w", **target_profile) as out_dataset:
                    out_dataset.write(img_buffer)
        else:
            print(f"WARNING: the key {key} to write is not known by the context manager")
    
    def update_profile(self, key: str, profile: dict) -> str:
        """
            This method update the profile of a given key and returns the new key
        """
        tmp_value = self.shared_resources[key]
        del self.shared_resources[key]
        tmp_value._release_profile()
        new_key: str = tmp_value._update_profile(profile)
        self.shared_resources[new_key] = tmp_value
        return new_key
    
    def start(self):
        if len(self.shared_resources) > 0:
            self._release_all()

    def end(self):
        self._release_all()