import os
import numpy as np
import rasterio
from rasterio import Affine
from multiprocessing import shared_memory
import json
import sys
from importlib.metadata import metadata
import uuid

def rasterioProfileToSerializableDict(rioDataset: rasterio.DatasetReader) -> dict:
    rioDict = {}
    rioProfile = rioDataset.profile
    for key, value in rioProfile.items():
        if key == "crs":
            rioDict['crs'] = int(str(rioDataset.crs).split(":")[1])
        elif key == "transform":
            rioDict['transform_1'] = rioProfile['transform'][0]
            rioDict['transform_2'] = rioProfile['transform'][1]
            rioDict['transform_3'] = rioProfile['transform'][2]
            rioDict['transform_4'] = rioProfile['transform'][3]
            rioDict['transform_5'] = rioProfile['transform'][4]
            rioDict['transform_6'] = rioProfile['transform'][5]
        else:
            rioDict[key] = value
    return rioDict


def dictToRasterioProfile(rioDict: dict) -> dict :
    """
        {'driver': 'GTiff', 'dtype': 'float32', 'nodata': -32768.0, 'width': 7755, 'height': 6523, 'count': 1, 'crs': CRS.from_epsg(32654), 
        'transform': Affine(0.5, 0.0, 388296.0,
       0.0, -0.5, 3950161.0), 'blockxsize': 256, 'blockysize': 256, 'tiled': True, 'interleave': 'band'}
    """
    rioProfile = {}
    rioProfile['driver'] = rioDict['driver'] if 'driver' in rioDict else None
    rioProfile['dtype'] = rioDict['dtype'] if 'dtype' in rioDict else None 
    rioProfile['nodata'] = rioDict['nodata'] if 'nodata' in rioDict else None
    rioProfile['width'] = rioDict['width'] if 'width' in rioDict else None
    rioProfile['height'] = rioDict['height'] if 'height' in rioDict else None
    rioProfile['count'] = rioDict['count'] if 'count' in rioDict else None
    rioProfile['crs'] = rasterio.crs.CRS.from_epsg(rioDict['crs']) if 'crs' in rioDict else None
    if 'transform_1' in rioDict:
        rioProfile['transform'] = Affine(rioDict['transform_1'], 
                                        rioDict['transform_2'], 
                                        rioDict['transform_3'], 
                                        rioDict['transform_4'], 
                                        rioDict['transform_5'], 
                                        rioDict['transform_6'])
    else:
        rioProfile['transform'] = Affine(1, 0, 0, 0, 1, 0)

    rioProfile['blockxsize'] = rioDict['blockxsize'] if 'blockxsize' in rioDict else None
    rioProfile['blockysize'] = rioDict['blockysize'] if 'blockysize' in rioDict else None
    rioProfile['tiled'] = rioDict['tiled'] if 'tiled' in rioDict else None
    rioProfile['interleave'] = rioDict['interleave'] if 'interleave' in rioDict else None

    return rioProfile

def getRasterioDimensionFromShape(shape: tuple) -> dict:
    assert(len(shape) == 2 or len(shape) == 3)
    width: int = 0
    height: int = 0
    count: int = 0
    if len(shape) == 2:
        count = 1
        height = shape[0]
        width = shape[1]
    elif len(shape) == 3:
        count = shape[0]
        height = shape[1]
        width = shape[2]
    return count, height, width

def getNumpyArrayShapeFromProfile(profile: dict) -> tuple:
    """ """
    return ( profile["height"], profile["width"] ) if profile["count"] == 1 else (profile["count"], profile["height"], profile["width"] )

class Shared(object):
    '''
    Shared allow to manage an array in shared memory
    The class :
     - store an array in multiprocessing.shared_memory
     - manage metada to transtyping data in a more complex object
     - manage lifetime of shared memory
    '''

    @staticmethod
    def make_shared_from_numpy(bigArray : np.ndarray, rioDataset: rasterio.DatasetReader = None):
        '''
        Construct a shared data from a numpy array 
        '''
        shd = Shared()
        shd.put_array_from_numpy(bigArray, rioDataset)
        return shd
    
    @staticmethod
    def make_shared_from_rasterio(path : str):
        '''
        Construct a shared data from a raster on filesystem
        '''
        shd = Shared()
        shd.put_array_from_rasterio(path)
        return shd
        
    @staticmethod
    def make_shared_array(shape, dtype):
        '''
        Construct directly an array in shared memory
        '''
        shd = Shared()
        shd.create(shape, dtype)
        return shd

    
    @staticmethod
    def is_shared_memory_path(path : str) :
        return path.startswith("shd:/")

    def __init__(self):
        '''
        Constructor
        '''
        self.shm = None
        self.pid = os.getpid()
        self.metadata: dict = {}
        # Shared key is made unique. Allow to call severals softawre using Shared without conflict
        self.resourceKey = str(uuid.uuid4()) 
        
        
    def __del__(self) :
        '''
        Destructor
        '''
        self.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def open(self, path : str):
        '''
        Open an existing Shared Object path is given by get_memory_path
        '''
        if not self.is_shared_memory_path(path) :
            raise FileNotFoundError(path)
        
        self.resourceKey = path.lstrip('shd:/') # path.removeprefix('shd:/')
        self.pid=-1
        
        shm = shared_memory.SharedMemory(name=self.resourceKey+'metadata')
        self.metadata = json.loads(shm.buf.tobytes())
        shm.close()

        
    
    def close(self):
        '''
        When close is called from 
        '''
        if self.shm is not None :
            self.shm.close()
            if self.pid == os.getpid() :
                self.shm.unlink()
                
                self.__close_if_exist(self.resourceKey + 'metadata')
                
        self.shm = None
        
    
    def create(self, shape, dtype) :
        '''
        Create directly an array in shared memory
        '''
        self.__close_if_exist(self.resourceKey)
        self.__close_if_exist(self.resourceKey + 'metadata')
                              
        # store metadata
        count, height, width = getRasterioDimensionFromShape(shape = shape)
        self.metadata = { "profile" : {"count": count, "height": height, "width": width, "dtype": np.dtype(dtype).name} }
        encoded_metadata = json.dumps(self.metadata).encode()
        mtd_size = len(encoded_metadata)
        
        # Create metadata to store metadata
        shm = shared_memory.SharedMemory(create=True, size=mtd_size, name=self.resourceKey + 'metadata')
        shm.buf[:] = encoded_metadata[:]
        shm.close()
        
        # Compute the number of bytes of this array
        d_size = np.dtype(dtype).itemsize * np.prod(shape)
        
        # Create a shared memory instance of it
        self.shm = shared_memory.SharedMemory(create=True, size=d_size, name=self.resourceKey)
        
        
        
    def put_array_from_numpy(self, bigArray : np.ndarray, rioDataset: rasterio.DatasetReader) :
        '''
        create a shared data from a raster a numpy array
        '''
        self.__close_if_exist(self.resourceKey)
        self.__close_if_exist(self.resourceKey + 'metadata')
    
    
        # store metadata
        if rioDataset is not None:
            self.metadata = {"profile" : rasterioProfileToSerializableDict(rioDataset=rioDataset)}
        else:
            count, height, width = getRasterioDimensionFromShape(shape = bigArray.shape)
            self.metadata = {"profile" : {"count": count, "height": height, "width": width, "dtype": np.dtype(bigArray.dtype).name}}
        encoded_metadata = json.dumps(self.metadata).encode()
        mtd_size = len(encoded_metadata)
        
        # Create metadata to store metadata
        shm = shared_memory.SharedMemory(create=True, size=mtd_size, name=self.resourceKey + 'metadata')
        shm.buf[:] = encoded_metadata[:]
        shm.close()
    
        # Compute the number of bytes of this array
        d_size = np.dtype(bigArray.dtype).itemsize * np.prod(bigArray.shape)
        
        # Create a shared memory instance of it
        self.shm = shared_memory.SharedMemory(create=True, size=d_size, name=self.resourceKey)
        
        # numpy array on shared memory buffer
        dst = np.ndarray(shape=bigArray.shape, dtype=bigArray.dtype, buffer=self.shm.buf)       
        dst[:]= bigArray[:]
        

    def put_array_from_rasterio(self, path: str):
        '''
        Load a raster from disk directly to shared memory 
        '''

        self.__close_if_exist(self.resourceKey)
        self.__close_if_exist(self.resourceKey + 'metadata')
    
        with rasterio.open(path, "r") as dst :
           
            # Compute the number of bytes of this array
            dtype = dst.dtypes[0]
            d_size = np.dtype(dst.dtypes[0]).itemsize * dst.height * dst.width
            
            # store metadata
            self.metadata= {"profile": rasterioProfileToSerializableDict(dst) }

            encoded_metadata = json.dumps(self.metadata).encode()
            mtd_size = len(encoded_metadata)
            
            # Create metadata to store metadata
            shm = shared_memory.SharedMemory(create=True, size=mtd_size, name=self.resourceKey + 'metadata')
            shm.buf[:] = encoded_metadata[:]
            shm.close()
            
            # Create a shared memory instance of it
            # shared memory must remain open to keep the memory view 
            self.shm = shared_memory.SharedMemory(create=True, size=d_size, name=self.resourceKey)
          
            array = np.ndarray(shape=dst.shape, dtype=dtype, buffer=self.shm.buf) 
            dst.read(1, out=array)
    
    def get_memory_path(self) :
        '''
        get a path to sharesobject 
        '''
        return "shd:/" + self.resourceKey


    def getArray(self) -> np.ndarray :
        '''
        Retriece object from childs processus 
        '''
        self.shm = shared_memory.SharedMemory(name=self.resourceKey)
        arrayShape = getNumpyArrayShapeFromProfile(self.metadata["profile"])
        return np.ndarray(arrayShape, 
                        dtype=self.metadata["profile"]["dtype"], 
                        buffer=self.shm.buf)

    def __close_if_exist(self, key):
        '''
        
        '''
        try:
            shm = shared_memory.SharedMemory(name=key)
            if shm is not None:
                shm.close()
                shm.unlink()
        except:
            pass


    