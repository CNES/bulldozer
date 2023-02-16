import os
import numpy as np
import rasterio
from multiprocessing import shared_memory
import json
import sys
from importlib.metadata import metadata
import uuid

class Shared(object):
    '''
    Shared allow to manage an array in shared memory
    The class :
     - store an array in multiprocessing.shared_memory
     - manage metada to transtyping data in a more complex object
     - manage lifetime of shared memory
    '''

    @staticmethod
    def make_shared_from_numpy(bigArray : np.ndarray):
        '''
        Construct a shared data from a numpy array 
        '''
        shd = Shared()
        shd.put_array_from_numpy(bigArray)
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
        Create directly an array in sahred memory
        '''
        self.__close_if_exist(self.resourceKey)
        self.__close_if_exist(self.resourceKey + 'metadata')
                              
        # store metadata
        self.metadata = {"shape": shape, "dtype": np.dtype(dtype).name}
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
        
        
        
    def put_array_from_numpy(self, bigArray : np.ndarray) :
        '''
        create a shared data from a raster a numpy array
        '''
        self.__close_if_exist(self.resourceKey)
        self.__close_if_exist(self.resourceKey + 'metadata')
    
    
        # store metadata
        self.metadata = {"shape": bigArray.shape, "dtype": np.dtype(bigArray.dtype).name}
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
            self.metadata = {"shape": dst.shape, "dtype": np.dtype(dtype).name}

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
        return np.ndarray(self.metadata["shape"], 
                        dtype=self.metadata["dtype"], 
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


    