# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

import unittest
import numpy as np
from bulldozer.core.dsm_preprocess import build_nodata_mask
from numpy.testing import assert_array_equal

class TestConfigParser(unittest.TestCase):
    
    def test_build_nodata_mask(self):
        # In this case the nodata value is set to -2
        input = np.array([[1,-2,3,4],[-2,1,-2,4],[-2,-2,3,-2],[-2,-2,-2,4]])
        # Expected inner nodata mask
        expected_inner = np.array([[False,False,False,False],[False,False,True,False],
                                    [False,False,False,False],[False,False,False,False]])
        # Expected border nodata mask
        expected_border = np.array([[False,True,False,False],[True,False,False,False],
                                    [True,True,False,True],[True,True,True,False]])
        # output[0] contains border nodata mask and output[1] contains inner nodata mask
        output = build_nodata_mask(input, -2)

        assert_array_equal(output[0], expected_border)
        assert_array_equal(output[1], expected_inner)

        # Same input dsm as previous but we replace the no data value to nan
        input = np.array([[1,np.nan,3,4],[np.nan,1,np.nan,4],
                        [np.nan,np.nan,3,np.nan],[np.nan,np.nan,np.nan,4]])
    
        output = build_nodata_mask(input, np.nan)

        # It should return the same results as previous
        assert_array_equal(output[0], expected_border)
        assert_array_equal(output[1], expected_inner) 
    