# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
# This file is part of Bulldozer
#
# All rights reserved.

import unittest
import numpy as np
from numpy.testing import assert_array_equal
from bulldozer.core.dtm_postprocess import compute_nodata_value, merge_nodata_masks

class TestDtmPostprocess(unittest.TestCase):
      
    def test_compute_nodata_value(self):
        # Checks all possible combinations 
        self.assertEqual(compute_nodata_value(False,False,False,False),0)
        self.assertEqual(compute_nodata_value(False,False,False,True),4)
        self.assertEqual(compute_nodata_value(False,False,True,False),3)
        self.assertEqual(compute_nodata_value(False,False,True,True),3)
        self.assertEqual(compute_nodata_value(False,True,False,False),2)
        self.assertEqual(compute_nodata_value(False,True,False,True),2)
        self.assertEqual(compute_nodata_value(False,True,True,False),2)
        self.assertEqual(compute_nodata_value(False,True,True,True),2)
        self.assertEqual(compute_nodata_value(True,False,False,False),1)
        self.assertEqual(compute_nodata_value(True,False,False,True),1)
        self.assertEqual(compute_nodata_value(True,False,True,False),1)
        self.assertEqual(compute_nodata_value(True,False,True,True),1)
        self.assertEqual(compute_nodata_value(True,True,False,False),1)
        self.assertEqual(compute_nodata_value(True,True,False,True),1)
        self.assertEqual(compute_nodata_value(True,True,True,False),1)
        self.assertEqual(compute_nodata_value(True,True,True,True),1)
        
    def test_merge_nodata_masks(self):
        # Checks all possible combinations (a border nodata can't be an inner nodata => 12 possible combinations)
        excepted = np.array([[1,3,3,1],[3,2,2,4],[0,2,2,3],[1,4,0,1]])
        border_nodata_mask = np.array([[True,False,False,True],[False, False,False, False],[False,False,False,False],[True,False,False,True]])
        inner_nodata_mask = np.array([[False,False,False,False],[False,True,True,False],[False,True,True,False],[False,False,False,False]])
        disturbance_mask = np.array([[False,True,True,True],[True,False,False,False],[False,True,True,True],[True,False,False,False]])
        sink_mask = np.array([[False,True,True,True],[False,True,False,True],[False,True,False,False],[False,True,False,True]])
        assert_array_equal(excepted, merge_nodata_masks(border_nodata_mask, inner_nodata_mask, disturbance_mask, sink_mask))