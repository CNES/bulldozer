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

import pytest
import logging
import os.path
import numpy as np
from pathlib import Path
from yaml import YAMLError

from bulldozer.utils.logging_helper import BulldozerLogger
from bulldozer.utils.config_parser import ConfigParser
    
@pytest.fixture
def setup():
    parser = ConfigParser(False)
    path = Path(__file__).parent / 'data/config_parser/'
    print("\nSetting up resources...")
    # Setting-up ressources
    yield {'parser': parser, 'path': path}  # Provide the data to the test
    # Teardown: Clean up resources (if any) after the test

def test_format_check(setup):
    # Should raise exception because it's not an YAML file
    path = "test.txt"
    parser = setup['parser']
    pytest.raises(ValueError, lambda: parser.read(path))
    # Should raise exception for file not found but should pass the format check
    path = "test.yaml"
    pytest.raises(FileNotFoundError, lambda: parser.read(path))
    # Should raise exception for file not found but should pass the format check
    path = "test.yml"
    pytest.raises(FileNotFoundError, lambda:parser.read(path))

def test_existence_check(setup):
    parser = setup['parser']
    # Should raise FileNotFoundException since the file doesn't exist
    path = str(setup['path']) + "/test.yaml"
    pytest.raises(FileNotFoundError, lambda: parser.read(path))
    
    # Shouldn't raise FileNotFoundException, if it raises an exception the unit test framework will flag this as an error
    path = str(setup['path']) + "/parser_test.yaml"
    parser.read(path)

def test_read(setup):
    path = str(setup['path']) + "/parser_test.yaml"
    cfg = setup['parser'].read(path)
    # Check data type
    assert isinstance(cfg, dict)

    # Check dict size (expected 6)
    assert len(cfg) == 6

    # Check string element read
    assert isinstance(cfg['str_test'], str)
    assert cfg['str_test'] == 'test'

    # Check integer element read
    assert isinstance(cfg['int_test'], int)
    assert cfg['int_test'] == 100

    # Check boolean element read
    assert isinstance(cfg['boolean_test'], bool)
    assert cfg['boolean_test'] == True

    # Check float sub-element read
    assert isinstance(cfg['parent'], dict)
    assert isinstance(cfg['parent']['child'], float)
    assert cfg['parent']['child'] == 10.3
    assert cfg['parent']['child2'] == 13
    
    # Check nan reading
    assert np.isnan(float(cfg['nan_test']))
    assert cfg['none_test'] is None
    assert not cfg['none_test']
    
    path = str(setup['path']) + "/wrong_syntax.yaml"
    # Should raise YAMLError due to the wrong YAML data format in the file
    pytest.raises(YAMLError, lambda: setup['parser'].read(path))


def test_verbose():
    non_verbose_parser = ConfigParser(verbose = False)
    # Check logging level value
    assert non_verbose_parser.level == logging.INFO
    
    verbose_parser = ConfigParser(verbose = True)
    # Check logging level value
    assert verbose_parser.level == logging.DEBUG