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

from setuptools import setup, Extension, find_packages
from distutils.util import convert_path
from Cython.Build import cythonize
import numpy

extensions = [
    Extension( "bulldozer.preprocessing.regular", ["bulldozer/preprocessing/regular_detection/cython/regular.pyx"]),
    Extension( "bulldozer.preprocessing.border", ["bulldozer/preprocessing/border_detection/cython/border.pyx"]),
    Extension( "bulldozer.preprocessing.fill", ["bulldozer/preprocessing/dsm_filling/cython/fill.pyx"], include_dirs=[numpy.get_include()])
]

compiler_directives = { "language_level": 3, "embedsignature": True}
extensions = cythonize(extensions, compiler_directives=compiler_directives)

main_ns = {}
ver_path = convert_path("bulldozer/_version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

try:
    setup(
        version = main_ns['__version__'],
        ext_modules=extensions,
        packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
    )
except Exception:
    print(
        "\n\nAn error occurred while building the project, "
        "please ensure you have the most updated version of setuptools, "
        "setuptools_scm and wheel with:\n"
        "\tpip install -U setuptools setuptools_scm wheel\n\n"
    )
    raise

