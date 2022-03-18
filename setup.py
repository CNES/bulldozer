# Copyright 2021 PIERRE LASSALLE
# All rights reserved

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

extensions = [
    Extension("bulldozer.disturbedareas.disturbedareas", ["src/bulldozer/disturbedareas/disturbedareas.pyx"]),
    Extension("bulldozer.springforce.springforce", ["src/bulldozer/springforce/springforce.pyx"])
]

compiler_directives = { "language_level": 3, "embedsignature": True}
extensions = cythonize(extensions, compiler_directives=compiler_directives)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

setup(
    ext_modules=extensions,
    install_requires=install_requires,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
)