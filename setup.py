# Copyright 2021 PIERRE LASSALLE
# All rights reserved

from setuptools import find_packages, setup
from Cython.Build import cythonize

setup(
    name="bulldozer",
    version="0.1.0",
    description="DTM extraction using improved drap cloth methods",
    url="https://gitlab.cnes.fr/ai4geo/lot6/bulldozer.git",
    author="Dimitri Lallement & Pierre Lassalle",
    author_email="dimitri.lallement@cnes.fr & pierre.lassalle@cnes.fr",
    license="A definir",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        "numpy",
        "scipy",
        "rasterio",
        "tqdm",
        "PyYAML"
    ],
    # extras_require={
    #     "dev": [
    #         "black",
    #         "flake8",
    #         "isort",
    #         "pre-commit",
    #         "pytest==5.0.1",  # pytest pined to v5.0.1 to avoid issue when run from VSCode
    #         "pytest-cov",
    #         "tox",
    #     ]
    # },
    zip_safe=False,
    # entry_points={
    #     "console_scripts": ["bulldozer=bulldozer.bulldozer:main"]
    # },
    ext_modules=cythonize(["bulldozer/core/cpp_core/*.pyx"], build_dir="bulldozer/core/cpp_core/build/"),
)
