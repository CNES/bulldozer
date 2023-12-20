import os
import pytest
import tempfile
import shutil

import rasterio as rio
import numpy as np

from bulldozer.pipeline import bulldozer_pipeline


@pytest.fixture
def input_dsm_path():
    return "/work/scratch/data/lallemd/RT_bulldozer_large_scale/data/in/ARCACHON/dsm_ARCACHON_tuile_6.tif"


@pytest.fixture
def ref_path():
    return "/work/scratch/data/emiliea/bulldozer/67_benchmark/out_ref_tests/"


def test_dsm_to_dtm(input_dsm_path, ref_path):

    with tempfile.TemporaryDirectory() as directory:
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory)

        dtm_path = os.path.join(directory, "final_dtm.tif")
        ref_dtm_path = os.path.join(ref_path, "final_dtm_default.tif")
        # shutil.copyfile(dtm_path, ref_dtm_path)

        with rio.open(dtm_path) as dtm:
            with rio.open(ref_dtm_path) as ref:
                np.testing.assert_allclose(
                    dtm.read(), ref.read()
                )


def test_dsm_to_dtm_anchor(input_dsm_path, ref_path):

    with tempfile.TemporaryDirectory() as directory:
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory,
                                      anchor_points_activation=True)

        dtm_path = os.path.join(directory, "final_dtm.tif")
        ref_dtm_path = os.path.join(ref_path, "final_dtm_anchor.tif")
        # shutil.copyfile(dtm_path, ref_dtm_path)

        with rio.open(dtm_path) as dtm:
            with rio.open(ref_dtm_path) as ref:
                np.testing.assert_allclose(
                    dtm.read(), ref.read()
                )


def test_dsm_to_dtm_reverse_drape(input_dsm_path, ref_path):

    with tempfile.TemporaryDirectory() as directory:
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory,
                                      reverse_drape_cloth_activation=True)

        dtm_path = os.path.join(directory, "final_dtm.tif")
        ref_dtm_path = os.path.join(ref_path, "final_dtm_reverse.tif")
        # shutil.copyfile(dtm_path, ref_dtm_path)

        with rio.open(dtm_path) as dtm:
            with rio.open(ref_dtm_path) as ref:
                np.testing.assert_allclose(
                    dtm.read(), ref.read()
                )


def test_dsm_to_dtm_anchor_reverse_drape(input_dsm_path, ref_path):

    with tempfile.TemporaryDirectory() as directory:
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory,
                                      anchor_points_activation=True,
                                      reverse_drape_cloth_activation=True)

        dtm_path = os.path.join(directory, "final_dtm.tif")
        ref_dtm_path = os.path.join(ref_path, "final_dtm_anchor_reverse.tif")
        # shutil.copyfile(dtm_path, ref_dtm_path)

        with rio.open(dtm_path) as dtm:
            with rio.open(ref_dtm_path) as ref:
                np.testing.assert_allclose(
                    dtm.read(), ref.read()
                )
