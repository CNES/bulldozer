import os
import pytest
import tempfile
import shutil

import rasterio as rio
import numpy as np

from bulldozer.pipeline import bulldozer_pipeline


@pytest.fixture
def input_dsm_path():
    # return "/work/scratch/data/lallemd/RT_bulldozer_large_scale/data/in/ARCACHON/dsm_ARCACHON_tuile_6.tif"
    return "/work/scratch/data/otty/albertville.tif"


@pytest.fixture
def left_input_dsm_path():
    return "/work/scratch/data/emiliea/bulldozer/dsm_ARCACHON_tuile_6_0_1048.tif"


@pytest.fixture
def right_input_dsm_path():
    return "/work/scratch/data/emiliea/bulldozer/dsm_ARCACHON_tuile_6_960_2048.tif"


@pytest.fixture
def ref_path():
    return "/work/scratch/data/emiliea/bulldozer/67_benchmark/tuile6/"


# def test_margin(left_input_dsm_path, right_input_dsm_path, ref_path):
#
#     with tempfile.TemporaryDirectory() as directory:
#         directory = "/work/scratch/data/emiliea/bulldozer/67_benchmark/min_max/margin/"
#         bulldozer_pipeline.dsm_to_dtm(dsm_path=right_input_dsm_path,
#                                       output_dir=os.path.join(directory, 'right'),
#                                       developer_mode=True)
#
#         bulldozer_pipeline.dsm_to_dtm(dsm_path=left_input_dsm_path,
#                                       output_dir=os.path.join(directory, 'left'),
#                                       developer_mode=True)


def test_dsm_to_dtm(input_dsm_path, ref_path):

    with tempfile.TemporaryDirectory() as directory:
        directory = "/work/scratch/data/emiliea/bulldozer/67_benchmark/albertville/min_max/default"
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory,
                                      developer_mode=True,
                                      nb_max_workers=16)

        # dtm_path = os.path.join(directory, "final_dtm.tif")
        # ref_dtm_path = os.path.join(ref_path, "final_dtm.tif")
        # shutil.copyfile(dtm_path, ref_dtm_path)
        #
        # with rio.open(dtm_path) as dtm:
        #     with rio.open(ref_dtm_path) as ref:
        #         np.testing.assert_allclose(
        #             dtm.read(), ref.read()
        #         )


def test_dsm_to_dtm_pre_anchor(input_dsm_path, ref_path):

    with tempfile.TemporaryDirectory() as directory:
        directory = "/work/scratch/data/emiliea/bulldozer/67_benchmark/albertville/min_max/pre_anchor/"
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory,
                                      pre_anchor_points_activation=True,
                                      nb_max_workers=16)

        # dtm_path = os.path.join(directory, "final_dtm.tif")
        # ref_dtm_path = os.path.join(ref_path, "final_dtm_pre_anchor.tif")
        # shutil.copyfile(dtm_path, ref_dtm_path)
        #
        # with rio.open(dtm_path) as dtm:
        #     with rio.open(ref_dtm_path) as ref:
        #         np.testing.assert_allclose(
        #             dtm.read(), ref.read()
        #         )


def test_dsm_to_dtm_post_anchor(input_dsm_path, ref_path):
    with tempfile.TemporaryDirectory() as directory:
        directory = "/work/scratch/data/emiliea/bulldozer/67_benchmark/albertville/min_max/post_anchor/"
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory,
                                      post_anchor_points_activation=True,
                                      nb_max_workers=16)

        # dtm_path = os.path.join(directory, "final_dtm.tif")
        # ref_dtm_path = os.path.join(ref_path, "final_dtm_post_anchor.tif")
        # shutil.copyfile(dtm_path, ref_dtm_path)
        #
        # with rio.open(dtm_path) as dtm:
        #     with rio.open(ref_dtm_path) as ref:
        #         np.testing.assert_allclose(
        #             dtm.read(), ref.read()
        #         )


def test_dsm_to_dtm_reverse_drape(input_dsm_path, ref_path):

    with tempfile.TemporaryDirectory() as directory:
        directory = "/work/scratch/data/emiliea/bulldozer/67_benchmark/albertville/min_max/reverse/"
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory,
                                      reverse_drape_cloth_activation=True)

        # dtm_path = os.path.join(directory, "final_dtm.tif")
        # ref_dtm_path = os.path.join(ref_path, "final_dtm_reverse.tif")
        # shutil.copyfile(dtm_path, ref_dtm_path)
        #
        # with rio.open(dtm_path) as dtm:
        #     with rio.open(ref_dtm_path) as ref:
        #         np.testing.assert_allclose(
        #             dtm.read(), ref.read()
        #         )


# def test_dsm_to_dtm_all_option(input_dsm_path, ref_path):
#
#     with tempfile.TemporaryDirectory() as directory:
#         bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
#                                       output_dir=directory,
#                                       pre_anchor_points_activation=True,
#                                       post_anchor_points_activation=True,
#                                       reverse_drape_cloth_activation=True)
#
#         dtm_path = os.path.join(directory, "final_dtm.tif")
#         ref_dtm_path = os.path.join(ref_path, "final_dtm_all_options.tif")
#         shutil.copyfile(dtm_path, ref_dtm_path)
#
#         with rio.open(dtm_path) as dtm:
#             with rio.open(ref_dtm_path) as ref:
#                 np.testing.assert_allclose(
#                     dtm.read(), ref.read()
#                 )
