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
    return "/work/scratch/data/emiliea/bulldozer/ref/tuile6/"


def compare_dataset(path, ref_path):
    with rio.open(path) as ds:
        with rio.open(ref_path) as ref_ds:
            assert ds.profile['count'] == ref_ds.profile['count']

            for band in range(ref_ds.profile['count']):
                np.testing.assert_allclose(
                    ds.read(band+1), ref_ds.read(band+1)
                )


def test_dsm_to_dtm(input_dsm_path, ref_path):

    with tempfile.TemporaryDirectory() as directory:
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory,
                                      developer_mode=True,
                                      nb_max_workers=16)

        dtm_path = os.path.join(directory, "final_dtm.tif")
        mask_path = os.path.join(directory, "quality_mask.tif")
        ref_dtm_path = os.path.join(ref_path, "final_dtm.tif")
        ref_masks_path = os.path.join(ref_path, "quality_mask.tif")
        # shutil.copyfile(dtm_path, ref_dtm_path)
        # shutil.copyfile(mask_path, ref_masks_path)
        
        compare_dataset(dtm_path, ref_dtm_path)
        compare_dataset(mask_path, ref_masks_path)


def test_dsm_to_dtm_pre_anchor(input_dsm_path, ref_path):

    with tempfile.TemporaryDirectory() as directory:
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory,
                                      pre_anchor_points_activation=True,
                                      nb_max_workers=16)

        dtm_path = os.path.join(directory, "final_dtm.tif")
        ref_dtm_path = os.path.join(ref_path, "final_dtm_pre_anchor.tif")
        #shutil.copyfile(dtm_path, ref_dtm_path)

        compare_dataset(dtm_path, ref_dtm_path)


def test_dsm_to_dtm_post_anchor(input_dsm_path, ref_path):
    with tempfile.TemporaryDirectory() as directory:
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory,
                                      post_anchor_points_activation=True,
                                      nb_max_workers=16)

        dtm_path = os.path.join(directory, "final_dtm.tif")
        ref_dtm_path = os.path.join(ref_path, "final_dtm_post_anchor.tif")
        #shutil.copyfile(dtm_path, ref_dtm_path)
        
        compare_dataset(dtm_path, ref_dtm_path)


def test_dsm_to_dtm_reverse_drape(input_dsm_path, ref_path):

    with tempfile.TemporaryDirectory() as directory:
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory,
                                      reverse_drape_cloth_activation=True)

        dtm_path = os.path.join(directory, "final_dtm.tif")
        ref_dtm_path = os.path.join(ref_path, "final_dtm_reverse.tif")
        #shutil.copyfile(dtm_path, ref_dtm_path)
        
        compare_dataset(dtm_path, ref_dtm_path)


def test_dsm_to_dtm_all_option(input_dsm_path, ref_path):

    with tempfile.TemporaryDirectory() as directory:
        bulldozer_pipeline.dsm_to_dtm(dsm_path=input_dsm_path,
                                      output_dir=directory,
                                      pre_anchor_points_activation=True,
                                      post_anchor_points_activation=True,
                                      reverse_drape_cloth_activation=True)

        dtm_path = os.path.join(directory, "final_dtm.tif")
        ref_dtm_path = os.path.join(ref_path, "final_dtm_all_options.tif")
        #shutil.copyfile(dtm_path, ref_dtm_path)

        compare_dataset(dtm_path, ref_dtm_path)
