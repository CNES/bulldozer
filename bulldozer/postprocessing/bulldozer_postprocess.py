#!/usr/bin/env python
#
# Copyright (c) 2022-2026 Centre National d'Etudes Spatiales (CNES).
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

"""
This module is used to apply postprocessing on DTM and generates nDSM.
"""

import numpy as np

from bulldozer.eomultiprocessing.utils import read
from bulldozer.utils.bulldozer_logger import Runtime, logger


def ensure_dsm_below_dtm(dsm: np.ndarray, dtm: np.ndarray, nodata_mask: np.bool) -> np.ndarray:
    """
    This method ensures that the DTM is lower than or equal to the DSM and returns the cleaned DTM

    Args:
        dsm: input DSM.
        dtm: DTM to clean.
        nodata_mask: nodata mask.
        nodata: DTM nodata value (if nan, the nodata is set to default value: -32768.0).
        dsm: input DSM. (optional)

    Returns:
        The cleaned DTM.

    """
    logger.info("Ensuring DTM <= DSM...")
    np.minimum(dtm, dsm, out=dtm, where=~nodata_mask)

    return dtm


def compute_ndsm(dsm: np.ndarray, dtm: np.ndarray, nodata_mask: np.bool, nodata: float) -> np.ndarray:
    """
    This method computes the nDSM.

    Args:
        dsm: input DSM.
        dtm: input DTM.
        nodata_mask: nodata mask.
        nodata: DSM nodata value.

    Returns:
        tThe normalized DSM.

    """
    logger.info("nDSM processing...")
    dsm[:] = dsm - dtm
    dsm[nodata_mask] = nodata

    return dsm


@Runtime
def run_postprocess(
    dsm_key: np.ndarray | str,
    dtm_key: np.ndarray | str,
    nodata_mask: np.bool,
    nodata: float,
    check_below_dsm: bool = False,
    generate_ndsm: bool = False,
) -> dict[str, np.ndarray]:
    """
    This method applies postprocessing on the computed DTM.

    Args:
        dsm_key: input DSM (numpy array or path to file).
        dtm_key: input DTM (numpy array or path to file).
        nodata_mask: nodata mask.
        nodata: DSM nodata value.
        check_below_dsm: whether to check DTM <= DSM or not (by default False).
        generate_ndsm: whether to generate ndsm or not (by default False).

    Returns:
        A dict containing the cleand DTM and the nDSM (if generated).

    """
    dsm = read(dsm_key) if isinstance(dsm_key, str) else dsm_key
    dtm = read(dtm_key) if isinstance(dtm_key, str) else dtm_key

    res = {}

    if check_below_dsm:
        dtm[:] = ensure_dsm_below_dtm(dsm, dtm, nodata_mask)
        res["dtm"] = dtm

    if generate_ndsm:
        dsm[:] = compute_ndsm(dsm, dtm, nodata_mask, nodata)
        res["ndsm"] = dsm

    return res
