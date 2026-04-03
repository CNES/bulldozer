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
This module contains the Bulldozer pipeline parameter structure and
the dictionary of default parameters.
"""

import multiprocessing as mp
from typing import Any

from bulldozer.utils.bulldozer_argparse import EXPERT_PARAM_KEY, OPT_PARAM_KEY, REQ_PARAM_KEY

# This value is used if the provided nodata value is None or NaN
DEFAULT_NODATA = -32768.0


class BulldozerParam:
    """
    Bulldozer pipeline parameter structure.
    """

    def __init__(
        self,
        name: str,
        alias: str,
        label: str,
        description: str,
        param_type: type,
        default_value: Any,
        value_label: str | None = None,
        value_choices: list[Any] | None = None,
    ) -> None:
        """
        BulldozerParam constructor.

        Args:
            name: parameter name
                  (key in the yaml configuration file and str in the CLI).
            alias: parameter alias used for the CLI.
            label: parameter label suitable for display (without underscore, etc.).
            param_type: parameter type (used in QGIS plugin).
            description: complete parameter description displayed in helper.
            default_value: parameter default value.
            value_label: parameter value print in helper (e.g. "<value>").
            value_choices: list of permitted values for the parameter.
        """
        self.name = name
        self.alias = alias
        self.label = label
        self.description = description
        self.param_type = param_type
        self.default_value = default_value
        self.value_label = value_label
        self.choices = value_choices

    def __str__(self) -> str:
        """
        Human-friendly string description of BulldozerParam
        (method called by built-in print(), str(), and format() functions).

        Returns:
            corresponding string.
        """
        return f"{self.name} {self.param_type} (default: {self.default_value})"

    def __repr__(self) -> str:
        """
        Detailed string that can be used to recreate the BulldozerParam.

        Returns:
            corresponding string.
        """
        return (
            f'BulldozerParam(name="{self.name}", alias="{self.alias}", '
            f'label="{self.label}", description="{self.description}", '
            f"param_type={self.param_type}, default_value={self.default_value})"
        )


# Dict with all the Bulldozer parameters description and default values
bulldozer_pipeline_params = {
    # Required parameters
    REQ_PARAM_KEY: [
        BulldozerParam(
            name="dsm_path",
            alias="dsm",
            label="Input DSM",
            description="Input DSM path.",
            param_type=str,
            default_value=None,
            value_label="<path/dsm.tif>",
        ),
        BulldozerParam(
            name="output_dir",
            alias="out",
            label="Output directory",
            description="Output directory path.",
            param_type=str,
            default_value=None,
            value_label="<path>",
        ),
    ],
    # Options
    OPT_PARAM_KEY: [
        BulldozerParam(
            name="generate_ndsm",
            alias="ndsm",
            label="Generate nDSM",
            description="Generate the Normalized Digital Surface Model (nDSM=DSM-DTM).",
            param_type=bool,
            default_value=False,
        ),
        BulldozerParam(
            name="max_object_size",
            alias="max_size",
            label="Max object size (m)",
            description="Foreground max object size (in meter).",
            param_type=float,
            default_value=16,
            value_label="<value>",
        ),
        BulldozerParam(
            name="ground_mask_path",
            alias="ground",
            label="Ground mask path",
            description="Path to the binary ground classification mask.",
            param_type=str,
            default_value=None,
            value_label="<mask.tif>",
        ),
        BulldozerParam(
            name="activate_ground_anchors",
            alias="anchors",
            label="Activate ground anchors",
            description="Activate ground anchor detection (ground pre-detection).",
            param_type=bool,
            default_value=False,
        ),
        BulldozerParam(
            name="nb_max_workers",
            alias="workers",
            label="Number of workers",
            description="Max number of CPU core to use.",
            param_type=int,
            default_value=None,
            value_label="<value>",
        ),
        BulldozerParam(
            name="developer_mode",
            alias="dev",
            label="Developper mode",
            description="To keep the intermediate results.",
            param_type=bool,
            default_value=False,
        ),
    ],
    # Expert options: these parameters are considered as core settings
    # must be changed by users who are experts
    EXPERT_PARAM_KEY: [
        BulldozerParam(
            name="reg_filtering_iter",
            alias="reg_it",
            label="Number of regular mask filtering iterations",
            description="Number of regular mask filtering iterations.",
            param_type=int,
            default_value=None,
            value_label="<value>",
        ),
        BulldozerParam(
            name="dsm_z_accuracy",
            alias="dsm_z",
            label="DSM altimetric accuracy (m)",
            description="Altimetric height accuracy of the input DSM (m). "
            "If null, use the default value: 2*planimetric resolution.",
            param_type=float,
            default_value=None,
            value_label="<value>",
        ),
        BulldozerParam(
            name="max_ground_slope",
            alias="max_slope",
            label="Max ground slope (%%)",
            description="Maximum slope of the observed landscape terrain (%%).",
            param_type=float,
            default_value=20.0,
            value_label="<value>",
        ),
        BulldozerParam(
            name="prevent_unhook_iter",
            alias="unhook_it",
            label="Unhook iterations",
            description="Number of unhook iterations.",
            param_type=int,
            default_value=10,
            value_label="<value>",
        ),
        BulldozerParam(
            name="num_outer_iter",
            alias="outer",
            label="Number of outer iterations",
            description="Number of gravity step iterations.",
            param_type=int,
            default_value=25,
            value_label="<value>",
        ),
        BulldozerParam(
            name="num_inner_iter",
            alias="inner",
            label="Number of inner iterations",
            description="Number of tension iterations.",
            param_type=int,
            default_value=5,
            value_label="<value>",
        ),
        BulldozerParam(
            name="mp_context",
            alias="context",
            label="Multiprocessing context",
            description=f"To use a multiprocessing context among those available : {mp.get_all_start_methods()}. "
            f"By default uses '{mp.get_start_method()}' (the default of the current OS)",
            param_type=str,
            default_value=mp.get_start_method(),
            value_label="<value>",
            value_choices=mp.get_all_start_methods(),
        ),
        BulldozerParam(
            name="intermediate_write",
            alias="inter_write",
            label="Write intermediate results",
            description="To write intermediate results instead of keeping all in memory.",
            param_type=bool,
            default_value=False,
        ),
        BulldozerParam(
            name="enforce_dtm_below_dsm",
            alias="below_dsm",
            label="Ensure that DTM <= DSM",
            description="Ensure that DTM <= DSm even in noisy areas.",
            param_type=bool,
            default_value=False,
        ),
    ],
}
