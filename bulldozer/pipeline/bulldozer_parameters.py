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

"""
    This module contains the Bulldozer pipeline parameter structure and the dictionnary of default parameters.
"""

from bulldozer.utils.bulldozer_argparse import REQ_PARAM_KEY, OPT_PARAM_KEY, EXPERT_PARAM_KEY

# This value is used if the provided nodata value is None or NaN
DEFAULT_NODATA = -32768.0


class BulldozerParam:
    """
        Bulldozer pipeline parameter structure.
    """

    def __init__(self, name: str, alias: str, label: str, description: str, 
                 param_type: type, default_value: object, value_label: str = None) -> None:
        """
        BulldozerParam constructor.
        
        Args:
            name: parameter name (key in the yaml configuration file and str in the CLI).
            alias: parameter alias used for the CLI.
            label: parameter label suitable for display (without underscore, etc.).
            param_type: parameter type (used in QGIS plugin).
            description: complete parameter description displayed in helper.
            default_value: parameter default value.
            value_label: parameter value print in helper (e.g. "<value>").
        """
        self.name = name
        self.alias = alias
        self.label = label
        self.description = description
        self.param_type = param_type
        self.default_value = default_value
        self.value_label = value_label
    
    def __str__(self) -> str:
        """
            Human-friendly string description of BulldozerParam (method called by built-in print(), str(), and format() functions).
            
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
        return f"BulldozerParam(name=\"{self.name}\", alias=\"{self.alias}\", label=\"{self.label}\",description=\"{self.description}\", param_type={self.param_type}, default_value={self.default_value})"

        
# This dict store all the Bulldozer parameters description and default values
bulldozer_pipeline_params = {
    # Required parameters
    REQ_PARAM_KEY: [
        BulldozerParam("dsm_path", "dsm", "Input DSM", "Input DSM path.", str, None, "<path/dsm.tif>"),
        BulldozerParam("output_dir", "out", "Output directory", "Output directory path.", str, None, "<path>")
    ],
    # Options
    OPT_PARAM_KEY: [
        BulldozerParam("generate_dhm", "dhm", "Generate DHM", "Generate the Digital Height Model (DHM=DSM-DTM).", bool, False),
        BulldozerParam("max_object_size", "max_size", "Max object size (m)", "Foreground max object size (in meter).", float, 16, "<value>"),
        BulldozerParam("ground_mask_path", "ground", "Ground mask path", "Path to the binary ground classification mask.", str, None, "<mask.tif>"),
        BulldozerParam("activate_ground_anchors", "anchors", "Activate ground anchors", "Activate ground anchor detection (ground pre-detection).", bool, False),
        BulldozerParam("nb_max_workers", "workers", "Number of workers", "Max number of CPU core to use.", int, None, "<value>"),
        BulldozerParam("developer_mode", "dev", "Developper mode", "To keep the intermediate results.", bool, False)
    ],
    # Expert options: these parameters are considered as core settings and must be changed by users who are experts
    EXPERT_PARAM_KEY: [
        BulldozerParam("reg_filtering_iter", "reg_it", "Number of regular mask filtering iterations", "Number of regular mask filtering iterations.", int, None, "<value>"),
        BulldozerParam("dsm_z_accuracy", "dsm_z", "DSM altimetric accuracy (m)", "Altimetric height accuracy of the input DSM (m). If null, use the default value: 2*planimetric resolution.", float, None, "<value>"),
        BulldozerParam("max_ground_slope", "max_slope", "Max ground slope (%%)", "Maximum slope of the observed landscape terrain (%%).", float, 20.0, "<value>"),
        BulldozerParam("prevent_unhook_iter", "unhook_it", "Unhook iterations", "Number of unhook iterations.", int, 10, "<value>"),
        BulldozerParam("num_outer_iter", "outer", "Number of outer iterations", "Number of gravity step iterations.", int, 25, "<value>"),
        BulldozerParam("num_inner_iter", "inner", "Number of inner iterations", "Number of tension iterations.", int, 5, "<value>")
    ]
}
