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
    This module is used to display the helper of the Bulldozer command.
"""

import argparse
from gettext import gettext as _
import sys as _sys

REQ_PARAM_KEY = "REQUIRED"
OPT_PARAM_KEY = "OPTIONAL"
EXPERT_PARAM_KEY = "EXPERT OPTIONAL"


class BulldozerHelpFormatter(argparse.RawTextHelpFormatter):
    """
        Bulldozer formatter of help messages for command line options.
        Displays help message for execution with configuration file and CLI arguments.
    """

    def __init__(self, **kwargs) -> None:
        """
            BulldozerHelpFormatter constructor.
        """
        argparse.HelpFormatter.__init__(self, max_help_position=55, **kwargs)


    def bulldozer_add_usage(self, positionals: list = [], 
                            optionals: list = [], commons: list = [], 
                            prefix: str = None) -> None:
        """
            This method generates the usage description.

            Args:
                positionals: list of all positional arguments as actions (for bulldozer: only config file path expected).
                optionals: list of all Bulldozer optional arguments as actions.
                commons: list of all common optional arguments (for example: help, version, long usage).
                prefix: string to describe the usage.
        """
        args = positionals, optionals, commons, prefix
        self._add_item(self._bulldozer_format_usage, args)


    def _bulldozer_format_usage(self, positionals: list, 
                                optionals: list, commons: list, 
                                prefix: str) -> str:
        """
            This method defines the usage description format.

                Args:
                    positionals: list of all positional arguments as actions (for bulldozer: only config file path expected).
                    optionals: list of all Bulldozer optional arguments as actions.
                    commons: list of all common optional arguments (for example: help, version, long usage).
                    prefix: string to describe the usage.

                Returns:
                    Usage description content as string.
                """
        # Define prefixes
        if prefix is None:
            prefix = _("usage: ")

        # Define usages
        prog = "%(prog)s" % dict(prog=self._prog)

        # build full usage string
        format = self._format_actions_usage
        commons_usage = format(commons, [])
        usage = prog + " " + " ".join([param for param in [commons_usage] if param])
        if optionals:
            # Formatter used for CLI with parameters usage
            for it, param in enumerate([arg+"] " for arg in format(optionals, []).split("] ") if arg]):
                # Skip a line every 3 parameters
                if param:
                    if it % 3 == 0:
                        # Add spacing before optionals parameters for alignement purpose 
                        usage+= "\n" + (len(prog)+len(prefix)+1)*" " + param
                    else :
                        usage+= param
            # Remove last "] " char
            usage = usage[:-2]
            # Add spacing before positionals parameters for alignement purpose 
            usage+= "\n" + (len(prog)+len(prefix)+1)*" " + " ".join([param for param in [format(positionals, [])] if param])
        else:
            # Formatter used for CLI with config file usage
            usage+= " " + " ".join([param for param in [format(positionals, [])] if param])


        return "%s%s\n\n" % (prefix, usage)


class BulldozerArgumentParser(argparse.ArgumentParser):
    """
        Bulldozer arguments parser.
    """

    def __init__(self, **kwargs):
        """
            BulldozerHelpFormatter constructor.
        """
        argparse.ArgumentParser.__init__(self, formatter_class=BulldozerHelpFormatter, **kwargs)


    def get_bulldozer_groups(self) -> tuple:
        """
            This method returns the two groups of arguments containing the Bulldozer parameters (required and optional).

            Returns:
                cli_positionals: Arguments group for required Bulldozer parameters.
                cli_optionals: Arguments group for optional Bulldozer parameters.

            Raises:
                ValueError: if unknown group description is found.
        """
        cli_positionals = []
        cli_optionals = []
        cli_experts_optionals = []
        for action_group in self._action_groups:
            if action_group.title is None:  # the group contains Bulldozer parameters
                if action_group.description == REQ_PARAM_KEY:
                    cli_positionals = action_group
                elif action_group.description == OPT_PARAM_KEY:
                    cli_optionals = action_group
                elif action_group.description == EXPERT_PARAM_KEY:
                    cli_experts_optionals = action_group
                else:
                    raise ValueError(
                        f"Unknown group description {action_group.description}: expects {REQ_PARAM_KEY}, {OPT_PARAM_KEY} or {EXPERT_PARAM_KEY}"
                    )

        return cli_positionals, cli_optionals, cli_experts_optionals
    

    def print_help(self, file: str = None, long_help: bool = False, expert_mode: bool = False) -> None:
        """
            This method generates the help message.

            Args:
                file: path to a file to save the output.
                long_help: whether print the full help or not (by default short help).
                export_mode: whether print the expert parameters or not (by default False).
        """
        if file is None:
            file = _sys.stdout
        self._print_message(self.format_help(long_help, expert_mode), file)

    def format_usage(self) -> str:
        """
            This method generates the Bulldozer global usage description.
            It contains both usages: using config file or using CLI arguments.

            Returns:
                The usage description for Bulldozer.
        """
        return self.format_help(add_description=False, add_epilog=False)
    
    def format_help(self, long_help: bool = False, expert_mode: bool = False,
                    add_description: bool = True, add_epilog: bool = True) -> str:
        """
            This method generates the bulldozer global help message.
            It contains both execution methods: using a config file or using CLI arguments.

            Args:
                long_help: whether to display only usage description (False) or full arguments descriptions (True).
                export_mode: whether to display expert arguments or not.
                add_description: whether to add description or not at the beginning of the help message.
                add_epilog: whether to add epilog or not at the end of the help message.

            Returns:
                The help message for Bulldozer.
        """
        # get groups corresponding to bulldozer parameters
        cli_pos_group, cli_opt_group, cli_expert_group = self.get_bulldozer_groups()

        # prepare input arguments for both config file and cli usages
        cli_positionals = cli_pos_group._group_actions  # positional arguments for CLI
        # We set positionals to required (for visual display)
        for action in cli_positionals:
            action.required = True
        cli_optionals = cli_opt_group._group_actions  # optionals arguments for CLI
        if expert_mode:
            cli_optionals += cli_expert_group._group_actions
        positionals = self._positionals._group_actions  # positional arguments (only config file path)
        positionals[0].nargs = None  # We set the nargs to None to highlight the requirement of this parameter

        # format
        formatter = self._get_formatter()
        
        # description
        if (add_description):
            formatter.add_text(self.description)

        # usage and help with config file
        formatter.bulldozer_add_usage(positionals, commons=self._optionals._group_actions,
                                      prefix=_("Usage with config file: "))
        
        if long_help:
            # long help positional arguments
            formatter.start_section(self._positionals.title)
            formatter.add_text(self._positionals.description)
            formatter.add_arguments(self._positionals._group_actions)
            formatter.end_section()
            # long help optional arguments
            formatter.start_section(self._optionals.title)
            formatter.add_text(self._optionals.description)
            formatter.add_arguments(self._optionals._group_actions)
            formatter.end_section()

        # epilog
        if add_epilog:
            formatter.add_text("If extra arguments are provided, these will override the original values from the configuration file.")
        
        formatter.add_text("---------------------------------")

        # usage and help with cli arguments
        formatter.bulldozer_add_usage(cli_positionals, cli_optionals, self._optionals._group_actions,
                                   prefix=_("Usage with parameters: "))
        
        if long_help:
            # long help positional arguments
            formatter.start_section("required arguments")
            formatter.add_arguments(cli_positionals)
            formatter.end_section()
            # long help optional arguments
            formatter.start_section(self._optionals.title)
            formatter.add_arguments(self._optionals._group_actions)
            formatter.end_section()
            formatter.start_section(None)  # for visual spacing
            formatter.add_arguments(cli_optionals)
            formatter.end_section()
        
        # epilog
        if add_epilog:
            if long_help:
                #TODO uncomment when doc is online
                #formatter.add_text("For more details, consult https://bulldozer.readthedocs.io/")
                formatter.add_text("For more details, consult the documentation.")
            else:
                epilog = self.epilog.replace("prog", "%(prog)s" % dict(prog=formatter._prog))
                formatter.add_text(epilog)

        # determine help from format above
        return formatter.format_help()
