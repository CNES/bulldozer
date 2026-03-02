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
This module is used to display the helper of the Bulldozer command.
"""

import argparse
import sys as _sys
from gettext import gettext as _
from typing import Any

REQ_PARAM_KEY = "REQUIRED"
OPT_PARAM_KEY = "OPTIONAL"
EXPERT_PARAM_KEY = "EXPERT OPTIONAL"


class BulldozerHelpFormatter(argparse.RawTextHelpFormatter):
    """
    Bulldozer formatter of help messages for command line options.
    Displays help message for execution with configuration file and CLI arguments.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        BulldozerHelpFormatter constructor.
        """
        super().__init__(max_help_position=55, **kwargs)

    def bulldozer_add_usage(
        self,
        positionals: list[argparse.Action] | None = None,
        optionals: list[argparse.Action] | None = None,
        commons: list[argparse.Action] | None = None,
        prefix: str | None = None,
    ) -> None:
        """
        This method generates the usage description.
        Args:
            positionals: list of all positional arguments as actions
                         (for bulldozer: only config file path expected).
            optionals: list of all Bulldozer optional arguments as actions.
            commons: list of all common optional arguments
                     (for example: help, version, long usage).
            prefix: string to describe the usage.
        """
        args = (positionals, optionals, commons, prefix)
        self._add_item(self._bulldozer_format_usage, args)

    def _bulldozer_format_usage(
        self,
        positionals: list[argparse.Action],
        optionals: list[argparse.Action],
        commons: list[argparse.Action],
        prefix: str,
    ) -> str:
        """
        This method defines the usage description format.
        Args:
            positionals: list of all positional arguments as actions
                         (for bulldozer: only config file path expected).
            optionals: list of all Bulldozer optional arguments as actions.
            commons: list of all common optional arguments
                     (for example: help, version, long usage).
            prefix: string to describe the usage.
        Returns:
            Usage description content as string.
        """
        # Define usages
        prog = f"{self._prog}"

        # Build full usage string
        format_usage = self._format_actions_usage
        commons_usage = format_usage(commons, [])
        usage = prog + " " + " ".join([param for param in [commons_usage] if param])

        if optionals:
            # Formatter used for CLI with parameters usage
            formatted_optionals = format_usage(optionals, []).split("] ")
            formatted_optionals = [arg + "]" for arg in formatted_optionals if arg]

            for it, param in enumerate(formatted_optionals):
                # Skip a line every 3 parameters
                if param:
                    if it % 3 == 0:
                        # Add spacing before optionals parameters for alignment purpose
                        usage += "\n" + (len(prog) + len(prefix) + 1) * " " + param
                    else:
                        usage += param

            # Remove last "]" char
            usage = usage[:-1]

            # Add spacing before positionals parameters for alignment purpose
            positionals_usage = format_usage(positionals, [])
            if positionals_usage:
                usage += "\n" + (len(prog) + len(prefix) + 1) * " " + positionals_usage
        else:
            # Formatter used for CLI with config file usage
            positionals_usage = format_usage(positionals, [])
            if positionals_usage:
                usage += " " + positionals_usage

        return prefix + usage + "\n\n"


class BulldozerArgumentParser(argparse.ArgumentParser):
    """
    Bulldozer arguments parser.
    """

    def __init__(self, **kwargs: Any):
        """
        BulldozerHelpFormatter constructor.
        """
        super().__init__(formatter_class=BulldozerHelpFormatter, **kwargs)

    def get_bulldozer_groups(self) -> tuple[list[argparse.Action], list[argparse.Action], list[argparse.Action]]:
        """
        This method returns the three groups of arguments containing
        the Bulldozer parameters (required, optional and expert).
        Returns:
            cli_positionals: Arguments group for required Bulldozer parameters.
            cli_optionals: Arguments group for optional Bulldozer parameters.
            cli_experts_optionals: Arguments group for expert Bulldozer parameters.
        Raises:
            ValueError: if unknown group description is found.
        """
        cli_positionals: list[argparse.Action] = []
        cli_optionals: list[argparse.Action] = []
        cli_experts_optionals: list[argparse.Action] = []

        for action_group in self._action_groups:
            if action_group.title is None:  # the group contains Bulldozer parameters
                if action_group.description == REQ_PARAM_KEY:
                    cli_positionals = action_group._group_actions
                elif action_group.description == OPT_PARAM_KEY:
                    cli_optionals = action_group._group_actions
                elif action_group.description == EXPERT_PARAM_KEY:
                    cli_experts_optionals = action_group._group_actions
                else:
                    raise ValueError(
                        f"Unknown group description {action_group.description}: "
                        f"expects {REQ_PARAM_KEY}, {OPT_PARAM_KEY} or "
                        f"{EXPERT_PARAM_KEY}"
                    )
        return cli_positionals, cli_optionals, cli_experts_optionals

    def print_help(
        self,
        file: Any = None,
        long_help: bool = False,
        expert_mode: bool = False,
    ) -> None:
        """
        This method generates the help message.
        Args:
            file: path to a file to save the output.
            long_help: whether print the full help or not (by default short help).
            expert_mode: whether print the expert parameters or not (by default False).
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

    def format_help(
        self,
        long_help: bool = False,
        expert_mode: bool = False,
        add_description: bool = True,
        add_epilog: bool = True,
    ) -> str:
        """
        This method generates the bulldozer global help message.
        It contains both execution methods: using a config file or using CLI arguments.
        Args:
            long_help: whether to display only usage description (False)
                       or full arguments descriptions (True).
            expert_mode: whether to display expert arguments or not.
            add_description: whether to add description or not at the beginning
                             of the help message.
            add_epilog: whether to add epilog or not at the end of the help message.
        Returns:
            The help message for Bulldozer.
        """
        # Get groups corresponding to bulldozer parameters
        cli_pos_group, cli_opt_group, cli_expert_group = self.get_bulldozer_groups()

        # Prepare input arguments for both config file and cli usages
        cli_positionals = cli_pos_group  # positional arguments for CLI
        # We set positionals to required (for visual display)
        for action in cli_positionals:
            action.required = True

        cli_optionals = cli_opt_group  # optional arguments for CLI
        if expert_mode:
            cli_optionals += cli_expert_group

        positionals = self._positionals._group_actions  # positional arguments (only config file path)
        # We set the nargs to None to highlight the requirement of this parameter
        if positionals:
            positionals[0].nargs = None

        # Format
        formatter = self._get_formatter()

        # Description
        if add_description:
            formatter.add_text(self.description)

        # Usage and help with config file
        formatter.bulldozer_add_usage(  # type: ignore
            positionals,
            commons=self._optionals._group_actions,
            prefix=_("Usage with config file: "),
        )

        if long_help:
            # Long help positional arguments
            formatter.start_section(self._positionals.title)
            formatter.add_text(self._positionals.description)
            formatter.add_arguments(self._positionals._group_actions)
            formatter.end_section()

            # Long help optional arguments
            formatter.start_section(self._optionals.title)
            formatter.add_text(self._optionals.description)
            formatter.add_arguments(self._optionals._group_actions)
            formatter.end_section()

        # Epilog
        if add_epilog:
            formatter.add_text(
                "If extra arguments are provided, these will override the original values from the configuration file."
            )

        formatter.add_text("---------------------------------")

        # Usage and help with cli arguments
        formatter.bulldozer_add_usage(  # type: ignore
            cli_positionals,
            cli_optionals,
            self._optionals._group_actions,
            prefix=_("Usage with parameters: "),
        )

        if long_help:
            # Long help positional arguments
            formatter.start_section("required arguments")
            formatter.add_arguments(cli_positionals)
            formatter.end_section()

            # Long help optional arguments
            formatter.start_section(self._optionals.title)
            formatter.add_arguments(self._optionals._group_actions)
            formatter.end_section()

            formatter.start_section(None)  # for visual spacing
            formatter.add_arguments(cli_optionals)
            formatter.end_section()

        # Epilog
        if add_epilog:
            if self.epilog:
                if long_help:
                    if not expert_mode:
                        # Only keep the expert mode Note
                        parts = self.epilog.split("\n", 1)
                        expert = parts[1] if len(parts) > 1 else self.epilog
                        expert = expert.replace("prog", f"{formatter._prog}")
                        expert = expert.replace("      ", "Note: ")
                        formatter.add_text(expert.replace("prog", f"{formatter._prog}"))
                    # TODO uncomment when doc is online
                    # formatter.add_text("For more details, consult https://bulldozer.readthedocs.io/")  # noqa: B950
                    formatter.add_text("For more details, consult the documentation.")
                else:
                    epilog = self.epilog.replace("prog", f"{formatter._prog}")
                    formatter.add_text(epilog)

        # Determine help from format above
        return formatter.format_help()
