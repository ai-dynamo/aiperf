# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main CLI entry point for the AIPerf system."""

################################################################################
# NOTE: Keep the imports here to a minimum. This file is read every time
# the CLI is run, including to generate the help text. Any imports here
# will cause a performance penalty during this process.
################################################################################

import sys
from typing import Annotated

import cyclopts
from pydantic import Field

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.config.config_defaults import CLIDefaults

app = cyclopts.App(name="aiperf", help="NVIDIA AIPerf")


@app.command(name="profile")
def profile(
    user_config: UserConfig,
    service_config: ServiceConfig | None = None,
) -> None:
    """Run the Profile subcommand.

    Args:
        user_config: User configuration for the benchmark
        service_config: Service configuration options
    """
    from aiperf.cli_runner import run_system_controller

    service_config = service_config or ServiceConfig()

    run_system_controller(user_config, service_config)


@app.command(name="analyze")
def analyze(
    user_config: UserConfig,
    service_config: ServiceConfig | None = None,
) -> None:
    """Sweep through one or more parameters."""
    # TODO: Implement this
    from aiperf.cli_runner import warn_command_not_implemented

    warn_command_not_implemented("analyze")


@app.command(name="create-template", help="Create a template configuration file")
def create_template(
    template_filename: Annotated[
        str,
        Field(
            description=f"Path to the template file. Defaults to {CLIDefaults.TEMPLATE_FILENAME}."
        ),
        cyclopts.Parameter(
            name=("--template-filename", "-t"),
        ),
    ] = CLIDefaults.TEMPLATE_FILENAME,
) -> None:
    """Create a template configuration file."""
    # TODO: Implement this
    from aiperf.cli_runner import warn_command_not_implemented

    warn_command_not_implemented("create-template")


@app.command(name="validate", help="Validate the configuration file")
def validate(
    user_config: UserConfig | None = None,
    service_config: ServiceConfig | None = None,
) -> None:
    """Validate the configuration file."""
    # TODO: Implement this
    from aiperf.cli_runner import warn_command_not_implemented

    warn_command_not_implemented("validate")


if __name__ == "__main__":
    sys.exit(app())
