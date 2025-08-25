# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main CLI entry point for the AIPerf system."""

################################################################################
# NOTE: Keep the imports here to a minimum. This file is read every time
# the CLI is run, including to generate the help text. Any imports here
# will cause a performance penalty during this process.
################################################################################

import sys

from cyclopts import App

from aiperf.cli_utils import exit_on_error
from aiperf.common.config import NodeConfig, ServiceConfig, UserConfig

app = App(name="aiperf", help="NVIDIA AIPerf")


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
    with exit_on_error(title="Error Running AIPerf System"):
        from aiperf.cli_runner import run_system_controller
        from aiperf.common.config import load_service_config

        service_config = service_config or load_service_config()

        run_system_controller(user_config, service_config)


@app.command(name="node")
def node(
    node_config: NodeConfig,
    service_config: ServiceConfig | None = None,
) -> None:
    """Start an AIPerf node with the given ID."""
    with exit_on_error(title="Error Running AIPerf Node"):
        from aiperf.cli_runner import run_node_controller
        from aiperf.common.config import load_service_config

        service_config = service_config or load_service_config()

        run_node_controller(service_config, node_config)


if __name__ == "__main__":
    sys.exit(app())
