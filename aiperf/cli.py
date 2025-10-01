# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main CLI entry point for the AIPerf system."""

################################################################################
# NOTE: Keep the imports here to a minimum. This file is read every time
# the CLI is run, including to generate the help text. Any imports here
# will cause a performance penalty during this process.
################################################################################

import sys
from pathlib import Path

from cyclopts import App

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import ServiceType

app = App(name="aiperf", help="NVIDIA AIPerf")


@app.command(name="profile")
def profile(
    user_config: UserConfig,
    service_config: ServiceConfig | None = None,
) -> None:
    """Profile an inference server.

    Args:
        user_config: User configuration for the benchmark
        service_config: Service configuration options
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title="Error Running AIPerf System"):
        from aiperf.cli_runner import run_system_controller
        from aiperf.common.config import load_service_config

        service_config = service_config or load_service_config()

        run_system_controller(user_config, service_config)


@app.command(name="service")
def service(
    service_type: ServiceType,
    user_config_file: Path,
    service_config_file: Path,
    service_id: str | None = None,
) -> None:
    """Run a specific AIPerf service.

    Args:
        service_type: Type of the service to run
        user_config_file: Path to user configuration JSON file
        service_config_file: Path to service configuration JSON file
        service_id: Service ID (auto-generated if not provided)
    """

    from aiperf.cli_utils import exit_on_error

    with exit_on_error(
        title=f"Error Running AIPerf {service_type.name.replace('_', ' ').title()} Service"
    ):
        import uuid

        from aiperf.cli_runner import run_service

        # TODO: Eventually this may be moved over to be handled as part of Profile_Configure command
        user_config = UserConfig.model_validate_json(user_config_file.read_text())
        service_config = ServiceConfig.model_validate_json(
            service_config_file.read_text()
        )

        run_service(
            service_type,
            service_config=service_config,
            user_config=user_config,
            service_id=service_id or f"{service_type}_{uuid.uuid4().hex[:8]}",
        )


if __name__ == "__main__":
    sys.exit(app())
