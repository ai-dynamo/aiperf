#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.config import ServiceConfig, UserConfig


def run_system_controller(
    user_config: UserConfig,
    service_config: ServiceConfig | None = None,
) -> None:
    """Run the system controller with the given configuration."""

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.services import SystemController

    logger = AIPerfLogger(__name__)

    service_config = service_config or ServiceConfig()

    log_queue = None
    if service_config.disable_ui:
        from aiperf.common.logging import setup_rich_logging

        setup_rich_logging(user_config, service_config)

    else:
        from aiperf.common.logging import get_global_log_queue

        log_queue = get_global_log_queue()

    # Create and start the system controller
    logger.info("Starting AIPerf System")

    try:
        bootstrap_and_run_service(
            SystemController,
            service_id="system_controller",
            service_config=service_config,
            user_config=user_config,
            log_queue=log_queue,
        )
    except Exception:
        logger.exception("Error starting AIPerf System")
        raise
    finally:
        logger.info("AIPerf System exited")


def warn_command_not_implemented(command: str) -> None:
    """Warn the user that the subcommand is not implemented."""
    import sys

    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(
        Panel(
            f"Command [bold red]{command}[/bold red] is not yet implemented",
            title="Error",
            title_align="left",
            border_style="red",
        )
    )

    sys.exit(1)
