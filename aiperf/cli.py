# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main CLI entry point for the AIPerf system."""

import logging
import os
import sys
from pathlib import Path

import cyclopts
from pydantic import BaseModel, Field
from rich.console import Console
from rich.logging import RichHandler

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import ServiceRunType
from aiperf.services.system_controller.system_controller import SystemController

logger = logging.getLogger(__name__)


class CLIConfig(BaseModel):
    """Configuration model for CLI arguments."""

    config: Path | None = Field(
        default=None,
        description="Path to configuration file",
    )
    run_type: ServiceRunType = Field(
        default=ServiceRunType.MULTIPROCESSING,
        description="Process manager backend to use (multiprocessing: 'process', or kubernetes: 'k8s')",
    )
    user_config: UserConfig = Field(
        default=UserConfig(),
        description="User configuration",
    )


app = cyclopts.App(name="aiperf", help="AIPerf Benchmarking System")


def _setup_logging() -> None:
    """Set up rich logging with appropriate configuration."""
    # Set logging level for the root logger (affects all loggers)
    logging.root.setLevel(
        getattr(logging, os.getenv("AIPERF_LOG_LEVEL", "INFO").upper())
    )

    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=True,
        console=Console(),
        tracebacks_show_locals=False,
        log_time_format="%H:%M:%S.%f",
        omit_repeated_times=False,
    )
    logging.root.addHandler(rich_handler)


@app.default
def main(
    config: Path | None = None,
    run_type: ServiceRunType = ServiceRunType.MULTIPROCESSING,
    user_config: UserConfig | None = None,
) -> None:
    """Main entry point for the AIPerf system."""

    # Setup logging
    _setup_logging()

    # Create CLI config
    cli_config = CLIConfig(
        config=config,
        run_type=run_type,
        user_config=user_config or UserConfig(),
    )

    # Load configuration
    service_config = ServiceConfig(
        service_run_type=cli_config.run_type,
    )

    if cli_config.config:
        # In a real implementation, this would load from the specified file
        logger.debug("Loading configuration from %s", cli_config.config)
        # service_config.load_from_file(cli_config.config)

    # Create and start the system controller
    logger.info("Starting AIPerf System")

    bootstrap_and_run_service(
        SystemController,
        service_config=service_config,
        user_config=cli_config.user_config,
    )

    logger.info("AIPerf System exited")


if __name__ == "__main__":
    sys.exit(app())
