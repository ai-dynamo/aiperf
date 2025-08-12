# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.cli_utils import raise_startup_error_and_exit
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums.ui_enums import AIPerfUIType


def run_system_controller(
    user_config: UserConfig,
    service_config: ServiceConfig,
) -> None:
    """Run the system controller with the given configuration."""

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.common.logging import get_global_log_queue
    from aiperf.controller import SystemController
    from aiperf.module_loader import ensure_modules_loaded

    logger = AIPerfLogger(__name__)

    log_queue = None
    if service_config.ui_type == AIPerfUIType.DASHBOARD:
        log_queue = get_global_log_queue()
    else:
        from aiperf.common.logging import setup_rich_logging

        setup_rich_logging(user_config, service_config)

    # Create and start the system controller
    logger.info("Starting AIPerf System")

    try:
        ensure_modules_loaded()
    except Exception as e:
        raise_startup_error_and_exit(
            f"Error loading modules: {e}",
            title="Error Loading Modules",
        )

    try:
        bootstrap_and_run_service(
            SystemController,
            service_id="system_controller",
            service_config=service_config,
            user_config=user_config,
            log_queue=log_queue,
        )
    except Exception:
        logger.exception("Error running AIPerf System")
        raise
    finally:
        logger.debug("AIPerf System exited")
