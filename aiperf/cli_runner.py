# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums import ServiceType
from aiperf.common.factories import ServiceFactory
from aiperf.common.logging import setup_logging
from aiperf.common.types import ServiceTypeT
from aiperf.module_loader import ensure_modules_loaded


def run_service(
    service_type: ServiceTypeT,
    service_config: ServiceConfig,
    user_config: UserConfig | None = None,
    service_id: str | None = None,
) -> None:
    """Run the specified service with the given configuration."""
    service_id = service_id or f"{service_type}_{uuid.uuid4().hex[:8]}"

    log_folder = (
        user_config.output.artifact_directory / OutputDefaults.LOG_FOLDER
        if user_config
        else None
    )

    setup_logging(
        service_type=service_type,
        service_id=service_id,
        level=service_config.log_level,
        log_folder=log_folder,
        debug_services=service_config.developer.debug_services,
        trace_services=service_config.developer.trace_services,
    )

    _logger = AIPerfLogger(service_id)

    try:
        ensure_modules_loaded()

        service_class = ServiceFactory.get_class_from_type(service_type)
        bootstrap_and_run_service(
            service_class,
            service_id=service_id,
            service_config=service_config,
            user_config=user_config,
        )
    except Exception:
        _logger.exception("Error running AIPerf Service")
        raise
    finally:
        _logger.debug("AIPerf Service exited")


def run_system_controller(
    user_config: UserConfig,
    service_config: ServiceConfig,
) -> None:
    """Run the system controller with the given configuration."""
    service_id = str(ServiceType.SYSTEM_CONTROLLER)
    _logger = AIPerfLogger(service_id)
    _logger.info("Starting AIPerf System")
    run_service(
        ServiceType.SYSTEM_CONTROLLER,
        service_config,
        user_config,
        service_id=service_id,
    )
