# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import uuid

from aiperf.cli_utils import raise_startup_error_and_exit
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums.service_enums import ServiceType
from aiperf.common.factories import ServiceFactory
from aiperf.common.logging import setup_logging
from aiperf.common.types import ServiceTypeT
from aiperf.module_loader import ensure_modules_loaded


def run_service(
    service_type: ServiceTypeT,
    service_config: ServiceConfig,
    user_config: UserConfig,
    service_id: str | None = None,
    use_structured_subprocess_format: bool = True,
) -> None:
    """Run the specified service with the given configuration."""
    service_id = service_id or f"{service_type}_{uuid.uuid4().hex[:8]}"

    setup_logging(
        service_id=service_id,
        service_config=service_config,
        user_config=user_config,
        use_structured_subprocess_format=use_structured_subprocess_format,
    )

    _logger = AIPerfLogger(service_id)

    try:
        start_time = time.perf_counter()
        ensure_modules_loaded()
        end_time = time.perf_counter()
        _logger.info(f"AIPerf modules loaded in {end_time - start_time:.2f} seconds")
    except Exception as e:
        raise_startup_error_and_exit(
            f"Error loading modules: {e}",
            title="Error Loading Modules",
        )

    try:
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
        use_structured_subprocess_format=False,
    )
