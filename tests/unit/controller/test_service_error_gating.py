# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only a required service's self-reported death may cancel the benchmark.

``BaseService._kill`` publishes ``BaseServiceErrorMessage`` from *every*
service, including optional ones (GPU telemetry, server metrics), whose loss
degrades a run rather than invalidating it.
"""

from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import LifecycleState, ServiceRegistrationStatus, SystemState
from aiperf.common.messages import BaseServiceErrorMessage
from aiperf.common.models import ErrorDetails, ServiceRunInfo
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType


def _track(
    system_controller: SystemController, service_id: str, service_type: ServiceType
) -> None:
    system_controller.service_manager.service_id_map = {
        service_id: ServiceRunInfo(
            service_id=service_id,
            service_type=service_type,
            registration_status=ServiceRegistrationStatus.REGISTERED,
            state=LifecycleState.RUNNING,
        )
    }


@pytest.mark.asyncio
async def test_optional_service_error_does_not_cancel_profiling(
    system_controller: SystemController,
) -> None:
    """A failing GPU telemetry manager must not kill an in-flight benchmark."""
    _track(
        system_controller, "gpu_telemetry_manager_0", ServiceType.GPU_TELEMETRY_MANAGER
    )
    system_controller._system_state = SystemState.PROFILING
    system_controller._cancel_profiling = AsyncMock()
    system_controller._check_and_trigger_shutdown = AsyncMock()

    await system_controller._process_service_error_message(
        BaseServiceErrorMessage(
            service_id="gpu_telemetry_manager_0",
            error=ErrorDetails(message="DCGM scrape failed"),
        )
    )

    system_controller._cancel_profiling.assert_not_awaited()
    system_controller._check_and_trigger_shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_required_service_error_cancels_profiling(
    system_controller: SystemController,
) -> None:
    """The required-service path keeps its fail-fast cancellation."""
    _track(system_controller, "records_manager_0", ServiceType.RECORDS_MANAGER)
    system_controller._system_state = SystemState.PROFILING
    system_controller._cancel_profiling = AsyncMock()
    system_controller._check_and_trigger_shutdown = AsyncMock()

    await system_controller._process_service_error_message(
        BaseServiceErrorMessage(
            service_id="records_manager_0",
            error=ErrorDetails(message="records manager died"),
        )
    )

    system_controller._cancel_profiling.assert_awaited_once()


@pytest.mark.asyncio
async def test_unknown_service_error_cancels_profiling(
    system_controller: SystemController,
) -> None:
    """An unidentifiable sender is treated as fatal rather than ignored."""
    system_controller.service_manager.service_id_map = {}
    system_controller._system_state = SystemState.PROFILING
    system_controller._cancel_profiling = AsyncMock()
    system_controller._check_and_trigger_shutdown = AsyncMock()

    await system_controller._process_service_error_message(
        BaseServiceErrorMessage(
            service_id="mystery-service",
            error=ErrorDetails(message="unknown sender"),
        )
    )

    system_controller._cancel_profiling.assert_awaited_once()


@pytest.mark.asyncio
async def test_optional_service_error_is_still_recorded_as_exit_error(
    system_controller: SystemController,
) -> None:
    """Not cancelling must not mean silently swallowing the failure."""
    _track(
        system_controller,
        "server_metrics_manager_0",
        ServiceType.SERVER_METRICS_MANAGER,
    )
    system_controller._system_state = SystemState.PROFILING
    system_controller._cancel_profiling = AsyncMock()
    system_controller._check_and_trigger_shutdown = AsyncMock()

    await system_controller._process_service_error_message(
        BaseServiceErrorMessage(
            service_id="server_metrics_manager_0",
            error=ErrorDetails(message="scrape endpoint gone"),
        )
    )

    assert [error.service_id for error in system_controller._exit_errors] == [
        "server_metrics_manager_0"
    ]
