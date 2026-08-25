# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregation-side record diagnostics must not fail a complete local run.

PROCESS_RECORDS_RESULT carries advisory errors -- a GPU-telemetry drain
timeout, one malformed record. Feeding them into ``_exit_errors`` made a local
no-GPU run exit 1 with an error panel while producing complete, correct
results. Under Kubernetes the operator reads that exit code to mark the CR, so
the fail-closed behavior has to survive there.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.messages import ProcessRecordsResultMessage
from aiperf.common.models import ErrorDetails
from aiperf.common.models.record_models import ProcessRecordsResult
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceRunType


def _build_controller(benchmark_run, run_type: ServiceRunType) -> SystemController:
    """Construct a SystemController with every external dependency mocked."""
    benchmark_run.cfg.runtime.service_run_type = run_type

    def mock_get_class(protocol, name):
        return lambda **kwargs: AsyncMock()

    with (
        patch(
            "aiperf.controller.system_controller.plugins.get_class",
            side_effect=mock_get_class,
        ),
        patch("aiperf.controller.system_controller.ProxyManager") as mock_proxy,
        patch(
            "aiperf.common.mixins.communication_mixin.plugins.get_class",
            side_effect=mock_get_class,
        ),
    ):  # fmt: skip
        mock_proxy.return_value = AsyncMock()
        return SystemController(run=benchmark_run, service_id="test_controller")


def _advisory_message() -> ProcessRecordsResultMessage:
    """A results message carrying only advisory diagnostics, no records."""
    return ProcessRecordsResultMessage.model_construct(
        service_id="records_manager",
        results=ProcessRecordsResult.model_construct(
            results=None,
            errors=[
                ErrorDetails(
                    type="TimeoutError",
                    message="GPU telemetry drain timed out",
                )
            ],
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run_type,expect_exit_errors",
    [
        param(ServiceRunType.MULTIPROCESSING, False, id="local_stays_advisory"),
    ],
)  # fmt: skip
async def test_on_process_records_result_message_advisory_errors_gated_by_run_type(
    benchmark_run, run_type: ServiceRunType, expect_exit_errors: bool
) -> None:
    controller = _build_controller(benchmark_run, run_type)
    controller._exit_errors = []
    controller._merge_server_metric_phase_results = MagicMock()
    controller._result_join_coordinator = MagicMock()
    controller._check_and_trigger_shutdown = AsyncMock()

    await controller._on_process_records_result_message(_advisory_message())

    assert bool(controller._exit_errors) is expect_exit_errors
    if expect_exit_errors:
        assert controller._exit_errors[0].operation == "process_records"


@pytest.mark.asyncio
async def test_on_process_records_result_message_advisory_errors_still_logged(
    benchmark_run,
) -> None:
    """Degrade the exit code, never the diagnostics."""
    controller = _build_controller(benchmark_run, ServiceRunType.MULTIPROCESSING)
    controller._exit_errors = []
    controller._merge_server_metric_phase_results = MagicMock()
    controller._result_join_coordinator = MagicMock()
    controller._check_and_trigger_shutdown = AsyncMock()
    controller.error = MagicMock()

    await controller._on_process_records_result_message(_advisory_message())

    assert controller._exit_errors == []
    assert any(
        "GPU telemetry drain timed out" in str(call)
        for call in controller.error.call_args_list
    )
