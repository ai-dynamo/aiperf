# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregation-side record diagnostics must not fail a complete local run.

PROCESS_RECORDS_RESULT carries advisory errors -- a GPU-telemetry drain
timeout, one malformed record. Feeding them into ``_exit_errors`` made a local
no-GPU run exit 1 with an error panel while producing complete, correct
results. Errors the producer explicitly marks fatal are the exception: those
mean the artifact set itself is incomplete and must withhold the export
announcement.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.messages import ProcessRecordsResultMessage
from aiperf.common.models import ErrorDetails
from aiperf.common.models.record_models import ProcessRecordsResult
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceRunType
from aiperf.records.records_manager import ERROR_FATAL_DETAIL_KEY


def _build_controller(
    benchmark_run: BenchmarkRun, run_type: ServiceRunType
) -> SystemController:
    """Construct a SystemController with every external dependency mocked."""
    benchmark_run.cfg.runtime.service_run_type = run_type

    def mock_get_class(protocol: Any, name: Any) -> Any:
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
        param(ServiceRunType.KUBERNETES, True, id="kubernetes_fails_closed"),
    ],
)  # fmt: skip
async def test_on_process_records_result_message_advisory_errors_gated_by_run_type(
    benchmark_run: BenchmarkRun,
    run_type: ServiceRunType,
    expect_exit_errors: bool,
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
    benchmark_run: BenchmarkRun,
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


def _fatal_message() -> ProcessRecordsResultMessage:
    """A results message whose producer marked the error fatal."""
    return ProcessRecordsResultMessage.model_construct(
        service_id="records_manager",
        results=ProcessRecordsResult.model_construct(
            results=None,
            errors=[
                ErrorDetails(
                    type="OSError",
                    message="stream exporter failed to finalize",
                    details={ERROR_FATAL_DETAIL_KEY: True},
                )
            ],
        ),
    )


@pytest.mark.asyncio
async def test_fatal_result_errors_withhold_the_export_announcement(
    benchmark_run: BenchmarkRun,
) -> None:
    """A fatal aggregation error means the artifact set is incomplete.

    Announcing it as exported would publish a partial result set as if it were
    whole, so ``_export_failed`` must gate ResultsExportedMessage.
    """
    controller = _build_controller(benchmark_run, ServiceRunType.MULTIPROCESSING)
    controller._exit_errors = []
    controller._export_failed = False
    controller._merge_server_metric_phase_results = MagicMock()
    controller._result_join_coordinator = MagicMock()
    controller._check_and_trigger_shutdown = AsyncMock()

    await controller._on_process_records_result_message(_fatal_message())

    assert controller._export_failed is True
    assert [e.operation for e in controller._exit_errors] == ["process_records"]


def _explicitly_non_fatal_message() -> ProcessRecordsResultMessage:
    """The GPU-telemetry drain timeout: producer marked it explicitly non-fatal."""
    return ProcessRecordsResultMessage.model_construct(
        service_id="records_manager",
        results=ProcessRecordsResult.model_construct(
            results=None,
            errors=[
                ErrorDetails(
                    type="TimeoutError",
                    message="GPU telemetry drain did not complete",
                    details={ERROR_FATAL_DETAIL_KEY: False},
                )
            ],
        ),
    )


@pytest.mark.asyncio
async def test_kubernetes_explicitly_non_fatal_errors_do_not_set_exit_errors(
    benchmark_run: BenchmarkRun,
) -> None:
    """Under K8s the operator reads the exit code to mark the CR.

    An error the producer explicitly marked non-fatal (the GPU-telemetry drain
    timeout) must not push a complete, correctly-exported run to exit 1 and be
    stamped Failed. Errors that say nothing about fatality still report.
    """
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    controller._exit_errors = []
    controller._export_failed = False
    controller._merge_server_metric_phase_results = MagicMock()
    controller._result_join_coordinator = MagicMock()
    controller._check_and_trigger_shutdown = AsyncMock()
    controller.error = MagicMock()

    await controller._on_process_records_result_message(_explicitly_non_fatal_message())

    assert controller._exit_errors == []
    assert controller._export_failed is False
    assert any(
        "GPU telemetry drain" in str(call) for call in controller.error.call_args_list
    )


@pytest.mark.asyncio
async def test_advisory_errors_leave_the_export_announcement_alone(
    benchmark_run: BenchmarkRun,
) -> None:
    """The non-fatal sibling of the test above: diagnostics never gate export."""
    controller = _build_controller(benchmark_run, ServiceRunType.MULTIPROCESSING)
    controller._exit_errors = []
    controller._export_failed = False
    controller._merge_server_metric_phase_results = MagicMock()
    controller._result_join_coordinator = MagicMock()
    controller._check_and_trigger_shutdown = AsyncMock()

    await controller._on_process_records_result_message(_advisory_message())

    assert controller._export_failed is False
    assert controller._exit_errors == []
