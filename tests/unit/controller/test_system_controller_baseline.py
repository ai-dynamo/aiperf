# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SystemController-level baseline registration wiring tests."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import (
    LifecycleState,
    ServiceCapability,
    make_result_producer_capability,
)
from aiperf.common.messages import (
    BaseServiceErrorMessage,
    ProcessRecordsResultMessage,
    ProcessServerMetricsResultMessage,
    ProcessTelemetryResultMessage,
    RegisterServiceCommand,
    ServerMetricsStatusMessage,
    TelemetryStatusMessage,
)
from aiperf.common.models import (
    ProcessRecordsResult,
    ProcessServerMetricsResult,
    ProcessTelemetryResult,
    ProfileResults,
    ServerMetricsResults,
    TelemetryExportData,
    TelemetrySummary,
)
from aiperf.common.models.error_models import ErrorDetails
from aiperf.controller.baseline_coordinator import BaselineCoordinator
from aiperf.controller.result_join_coordinator import ResultJoinCoordinator
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType


async def _no_op_publish(msg) -> None:
    return None


class _ServiceManagerStub:
    def __init__(self) -> None:
        self.service_id_map = {}
        self.service_map = {}


class _BaselineCoordinatorStub:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def register(self, service_id: str) -> None:
        self.registered.append(service_id)


def _controller_for_registration() -> SystemController:
    controller = SystemController.__new__(SystemController)
    controller.service_manager = _ServiceManagerStub()
    controller._baseline_coordinator = _BaselineCoordinatorStub()
    controller._result_join_coordinator = ResultJoinCoordinator()
    controller.debug = lambda _message: None
    controller.info = lambda _message: None
    return controller


def _controller_for_shutdown() -> SystemController:
    controller = SystemController.__new__(SystemController)
    controller._result_join_coordinator = ResultJoinCoordinator()
    controller._profile_results = None
    controller._telemetry_results = None
    controller._server_metrics_results = None
    controller._shutdown_triggered = False
    controller._shutdown_lock = asyncio.Lock()
    controller._exit_errors = []
    controller._telemetry_endpoints_configured = []
    controller._telemetry_endpoints_reachable = []
    controller._server_metrics_endpoints_configured = []
    controller._server_metrics_endpoints_reachable = []
    controller.info_messages = []
    controller.info = controller.info_messages.append
    controller.debug = lambda _message: None
    controller.error = lambda _message: None
    controller.exception = lambda _message: None
    controller.trace_or_debug = lambda _trace_message, _debug_message: None
    controller.stop = AsyncMock()
    return controller


def _profile_result_message(
    service_id: str = "records-1",
) -> ProcessRecordsResultMessage:
    return ProcessRecordsResultMessage(
        service_id=service_id,
        results=ProcessRecordsResult(
            results=ProfileResults(
                records=[],
                completed=0,
                start_ns=1,
                end_ns=2,
            )
        ),
    )


def _telemetry_result_message(
    service_id: str = "records-manager",
) -> ProcessTelemetryResultMessage:
    return ProcessTelemetryResultMessage(
        service_id=service_id,
        telemetry_result=ProcessTelemetryResult(
            results=TelemetryExportData(
                summary=TelemetrySummary(
                    start_time=datetime.fromtimestamp(1, tz=UTC),
                    end_time=datetime.fromtimestamp(2, tz=UTC),
                ),
                endpoints={},
            )
        ),
    )


def _server_metrics_result_message(
    service_id: str = "records-manager",
) -> ProcessServerMetricsResultMessage:
    return ProcessServerMetricsResultMessage(
        service_id=service_id,
        server_metrics_result=ProcessServerMetricsResult(
            results=ServerMetricsResults(start_ns=1, end_ns=2)
        ),
    )


def test_coordinator_registers_baseline_collector() -> None:
    coord = BaselineCoordinator(publish=_no_op_publish, gate_timeout_s=0.05)
    cmd = RegisterServiceCommand(
        command_id="c1",
        service_id="svc-a",
        service_type=ServiceType.GPU_TELEMETRY_MANAGER,
        state=LifecycleState.RUNNING,
        capabilities=(ServiceCapability.BASELINE_COLLECTOR,),
    )
    if ServiceCapability.BASELINE_COLLECTOR in cmd.capabilities:
        coord.register(cmd.service_id)
    assert coord.registered_count == 1


def test_coordinator_skips_service_without_capability() -> None:
    coord = BaselineCoordinator(publish=_no_op_publish, gate_timeout_s=0.05)
    cmd = RegisterServiceCommand(
        command_id="c1",
        service_id="svc-a",
        service_type=ServiceType.WORKER,
        state=LifecycleState.RUNNING,
    )
    if ServiceCapability.BASELINE_COLLECTOR in cmd.capabilities:
        coord.register(cmd.service_id)
    assert coord.registered_count == 0


@pytest.mark.asyncio
async def test_register_service_registers_result_producer_domain() -> None:
    controller = _controller_for_registration()
    cmd = RegisterServiceCommand(
        command_id="c1",
        service_id="records-1",
        service_type=ServiceType.RECORDS_MANAGER,
        state=LifecycleState.RUNNING,
        capabilities=(make_result_producer_capability("profile"),),
    )

    await controller._handle_register_service_command(cmd)

    assert controller._result_join_coordinator.pending_domains == ("profile",)


@pytest.mark.asyncio
async def test_register_service_ignores_unknown_result_capabilities() -> None:
    controller = _controller_for_registration()
    cmd = RegisterServiceCommand(
        command_id="c1",
        service_id="worker-1",
        service_type=ServiceType.WORKER,
        state=LifecycleState.RUNNING,
        capabilities=("result_producer", "result_producer:", "unknown:domain"),
    )

    await controller._handle_register_service_command(cmd)

    assert controller._result_join_coordinator.ready
    assert controller._baseline_coordinator.registered == []


@pytest.mark.asyncio
async def test_register_service_preserves_baseline_collector_registration() -> None:
    controller = _controller_for_registration()
    cmd = RegisterServiceCommand(
        command_id="c1",
        service_id="telemetry-1",
        service_type=ServiceType.GPU_TELEMETRY_MANAGER,
        state=LifecycleState.RUNNING,
        capabilities=(
            ServiceCapability.BASELINE_COLLECTOR,
            make_result_producer_capability("telemetry"),
        ),
    )

    await controller._handle_register_service_command(cmd)

    assert controller._baseline_coordinator.registered == ["telemetry-1"]
    assert controller._result_join_coordinator.pending_domains == ("telemetry",)


@pytest.mark.asyncio
async def test_shutdown_waits_for_registered_result_domains_with_deduped_logs() -> None:
    controller = _controller_for_shutdown()
    controller._result_join_coordinator.register("profile", "records-manager")
    controller._result_join_coordinator.register("telemetry", "telemetry-manager")
    controller._result_join_coordinator.register(
        "server_metrics", "server-metrics-manager"
    )

    await controller._on_process_records_result_message(_profile_result_message())

    assert controller.stop.await_count == 0
    assert controller.info_messages == [
        "Waiting for result domains: server_metrics, telemetry"
    ]

    await controller._on_process_telemetry_result_message(_telemetry_result_message())

    assert controller.stop.await_count == 0
    assert controller.info_messages == [
        "Waiting for result domains: server_metrics, telemetry",
        "Waiting for result domains: server_metrics",
    ]

    await controller._on_process_server_metrics_result_message(
        _server_metrics_result_message()
    )

    assert controller.info_messages == [
        "Waiting for result domains: server_metrics, telemetry",
        "Waiting for result domains: server_metrics",
        "All results received, initiating shutdown",
    ]
    controller.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_pending_check_does_not_repeat_unchanged_wait_log() -> None:
    controller = _controller_for_shutdown()
    controller._result_join_coordinator.register("profile", "records-manager")
    controller._result_join_coordinator.register("telemetry", "telemetry-manager")

    await controller._on_process_records_result_message(_profile_result_message())
    await controller._check_and_trigger_shutdown()

    assert controller.info_messages == ["Waiting for result domains: telemetry"]
    assert controller.stop.await_count == 0


@pytest.mark.asyncio
async def test_service_error_unregisters_failed_result_producer_and_allows_shutdown() -> (
    None
):
    controller = _controller_for_shutdown()
    controller._result_join_coordinator.register("profile", "records-manager")
    controller._result_join_coordinator.register("telemetry", "telemetry-manager")

    await controller._on_process_records_result_message(
        _profile_result_message("records-manager")
    )
    await controller._process_service_error_message(
        BaseServiceErrorMessage(
            service_id="telemetry-manager",
            error=ErrorDetails(type="RuntimeError", message="telemetry failed"),
        )
    )

    assert controller._exit_errors[0].service_id == "telemetry-manager"
    assert controller._exit_errors[0].error_details.message == "telemetry failed"
    assert controller._result_join_coordinator.pending_domains == ()
    assert controller.info_messages == [
        "Waiting for result domains: telemetry",
        "All results received, initiating shutdown",
    ]
    controller.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_telemetry_status_unregisters_result_domain_and_allows_shutdown() -> (
    None
):
    controller = _controller_for_shutdown()
    controller._result_join_coordinator.register("profile", "records-manager")
    controller._result_join_coordinator.register("telemetry", "telemetry-manager")

    await controller._on_process_records_result_message(
        _profile_result_message("records-manager")
    )
    await controller._on_telemetry_status_message(
        TelemetryStatusMessage(
            service_id="telemetry-manager",
            enabled=False,
            reason="no DCGM endpoints reachable",
            endpoints_configured=[],
            endpoints_reachable=[],
        )
    )

    assert controller._result_join_coordinator.pending_domains == ()
    assert controller.info_messages == [
        "Waiting for result domains: telemetry",
        "DCGM telemetry skipped: no DCGM endpoints reachable",
        "All results received, initiating shutdown",
    ]
    controller.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_server_metrics_status_unregisters_result_domain_and_allows_shutdown() -> (
    None
):
    controller = _controller_for_shutdown()
    controller._result_join_coordinator.register("profile", "records-manager")
    controller._result_join_coordinator.register(
        "server_metrics", "server-metrics-manager"
    )

    await controller._on_process_records_result_message(
        _profile_result_message("records-manager")
    )
    await controller._on_server_metrics_status_message(
        ServerMetricsStatusMessage(
            service_id="server-metrics-manager",
            enabled=False,
            reason="no Prometheus endpoints reachable",
            endpoints_configured=[],
            endpoints_reachable=[],
        )
    )

    assert controller._result_join_coordinator.pending_domains == ()
    assert controller.info_messages == [
        "Waiting for result domains: server_metrics",
        "Server metrics disabled - no Prometheus endpoints reachable",
        "All results received, initiating shutdown",
    ]
    controller.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_triggers_when_no_result_producers_registered() -> None:
    controller = _controller_for_shutdown()

    await controller._check_and_trigger_shutdown()

    assert controller.info_messages == ["All results received, initiating shutdown"]
    controller.stop.assert_awaited_once()
