# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest
from pytest import param

from aiperf.common.control_structs import Registration
from aiperf.common.enums import (
    LifecycleState,
    SystemState,
    make_result_producer_capability,
)
from aiperf.common.messages import (
    BaseServiceErrorMessage,
    ProcessServerMetricsResultMessage,
)
from aiperf.common.models import ErrorDetails, ErrorDetailsCount
from aiperf.common.models.server_metrics_models import (
    ProcessServerMetricsResult,
    ServerMetricsResults,
)
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType


def registration(instance_id: str) -> Registration:
    """Build a registration from one metrics-sidecar process epoch."""
    return Registration(
        sid="server_metrics_manager",
        rid=f"request-{instance_id}",
        stype=ServiceType.SERVER_METRICS_MANAGER,
        state=LifecycleState.RUNNING,
        instance_id=instance_id,
        capabilities=(make_result_producer_capability("server_metrics"),),
    )


@pytest.mark.asyncio
async def test_restart_releases_metrics_barrier_without_cancelling_requests(
    system_controller: SystemController,
) -> None:
    """A fast container restart cannot masquerade as the old live result producer."""
    controller = system_controller
    controller._check_and_trigger_shutdown = AsyncMock()
    controller._cancel_profiling = AsyncMock()
    controller._on_registration(registration("first-process"))
    controller._result_join_coordinator.register("records", "records-manager")
    controller._system_state = SystemState.PROFILING
    controller._on_registration(registration("second-process"))
    await controller.wait_for_tasks()
    assert controller._result_join_coordinator.pending_domains == ("records",)
    assert "server_metrics_manager" in controller._retired_result_producer_ids
    assert (
        controller._server_metrics_results.error_summary[0].error_details.type
        == "ServerMetricsRestarted"
    )
    controller._cancel_profiling.assert_not_awaited()
    assert controller._system_state == SystemState.PROFILING
    controller._on_registration(registration("third-process"))
    assert len(controller._exit_errors) == 1
    assert controller._result_join_coordinator.pending_domains == ("records",)


@pytest.mark.asyncio
async def test_reconnect_same_process_preserves_metrics_barrier(
    system_controller: SystemController,
) -> None:
    """A connection retry carries the same process ID and loses no telemetry."""
    controller = system_controller
    controller._on_registration(registration("first-process"))
    controller._system_state = SystemState.PROFILING
    controller._on_registration(registration("first-process"))
    assert controller._result_join_coordinator.pending_domains == ("server_metrics",)
    assert not controller._retired_result_producer_ids
    assert not controller._exit_errors


@pytest.mark.asyncio
async def test_restart_after_results_does_not_rearm_barrier(
    system_controller: SystemController,
) -> None:
    """A sidecar that already reported cannot delay or degrade completed results."""
    controller = system_controller
    controller._check_and_trigger_shutdown = AsyncMock()
    controller._on_registration(registration("first-process"))
    controller._result_join_coordinator.complete_domain("server_metrics")
    controller._system_state = SystemState.PROFILING
    controller._on_registration(registration("second-process"))
    await controller.wait_for_tasks()
    assert controller._result_join_coordinator.ready
    assert not controller._exit_errors


@pytest.mark.asyncio
async def test_reaped_metrics_report_failure_and_do_not_rejoin_on_restart(
    system_controller: SystemController,
) -> None:
    """A sidecar that stays dead still permits an honest inference export."""
    controller = system_controller
    controller._check_and_trigger_shutdown = AsyncMock()
    controller._cancel_profiling = AsyncMock()
    controller._on_registration(registration("first-process"))
    first_seen_ns = ServiceRegistry.get_service("server_metrics_manager").first_seen_ns
    controller._system_state = SystemState.PROFILING
    await controller._on_service_reaped(
        "server_metrics_manager", "heartbeat expired", first_seen_ns
    )
    assert controller._result_join_coordinator.ready
    assert (
        controller._server_metrics_results.error_summary[0].error_details.type
        == "ProducerReaped"
    )
    controller._cancel_profiling.assert_not_awaited()
    controller._on_registration(registration("second-process"))
    await controller.wait_for_tasks()
    assert controller._result_join_coordinator.ready
    assert len(controller._exit_errors) == 1


@pytest.mark.asyncio
async def test_late_result_cannot_erase_restart_failure(
    system_controller: SystemController,
) -> None:
    """Queued messages from the old process must not hide missing telemetry."""
    controller = system_controller
    controller._check_and_trigger_shutdown = AsyncMock()
    controller._on_registration(registration("first-process"))
    controller._system_state = SystemState.PROFILING
    controller._on_registration(registration("second-process"))
    await controller.wait_for_tasks()
    await controller._on_process_server_metrics_result_message(
        ProcessServerMetricsResultMessage(
            service_id="server_metrics_manager",
            server_metrics_result=ProcessServerMetricsResult(results=None),
        )
    )
    assert (
        controller._server_metrics_results.error_summary[0].error_details.type
        == "ServerMetricsRestarted"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type,exit_error_count",
    [
        param("ServerMetricsCollectionStopped", 1, id="terminal-resource-failure"),
        param("TimeoutError", 0, id="recoverable-scrape-error"),
    ],
)  # fmt: skip
async def test_collection_resource_failure_reports_degraded_run_without_cancel(
    system_controller: SystemController, error_type: str, exit_error_count: int
) -> None:
    """A terminal budget failure must export and report a degraded outcome."""
    controller = system_controller
    controller._check_and_trigger_shutdown = AsyncMock()
    controller._cancel_profiling = AsyncMock()
    controller._system_state = SystemState.PROFILING
    await controller._on_process_server_metrics_result_message(
        ProcessServerMetricsResultMessage(
            service_id="server_metrics_manager",
            server_metrics_result=ProcessServerMetricsResult(
                results=ServerMetricsResults(
                    benchmark_id=controller.run.benchmark_id,
                    start_ns=0,
                    end_ns=0,
                    error_summary=[
                        ErrorDetailsCount(
                            error_details=ErrorDetails(
                                type=error_type,
                                message="Server metrics collection failed",
                            ),
                            count=1,
                        )
                    ],
                )
            ),
        )
    )
    assert len(controller._exit_errors) == exit_error_count
    controller._cancel_profiling.assert_not_awaited()
    assert controller._system_state == SystemState.PROFILING


@pytest.mark.asyncio
@pytest.mark.parametrize("next_instance", ["first-process", "second-process"])
async def test_service_error_preserves_telemetry_failure_through_reconnect(
    system_controller: SystemController, next_instance: str
) -> None:
    """Reconnects and late results cannot erase a reported telemetry failure."""
    controller = system_controller
    controller._check_and_trigger_shutdown = AsyncMock()
    controller._cancel_profiling = AsyncMock()
    controller._on_registration(registration("first-process"))
    controller._result_join_coordinator.register("records", "records-manager")
    controller._system_state = SystemState.PROFILING
    error = ErrorDetails(type="RuntimeError", message="Metrics process failed")
    message = BaseServiceErrorMessage(service_id="server_metrics_manager", error=error)

    await controller._process_service_error_message(message)
    assert controller._server_metrics_results is not None
    assert controller._server_metrics_result_arrived.is_set()
    assert controller._result_join_coordinator.pending_domains == ("records",)
    controller._cancel_profiling.assert_not_awaited()
    assert controller._system_state == SystemState.PROFILING

    await controller._process_service_error_message(message)
    controller._on_registration(registration(next_instance))
    await controller.wait_for_tasks()
    await controller._on_process_server_metrics_result_message(
        ProcessServerMetricsResultMessage(
            service_id="server_metrics_manager",
            server_metrics_result=ProcessServerMetricsResult(results=None),
        )
    )
    assert controller._result_join_coordinator.pending_domains == ("records",)
    assert controller._server_metrics_results.error_summary == [
        ErrorDetailsCount(error_details=error, count=1)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("producer_state", ["disabled", "complete"])
async def test_service_error_does_not_degrade_nonpending_telemetry(
    system_controller: SystemController, producer_state: str
) -> None:
    """Disabled or already-delivered telemetry is not missing after an error."""
    controller = system_controller
    controller._check_and_trigger_shutdown = AsyncMock()
    controller._cancel_profiling = AsyncMock()
    controller._on_registration(registration("first-process"))
    if producer_state == "disabled":
        controller._result_join_coordinator.unregister(
            "server_metrics", "server_metrics_manager"
        )
    else:
        controller._result_join_coordinator.complete_domain("server_metrics")
    controller._system_state = SystemState.PROFILING
    original_results = controller._server_metrics_results

    await controller._process_service_error_message(
        BaseServiceErrorMessage(
            service_id="server_metrics_manager",
            error=ErrorDetails(type="RuntimeError", message="Metrics process failed"),
        )
    )
    assert controller._server_metrics_results is original_results
    assert not controller._result_join_coordinator.evicted
    controller._cancel_profiling.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("results_delivered", [False, True])
async def test_late_service_error_after_reap_does_not_cancel_inference(
    system_controller: SystemController,
    results_delivered: bool,
) -> None:
    """A queued optional-service error keeps its identity after command-map cleanup."""
    controller = system_controller
    controller._check_and_trigger_shutdown = AsyncMock()
    controller._cancel_profiling = AsyncMock()
    controller._on_registration(registration("first-process"))
    controller._result_join_coordinator.register("records", "records-manager")
    if results_delivered:
        controller._result_join_coordinator.complete_domain("server_metrics")
    first_seen_ns = ServiceRegistry.get_service("server_metrics_manager").first_seen_ns
    controller._system_state = SystemState.PROFILING
    await controller._on_service_reaped(
        "server_metrics_manager", "process exited", first_seen_ns
    )
    assert "server_metrics_manager" not in controller.service_manager.service_id_map
    retained_results = controller._server_metrics_results

    await controller._process_service_error_message(
        BaseServiceErrorMessage(
            service_id="server_metrics_manager",
            error=ErrorDetails(message="Queued metrics process error"),
        )
    )

    controller._cancel_profiling.assert_not_awaited()
    assert controller._system_state == SystemState.PROFILING
    assert controller._result_join_coordinator.pending_domains == ("records",)
    assert controller._server_metrics_results is retained_results
