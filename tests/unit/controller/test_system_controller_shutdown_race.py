# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for result-barrier readiness during startup."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import SystemState
from aiperf.controller.result_join_coordinator import ResultJoinCoordinator
from aiperf.controller.system_controller import SystemController


def _controller(state: SystemState) -> MagicMock:
    """Build a controller stub with the real shutdown check bound."""
    controller = MagicMock()
    controller._system_state = state
    controller._shutdown_triggered = False
    controller._result_join_coordinator = ResultJoinCoordinator()
    controller.debug = MagicMock()
    controller.info = MagicMock()
    controller.stop = AsyncMock()
    controller._set_system_state = AsyncMock()
    controller._check_and_trigger_shutdown = (
        SystemController._check_and_trigger_shutdown.__get__(controller)
    )
    return controller


class TestVacuousResultBarrier:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "state",
        [
            SystemState.INITIALIZING,
            SystemState.CONFIGURING,
            SystemState.READY,
        ],
    )
    async def test_empty_barrier_before_profiling_does_not_shut_down(
        self, state: SystemState
    ) -> None:
        controller = _controller(state)
        controller._result_join_coordinator.register(
            "telemetry", "gpu_telemetry_manager"
        )
        controller._result_join_coordinator.unregister(
            "telemetry", "gpu_telemetry_manager"
        )
        assert controller._result_join_coordinator.ready

        await controller._check_and_trigger_shutdown()

        controller.stop.assert_not_awaited()
        assert controller._shutdown_triggered is False

    @pytest.mark.asyncio
    async def test_satisfied_barrier_after_profiling_shuts_down(self) -> None:
        controller = _controller(SystemState.PROCESSING)
        controller._result_join_coordinator.register("profile", "records_manager")
        controller._result_join_coordinator.complete("profile", "records_manager")

        await controller._check_and_trigger_shutdown()
        controller.stop.assert_awaited_once()
        assert controller._shutdown_triggered is True

    @pytest.mark.asyncio
    async def test_all_producers_dying_after_profiling_shuts_down(self) -> None:
        controller = _controller(SystemState.PROFILING)
        controller._result_join_coordinator.register("profile", "records_manager")
        controller._result_join_coordinator.unregister_service("records_manager")

        await controller._check_and_trigger_shutdown()
        controller.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_result_domain_waits_start_concurrently() -> None:
    """Each result wait needs its peer to start before it can finish."""
    controller = _controller(SystemState.PROCESSING)
    controller._result_join_coordinator.register("accuracy", "records_manager")
    controller._result_join_coordinator.register("server_metrics", "metrics_manager")
    accuracy_started = asyncio.Event()
    metrics_started = asyncio.Event()

    async def wait_for_accuracy() -> None:
        accuracy_started.set()
        await metrics_started.wait()

    async def wait_for_metrics() -> None:
        metrics_started.set()
        await accuracy_started.wait()

    controller._await_accuracy_results_for_cancel = wait_for_accuracy
    controller._await_server_metrics_results_for_cancel = wait_for_metrics
    controller._await_cancel_result_domains = (
        SystemController._await_cancel_result_domains.__get__(controller)
    )

    await asyncio.wait_for(controller._await_cancel_result_domains(True), timeout=0.1)
