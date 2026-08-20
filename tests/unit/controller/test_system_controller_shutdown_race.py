# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for result-barrier readiness during startup."""

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
    controller._finalize_kubernetes_raw_artifacts = AsyncMock()
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

        controller._finalize_kubernetes_raw_artifacts.assert_awaited_once()
        controller.stop.assert_awaited_once()
        assert controller._shutdown_triggered is True

    @pytest.mark.asyncio
    async def test_all_producers_dying_after_profiling_shuts_down(self) -> None:
        controller = _controller(SystemState.PROFILING)
        controller._result_join_coordinator.register("profile", "records_manager")
        controller._result_join_coordinator.unregister_service("records_manager")

        await controller._check_and_trigger_shutdown()

        controller._finalize_kubernetes_raw_artifacts.assert_awaited_once()
        controller.stop.assert_awaited_once()
