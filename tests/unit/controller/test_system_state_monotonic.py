# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SystemState advances forward only.

_set_system_state deduped the same state but had no ordering guard.
_cancel_profiling sets STOPPING and then blocks on ProfileCancelCommand; during
that window the cancelled PhaseOrchestrator's `finally:` unconditionally
publishes CreditsCompleteMessage, which stamps PROCESSING. status.subPhase then
went stopping -> processing -> shutdown, breaking every consumer that treats
the sequence as forward-only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pytest import param

from aiperf.common.enums import SystemState
from aiperf.controller.system_controller import SystemController


class TestSystemStateRank:
    def test_lifecycle_is_ordered(self) -> None:
        ranks = [
            SystemState.INITIALIZING.rank,
            SystemState.CONFIGURING.rank,
            SystemState.READY.rank,
            SystemState.PROFILING.rank,
            SystemState.PROCESSING.rank,
            SystemState.STOPPING.rank,
            SystemState.SHUTDOWN.rank,
        ]
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == len(ranks)

    def test_every_member_has_a_rank(self) -> None:
        for state in SystemState:
            assert isinstance(state.rank, int)


class TestSetSystemStateMonotonic:
    def _controller(self) -> SystemController:
        controller = SystemController.__new__(SystemController)
        controller._system_state = SystemState.STOPPING
        controller.service_id = "system_controller"
        controller.publish = AsyncMock()
        controller.info = lambda *a, **k: None
        controller.debug = lambda *a, **k: None
        return controller

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "target,should_publish",
        [
            param(SystemState.PROCESSING, False, id="backwards-ignored"),
            param(SystemState.PROFILING, False, id="far-backwards-ignored"),
            param(SystemState.STOPPING, False, id="same-state-deduped"),
            param(SystemState.SHUTDOWN, True, id="forward-accepted"),
        ],
    )  # fmt: skip
    async def test_transitions(self, target: SystemState, should_publish: bool) -> None:
        controller = self._controller()
        await SystemController._set_system_state(controller, target)
        assert controller.publish.called is should_publish
        expected = target if should_publish else SystemState.STOPPING
        assert controller._system_state is expected
