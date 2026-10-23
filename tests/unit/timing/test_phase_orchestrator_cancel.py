# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancelling or stopping the orchestrator while a phase is running.

``cancel()`` clears ``_active_runners`` while ``_execute_phases`` is still
awaiting the runner, so the loop must tolerate the runner already being gone
and must not start any later phase. The stop path (``_stop_orchestrator``,
the ``@on_stop`` hook) tears the same runners down without going through
``cancel()``, so it must gate the phase loop too — including the window
where stop arrives before the loop has appended its first runner.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.timing.phase_orchestrator import PhaseOrchestrator


class _CancellingRunner:
    """Stands in for PhaseRunner; cancels the orchestrator mid-run when told to."""

    started: list[CreditPhase] = []
    cancel_during: CreditPhase | None = None
    orchestrator: PhaseOrchestrator | None = None

    def __init__(self, *, config: SimpleNamespace, **_: object) -> None:
        self.phase = config.phase
        self.cancelled = False

    def set_phase_complete_callback(self, _cb: object) -> None:
        pass

    def set_phase_error_callback(self, _cb: object) -> None:
        pass

    def cancel(self) -> None:
        self.cancelled = True

    async def run(self, *, is_final_phase: bool, seamless_to_next: bool) -> None:
        _CancellingRunner.started.append(self.phase)
        if self.phase == _CancellingRunner.cancel_during:
            assert _CancellingRunner.orchestrator is not None
            await _CancellingRunner.orchestrator.cancel()


class _StoppingRunner:
    """Stands in for PhaseRunner; runs the orchestrator's on_stop hook mid-run."""

    started: list[CreditPhase] = []
    stop_during: CreditPhase | None = None
    orchestrator: PhaseOrchestrator | None = None

    def __init__(self, *, config: SimpleNamespace, **_: object) -> None:
        self.phase = config.phase
        self.cancelled = False

    def set_phase_complete_callback(self, _cb: object) -> None:
        pass

    def set_phase_error_callback(self, _cb: object) -> None:
        pass

    def cancel(self) -> None:
        self.cancelled = True

    async def run(self, *, is_final_phase: bool, seamless_to_next: bool) -> None:
        _StoppingRunner.started.append(self.phase)
        if self.phase == _StoppingRunner.stop_during:
            assert _StoppingRunner.orchestrator is not None
            await _StoppingRunner.orchestrator._stop_orchestrator()


def _phase_config(phase: CreditPhase) -> SimpleNamespace:
    return SimpleNamespace(phase=phase, seamless=False, request_cancellation=None)


@pytest.fixture
def orchestrator(create_orchestrator_harness) -> PhaseOrchestrator:
    orch = create_orchestrator_harness([("c1", 1)]).orchestrator
    orch._ordered_phase_configs = [
        _phase_config(CreditPhase.WARMUP),
        _phase_config(CreditPhase.PROFILING),
    ]
    _CancellingRunner.started = []
    _CancellingRunner.orchestrator = orch
    _StoppingRunner.started = []
    _StoppingRunner.stop_during = None
    _StoppingRunner.orchestrator = orch
    return orch


@pytest.mark.asyncio
async def test_execute_phases_cancel_during_warmup_stops_cleanly(
    orchestrator: PhaseOrchestrator,
) -> None:
    """Covers both halves: the post-run removal must tolerate the runner
    already being gone, and profiling must never start."""
    _CancellingRunner.cancel_during = CreditPhase.WARMUP
    with (
        patch("aiperf.timing.phase_orchestrator.PhaseRunner", _CancellingRunner),
        patch("aiperf.timing.phase_orchestrator.RequestCancellationSimulator"),
    ):
        await orchestrator._execute_phases()

    assert _CancellingRunner.started == [CreditPhase.WARMUP]
    assert orchestrator._active_runners == []


@pytest.mark.asyncio
async def test_execute_phases_stop_during_warmup_starts_no_later_phase(
    orchestrator: PhaseOrchestrator,
) -> None:
    """The on_stop hook clears ``_active_runners`` without going through
    ``cancel()``; the loop must tolerate the cleared list and must not start
    the next phase once stop-path teardown has begun."""
    _StoppingRunner.stop_during = CreditPhase.WARMUP
    with (
        patch("aiperf.timing.phase_orchestrator.PhaseRunner", _StoppingRunner),
        patch("aiperf.timing.phase_orchestrator.RequestCancellationSimulator"),
    ):
        await orchestrator._execute_phases()

    assert _StoppingRunner.started == [CreditPhase.WARMUP]
    assert orchestrator._active_runners == []


@pytest.mark.asyncio
async def test_execute_phases_after_stop_starts_no_phase(
    orchestrator: PhaseOrchestrator,
) -> None:
    """Stop arriving before the loop appends its first runner (empty active
    list) must still gate the loop: no phase may start after stop began."""
    await orchestrator._stop_orchestrator()
    with (
        patch("aiperf.timing.phase_orchestrator.PhaseRunner", _StoppingRunner),
        patch("aiperf.timing.phase_orchestrator.RequestCancellationSimulator"),
    ):
        await orchestrator._execute_phases()

    assert _StoppingRunner.started == []
    assert orchestrator._active_runners == []
