# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seamless-mode fatal-error propagation in PhaseOrchestrator.

Seamless non-final phases run their return-wait detached (``PhaseRunner.run()``
returns without awaiting it), and several runners can be active at once. These
tests cover the two ways a fatal request-free control-node failure would
otherwise be lost or mis-attributed:

- ``_record_control_fatal_error`` must record on EVERY active runner, not the
  callback handler's single mutable ``progress`` slot (which may point at the
  wrong phase under concurrent seamless runners).
- ``_execute_phases`` must await outstanding seamless waits and re-raise a
  captured fatal error so the run FAILS instead of reporting success.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiperf.timing.phase_orchestrator import PhaseOrchestrator


def _bare_orchestrator() -> PhaseOrchestrator:
    orch = PhaseOrchestrator.__new__(PhaseOrchestrator)
    orch._active_runners = []
    orch._seamless_phase_error = None
    orch._callback_handler = MagicMock()
    return orch


def _fake_runner() -> MagicMock:
    r = MagicMock()
    r.recorded_errors = []
    r.record_control_fatal_error.side_effect = r.recorded_errors.append
    r.return_wait_task = None
    r.control_fatal_error = None
    return r


def test_record_control_fatal_error_hits_every_active_runner() -> None:
    orch = _bare_orchestrator()
    r1, r2 = _fake_runner(), _fake_runner()
    orch._active_runners = [r1, r2]

    err = RuntimeError("virtual return blew up")
    orch._record_control_fatal_error(err)

    # Recorded on BOTH active phases -- not just the callback handler's slot.
    assert r1.recorded_errors == [err]
    assert r2.recorded_errors == [err]
    orch._callback_handler.progress.record_fatal_error.assert_not_called()


def test_record_control_fatal_error_falls_back_when_no_active_runner() -> None:
    """A failure arriving between phases (no active runner) still lands on the
    callback handler's current progress tracker."""
    orch = _bare_orchestrator()
    err = RuntimeError("between phases")

    orch._record_control_fatal_error(err)

    orch._callback_handler.progress.record_fatal_error.assert_called_once_with(err)


def test_on_seamless_phase_error_keeps_first() -> None:
    orch = _bare_orchestrator()
    first = RuntimeError("first")
    second = RuntimeError("second")

    orch._on_seamless_phase_error(first)
    orch._on_seamless_phase_error(second)

    assert orch._seamless_phase_error is first


@pytest.mark.asyncio
async def test_await_outstanding_seamless_waits_captures_recorded_fatal() -> None:
    """A seamless runner whose detached wait recorded a fatal control-node
    failure surfaces it into ``_seamless_phase_error`` (so ``_execute_phases``
    re-raises), even when the task's done-callback has not fired."""
    orch = _bare_orchestrator()
    runner = _fake_runner()
    err = RuntimeError("late control-node failure")
    runner.control_fatal_error = err
    orch._active_runners = [runner]

    await orch._await_outstanding_seamless_waits()

    assert orch._seamless_phase_error is err


@pytest.mark.asyncio
async def test_await_outstanding_seamless_waits_noop_when_clean() -> None:
    orch = _bare_orchestrator()
    orch._active_runners = [_fake_runner(), _fake_runner()]

    await orch._await_outstanding_seamless_waits()

    assert orch._seamless_phase_error is None
