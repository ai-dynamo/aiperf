# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run that stops making progress must fail, not wait forever.

A phase is only complete once the credit phase reports its final count, so a
credit that is never returned leaves that condition unevaluable: the run waits
with no output until something external kills it. Per-request timeouts do not
cover it -- they default to six hours, and never apply to a request that was
never dispatched.
"""

from __future__ import annotations

import asyncio

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.records.records_manager import RecordsManager


class _Tracker:
    def __init__(self) -> None:
        self.total = 0

    def total_records_for_phase(self, phase: CreditPhase) -> int:
        assert phase == CreditPhase.PROFILING
        return self.total


class _Manager:
    """Drives the real watchdog body against controllable record counts."""

    def __init__(self) -> None:
        self._records_tracker = _Tracker()
        self._complete_credit_phases: set[CreditPhase] = set()
        self._profiling_started = True
        self._progress_stall_last_total = -1
        self._progress_stall_since = 0.0
        self.warnings: list[str] = []

    def warning(self, msg) -> None:
        self.warnings.append(msg() if callable(msg) else msg)


async def _tick(mgr: _Manager, monkeypatch, total: int, now: float) -> None:
    mgr._records_tracker.total = total
    monkeypatch.setattr("aiperf.records.records_manager.time.monotonic", lambda: now)
    await RecordsManager._watch_for_progress_stall(mgr)


@pytest.mark.asyncio
async def test_progress_keeps_the_watchdog_quiet(monkeypatch) -> None:
    mgr = _Manager()
    for i, now in enumerate([0.0, 100.0, 200.0, 5000.0]):
        await _tick(mgr, monkeypatch, total=i + 1, now=now)
    assert mgr.warnings == []


@pytest.mark.asyncio
async def test_stall_warns_then_fails(monkeypatch) -> None:
    mgr = _Manager()
    await _tick(mgr, monkeypatch, total=198, now=0.0)  # stall clock starts here
    await _tick(mgr, monkeypatch, total=198, now=100.0)  # warn, under the limit
    assert len(mgr.warnings) == 1
    assert "198" in mgr.warnings[0]

    with pytest.raises(RuntimeError, match="stalled"):
        await _tick(mgr, monkeypatch, total=198, now=1000.0)


@pytest.mark.asyncio
async def test_stall_is_measured_from_last_progress_not_from_start(
    monkeypatch,
) -> None:
    """A long run that progressed recently must not fail on total elapsed time."""
    mgr = _Manager()
    await _tick(mgr, monkeypatch, total=1, now=0.0)
    await _tick(mgr, monkeypatch, total=2, now=5000.0)  # progress 5000s in
    # Only 100s without progress, even though the run is 5100s old.
    await _tick(mgr, monkeypatch, total=2, now=5100.0)
    assert len(mgr.warnings) == 1
    assert "100s" in mgr.warnings[0]


@pytest.mark.asyncio
async def test_watchdog_disarmed_until_profiling_starts(monkeypatch) -> None:
    """Dataset generation produces no records for as long as it takes."""
    mgr = _Manager()
    mgr._profiling_started = False
    for now in (0.0, 10.0, 100_000.0):
        await _tick(mgr, monkeypatch, total=0, now=now)
    assert mgr.warnings == []


@pytest.mark.asyncio
async def test_completed_phase_is_not_treated_as_a_stall(monkeypatch) -> None:
    """After the profiling phase completes, no further progress is expected."""
    mgr = _Manager()
    mgr._complete_credit_phases.add(CreditPhase.PROFILING)
    for now in (0.0, 10.0, 100_000.0):
        await _tick(mgr, monkeypatch, total=200, now=now)
    assert mgr.warnings == []


@pytest.mark.asyncio
async def test_zero_timeout_disables_the_watchdog(monkeypatch) -> None:
    monkeypatch.setattr(Environment.RECORD, "PROGRESS_STALL_TIMEOUT", 0.0)
    mgr = _Manager()
    with pytest.raises(asyncio.CancelledError):
        await _tick(mgr, monkeypatch, total=0, now=0.0)
