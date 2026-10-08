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
from aiperf.common.models import CreditPhaseStats
from aiperf.records.records_manager import RecordsManager


class _Manager:
    """Drives the real watchdog body against real credit-phase snapshots.

    The snapshot is a genuine ``CreditPhaseStats`` pushed through the real
    ``_remember_profiling_credit_stats``. An earlier version of this file
    defined a local fake carrying ``in_flight_requests``/``requests_completed``,
    which let the watchdog ship reading those off ``PhaseRecordsStats`` -- a
    type that has neither. Every tick raised ``AttributeError``, the background
    task swallowed it, and the suite stayed green while the watchdog never ran.
    """

    def __init__(self) -> None:
        self._profiling_started = True
        self._progress_stall_last_total = -1
        self._progress_stall_since = 0.0
        self._latest_profiling_credit_stats: CreditPhaseStats | None = None
        self.warnings: list[str] = []
        self.terminal_failures: list[BaseException] = []

    async def _publish_terminal_failure_result(
        self, phase, cancelled, error, stage=None, reason_prefix=None
    ):
        self.terminal_failures.append(error)

    def warning(self, msg) -> None:
        self.warnings.append(msg() if callable(msg) else msg)


def _observe(mgr: _Manager, completed: int, in_flight: int) -> None:
    """Publish a real profiling snapshot the way the message handlers do."""
    RecordsManager._remember_profiling_credit_stats(
        mgr,
        CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            requests_sent=completed + in_flight,
            requests_completed=completed,
            requests_cancelled=0,
        ),
    )


async def _tick(
    mgr: _Manager, monkeypatch, total: int, now: float, in_flight: int = 1
) -> None:
    _observe(mgr, completed=total, in_flight=in_flight)
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

    await _tick(mgr, monkeypatch, total=198, now=1000.0)
    assert len(mgr.terminal_failures) == 1, (
        "a stall must publish a fatal result; raising is only logged because "
        "@background_task defaults to stop_on_error=False, so the run would "
        "hang on regardless"
    )
    assert "stalled" in str(mgr.terminal_failures[0])


@pytest.mark.asyncio
async def test_stall_is_measured_from_last_progress_not_from_start(
    monkeypatch,
) -> None:
    """Elapsed run time is not the measure; time since the last completion is."""
    mgr = _Manager()
    await _tick(mgr, monkeypatch, total=1, now=0.0)
    await _tick(mgr, monkeypatch, total=2, now=5000.0)  # progress 5000s in
    # Only 100s without progress, even though the run is 5100s old.
    await _tick(mgr, monkeypatch, total=2, now=5100.0)
    assert len(mgr.warnings) == 1
    assert "100s" in mgr.warnings[0]


@pytest.mark.asyncio
async def test_watchdog_disarmed_until_profiling_starts(monkeypatch) -> None:
    """Dataset generation can run arbitrarily long before the first request."""
    mgr = _Manager()
    mgr._profiling_started = False
    await _tick(mgr, monkeypatch, total=0, now=100_000.0)
    assert mgr.warnings == []
    assert mgr.terminal_failures == []


@pytest.mark.asyncio
async def test_zero_timeout_disables_the_watchdog(monkeypatch) -> None:
    monkeypatch.setattr(Environment.RECORD, "PROGRESS_STALL_TIMEOUT", 0.0)
    mgr = _Manager()
    with pytest.raises(asyncio.CancelledError):
        await _tick(mgr, monkeypatch, total=0, now=0.0)


@pytest.mark.asyncio
async def test_no_requests_in_flight_is_not_a_stall(monkeypatch) -> None:
    """Zero records is normal whenever nothing is pending.

    A low request rate, a fixed-schedule replay sitting in a recorded idle gap,
    or a slow dataset build all produce long quiet stretches on a perfectly
    healthy run. Only an outstanding request that never lands leaves the phase
    unable to finish.
    """
    mgr = _Manager()
    for now in (0.0, 10.0, 100_000.0):
        await _tick(mgr, monkeypatch, total=5, now=now, in_flight=0)
    assert mgr.warnings == []
    assert mgr.terminal_failures == []


@pytest.mark.asyncio
async def test_a_quiet_stretch_does_not_accumulate_toward_a_later_stall(
    monkeypatch,
) -> None:
    mgr = _Manager()
    await _tick(mgr, monkeypatch, total=5, now=0.0, in_flight=0)
    await _tick(mgr, monkeypatch, total=5, now=5_000.0, in_flight=0)
    # Work starts; only 100s of genuine stall has elapsed.
    await _tick(mgr, monkeypatch, total=5, now=5_000.0, in_flight=2)
    await _tick(mgr, monkeypatch, total=5, now=5_100.0, in_flight=2)

    assert mgr.terminal_failures == []
    assert len(mgr.warnings) == 1
    assert "100s" in mgr.warnings[0]


def test_the_watchdog_is_actually_scheduled() -> None:
    """The watchdog must stay attached to ``@background_task``.

    Every other test here calls ``_watch_for_progress_stall`` directly, so none
    of them notice if the decorator stops applying to it -- which is exactly
    what happened when a helper was inserted between the decorator and the
    function: the helper became the background task, the watchdog was never
    scheduled, and the suite stayed green.
    """
    hook_type = getattr(
        RecordsManager._watch_for_progress_stall, "__aiperf_hook_type__", None
    )
    assert hook_type == "@background_task", (
        "the progress-stall watchdog is not registered as a background task, "
        "so it will never run"
    )
    assert not hasattr(
        RecordsManager._remember_profiling_credit_stats, "__aiperf_hook_type__"
    ), "the snapshot helper is called from message handlers, not scheduled"
