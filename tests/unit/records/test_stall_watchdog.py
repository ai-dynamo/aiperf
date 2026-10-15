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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase, ProfileCancelReason, WorkerStatus
from aiperf.common.environment import Environment
from aiperf.common.messages import WorkerGroupStatsMessage, WorkerHealthMessage
from aiperf.common.models import CreditPhaseStats, ProcessHealth, WorkerTaskStats
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

    # The watchdog delegates per-phase bookkeeping here; use the real
    # implementation so the fake cannot drift from what ships.
    _longest_stalled_phase = RecordsManager._longest_stalled_phase
    _stream_activity_since_last_check = RecordsManager._stream_activity_since_last_check

    def __init__(self) -> None:
        self._profiling_started = True
        self._worker_stream_chunks: dict[str, int] = {}
        self._progress_stall_last_stream_chunks: dict[str, int] = {}
        self._progress_stall_last_total: dict[int | None, int] = {}
        self._progress_stall_since: dict[int | None, float] = {}
        self._latest_profiling_credit_stats: dict[int | None, CreditPhaseStats] = {}
        self._progress_stall_triggered = False
        self.warnings: list[str] = []
        self.cancels: list[tuple[str, str]] = []

    async def _request_profile_cancel(self, reason, reason_detail) -> bool:
        self.cancels.append((reason, reason_detail))
        return True

    def warning(self, msg) -> None:
        self.warnings.append(msg() if callable(msg) else msg)


def _observe(
    mgr: _Manager, completed: int, in_flight: int, phase_index: int | None = None
) -> None:
    """Publish a real profiling snapshot the way the message handlers do."""
    RecordsManager._remember_profiling_credit_stats(
        mgr,
        CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            phase_index=phase_index,
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
    assert len(mgr.cancels) == 1, (
        "a stall must end the run through the cancel path: publishing a "
        "terminal result fills only the profile domain, leaving the "
        "controller waiting on server_metrics, which cannot arrive until the "
        "profiling phase completes"
    )
    reason, detail = mgr.cancels[0]
    assert reason == ProfileCancelReason.PROGRESS_STALL
    assert "stalled" in detail

    # Latched: a stalled run asks once, not on every subsequent tick.
    await _tick(mgr, monkeypatch, total=198, now=2000.0)
    assert len(mgr.cancels) == 1


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
    assert mgr.cancels == []


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
    assert mgr.cancels == []


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

    assert mgr.cancels == []
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


@pytest.mark.asyncio
async def test_a_stuck_phase_is_caught_while_another_phase_keeps_completing(
    monkeypatch,
) -> None:
    """Overlapping profiling phases must each be judged on their own counters.

    Under ``seamless`` a later profiling phase starts while the previous one
    drains, and both publish every interval. Held in one slot the snapshots
    interleaved, and since each phase carries its own ``requests_completed``
    the count changed on nearly every tick -- which reads as progress. The
    stall clock reset perpetually and a request stuck in either phase was
    never caught, so the run hung with the watchdog armed and silent.
    """
    monkeypatch.setattr(Environment.RECORD, "PROGRESS_STALL_TIMEOUT", 100.0)
    monkeypatch.setattr(Environment.RECORD, "PROGRESS_STALL_CHECK_INTERVAL", 10.0)
    mgr = _Manager()

    # Phase 0 is wedged at 5 completed with one request outstanding, while
    # phase 1 keeps completing work the whole time.
    for i, now in enumerate([0.0, 50.0, 150.0, 250.0]):
        _observe(mgr, completed=5, in_flight=1, phase_index=0)
        _observe(mgr, completed=100 + i, in_flight=4, phase_index=1)
        monkeypatch.setattr(
            "aiperf.records.records_manager.time.monotonic", lambda now=now: now
        )
        await RecordsManager._watch_for_progress_stall(mgr)

    assert mgr.cancels, (
        "a request stuck in phase 0 never triggered the watchdog while phase 1 "
        "kept reporting completions"
    )
    assert "in phase 0" in mgr.cancels[0][1], mgr.cancels[0][1]


@pytest.mark.asyncio
async def test_two_phases_both_progressing_stay_quiet(monkeypatch) -> None:
    """The per-phase clocks must not manufacture a stall out of two healthy phases."""
    monkeypatch.setattr(Environment.RECORD, "PROGRESS_STALL_TIMEOUT", 100.0)
    mgr = _Manager()
    for i, now in enumerate([0.0, 50.0, 150.0, 250.0, 5_000.0]):
        _observe(mgr, completed=i, in_flight=2, phase_index=0)
        _observe(mgr, completed=i * 3, in_flight=2, phase_index=1)
        monkeypatch.setattr(
            "aiperf.records.records_manager.time.monotonic", lambda now=now: now
        )
        await RecordsManager._watch_for_progress_stall(mgr)

    assert mgr.cancels == []
    assert mgr.warnings == []


@pytest.mark.asyncio
async def test_no_snapshot_yet_is_not_a_stall(monkeypatch) -> None:
    """Before the first credit snapshot there is nothing to judge.

    Ticking in that window must not start a clock: a run whose first phase has
    not reported yet would otherwise accumulate toward a stall it was never in.
    """
    monkeypatch.setattr(Environment.RECORD, "PROGRESS_STALL_TIMEOUT", 100.0)
    mgr = _Manager()

    for now in (0.0, 500.0, 50_000.0):
        monkeypatch.setattr(
            "aiperf.records.records_manager.time.monotonic", lambda now=now: now
        )
        await RecordsManager._watch_for_progress_stall(mgr)

    assert mgr.cancels == []
    assert mgr.warnings == []


@pytest.mark.asyncio
async def test_a_cancel_that_cannot_be_sent_reports_failure(monkeypatch) -> None:
    """If the PROFILE_CANCEL command cannot go out, say so and report False.

    The caller uses the return value to decide whether its own abort latch
    stays set, so swallowing the error would leave the run believing a
    cancellation is in flight that never left this process.
    """
    mgr = SimpleNamespace(
        service_id="records-manager-test",
        warning=MagicMock(),
        send_command_to_controller=AsyncMock(side_effect=RuntimeError("bus down")),
    )

    sent = await RecordsManager._request_profile_cancel(
        mgr, ProfileCancelReason.PROGRESS_STALL, "stalled"
    )

    assert sent is False
    assert "Failed to request PROFILE_CANCEL" in mgr.warning.call_args[0][0]


async def _tick_with_stream(
    mgr: _Manager, monkeypatch, *, now: float, chunks: dict[str, int]
) -> None:
    """A tick where nothing completes but workers report streamed chunks."""
    for worker_id, count in chunks.items():
        await RecordsManager._on_worker_health_for_stall(
            mgr,
            WorkerHealthMessage(
                service_id=worker_id,
                health=ProcessHealth(
                    pid=1, create_time=0.0, uptime=0.0, cpu_usage=0.0, memory_usage=0
                ),
                task_stats=WorkerTaskStats(total=1, stream_chunks_received=count),
            ),
        )
    await _tick(mgr, monkeypatch, total=19, now=now)


@pytest.mark.asyncio
async def test_a_request_still_streaming_is_not_a_stall(monkeypatch) -> None:
    """A long generation keeps receiving chunks; it is slow, not stuck.

    This is the docs-e2e vision failure: one request streamed for over 600s
    with no completion, and the watchdog cancelled a healthy run.
    """
    mgr = _Manager()
    for i, now in enumerate([0.0, 300.0, 700.0, 5000.0]):
        await _tick_with_stream(mgr, monkeypatch, now=now, chunks={"w1": 100 * i})
    assert mgr.cancels == []
    assert mgr.warnings == []


@pytest.mark.asyncio
async def test_flat_stream_activity_still_fails(monkeypatch) -> None:
    mgr = _Manager()
    await _tick_with_stream(mgr, monkeypatch, now=0.0, chunks={"w1": 500})
    await _tick_with_stream(mgr, monkeypatch, now=100.0, chunks={"w1": 500})
    await _tick_with_stream(mgr, monkeypatch, now=1000.0, chunks={"w1": 500})
    assert len(mgr.cancels) == 1
    assert mgr.cancels[0][0] == ProfileCancelReason.PROGRESS_STALL


@pytest.mark.asyncio
async def test_offsetting_worker_counts_still_read_as_activity(monkeypatch) -> None:
    """A restarted worker's drop must not cancel out another worker's gain.

    Summing would see 10 both times and let the stall clock run, though w2
    received five new chunks.
    """
    mgr = _Manager()
    await _tick_with_stream(mgr, monkeypatch, now=0.0, chunks={"w1": 10, "w2": 0})
    await _tick_with_stream(mgr, monkeypatch, now=300.0, chunks={"w1": 5, "w2": 5})
    await _tick_with_stream(mgr, monkeypatch, now=700.0, chunks={"w1": 5, "w2": 5})
    assert mgr.cancels == []


@pytest.mark.asyncio
async def test_a_restarted_worker_s_counter_reset_is_activity(monkeypatch) -> None:
    """A lone worker whose count drops has restarted, which is not a freeze.

    This is why the comparison is `!=` and not `>`. With a single worker there
    is no second count to carry the signal, so treating only increases as
    activity would let the clock run through a restart and cancel a run that
    is streaming.
    """
    mgr = _Manager()
    await _tick_with_stream(mgr, monkeypatch, now=0.0, chunks={"w1": 900})
    await _tick_with_stream(mgr, monkeypatch, now=300.0, chunks={"w1": 3})
    await _tick_with_stream(mgr, monkeypatch, now=700.0, chunks={"w1": 3})
    assert mgr.cancels == []


@pytest.mark.asyncio
async def test_a_newly_reporting_idle_worker_is_not_activity(monkeypatch) -> None:
    mgr = _Manager()
    await _tick_with_stream(mgr, monkeypatch, now=0.0, chunks={"w1": 500})
    await _tick_with_stream(mgr, monkeypatch, now=100.0, chunks={"w2": 0})
    await _tick_with_stream(mgr, monkeypatch, now=650.0, chunks={})
    assert len(mgr.cancels) == 1, "the idle worker's first report reset the clock"


@pytest.mark.asyncio
async def test_stream_chunks_are_keyed_by_worker_across_both_sources() -> None:
    mgr = _Manager()
    await RecordsManager._on_worker_health_for_stall(
        mgr,
        WorkerHealthMessage(
            service_id="w1",
            health=ProcessHealth(
                pid=1, create_time=0.0, uptime=0.0, cpu_usage=0.0, memory_usage=0
            ),
            task_stats=WorkerTaskStats(stream_chunks_received=5),
        ),
    )
    await RecordsManager._on_worker_group_stats_for_stall(
        mgr,
        WorkerGroupStatsMessage(
            service_id="wgm-0",
            group_id="wgm-0",
            status=WorkerStatus.HEALTHY,
            task_stats=WorkerTaskStats(stream_chunks_received=10),
            worker_task_stats={
                "w1": WorkerTaskStats(stream_chunks_received=7),
                "w2": WorkerTaskStats(stream_chunks_received=3),
            },
        ),
    )
    assert mgr._worker_stream_chunks == {"w1": 7, "w2": 3}
