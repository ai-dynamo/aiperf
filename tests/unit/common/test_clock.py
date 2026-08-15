# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The AIPerf clock abstraction: protocol conformance, unit conversions, and SimClock deadline/cancellation bookkeeping."""

import asyncio
import time

import pytest
from pytest import param

from aiperf.common.clock import Clock, RealClock, SimClock


@pytest.mark.parametrize(
    "clock_cls",
    [
        param(RealClock, id="real"),
        param(SimClock, id="sim"),
    ],
)  # fmt: skip
def test_clock_satisfies_protocol(clock_cls: type) -> None:
    """Both clock implementations are runtime-checkable Clock instances."""
    assert isinstance(clock_cls(), Clock)


def test_clock_surface_is_nanoseconds_only() -> None:
    """The seam is integer ns; no seconds/millisecond layer exists to drift.

    Pins the deliberate minimalism: an earlier revision carried
    now/sleep/sleep_until in three unit flavours, six of whose members had no
    callers anywhere.
    """
    clock = SimClock()
    clock.advance_to(2_500_000_000)
    assert clock.perf_ns() == 2_500_000_000
    for absent in (
        "now",
        "now_ms",
        "sleep",
        "sleep_ms",
        "sleep_until",
        "sleep_until_ms",
    ):
        assert not hasattr(clock, absent), f"{absent} should not exist on the seam"


@pytest.mark.asyncio
async def test_real_clock_nonpositive_sleep_is_immediate() -> None:
    """Zero and negative real-clock sleeps return immediately instead of raising or hanging."""
    clock = RealClock()
    await clock.sleep_ns(0)
    await clock.sleep_ns(-5)


@pytest.mark.asyncio
async def test_sim_clock_advance_wakes_crossed_sleeper() -> None:
    """Advancing sim time across a parked sleeper's deadline wakes it at exactly that deadline."""
    clock = SimClock()
    woke_at: list[int] = []

    async def sleeper() -> None:
        await clock.sleep_ns(1_000)
        woke_at.append(clock.perf_ns())

    task = asyncio.ensure_future(sleeper())
    await asyncio.sleep(0)  # let the sleeper park
    assert clock.has_sleepers()
    assert clock.next_event_time() == 1_000

    clock.advance_to(1_000)
    await task
    assert woke_at == [1_000]
    assert not clock.has_sleepers()


@pytest.mark.asyncio
async def test_sim_clock_advance_is_monotonic() -> None:
    """Backwards and equal advances are silently ignored, so sim time never regresses."""
    clock = SimClock()
    clock.advance_to(5_000)
    assert clock.perf_ns() == 5_000
    clock.advance_to(4_000)
    assert clock.perf_ns() == 5_000
    clock.advance_to(5_000)
    assert clock.perf_ns() == 5_000


@pytest.mark.asyncio
async def test_sim_clock_next_event_time_is_none_when_idle() -> None:
    """An idle clock reports no waiters and no minimum deadline."""
    clock = SimClock()
    assert clock.next_event_time() is None
    assert not clock.has_sleepers()


@pytest.mark.asyncio
async def test_real_clock_nonpositive_sleep_yields_to_event_loop() -> None:
    """C4: a tight loop of non-positive real-clock sleeps must not starve a concurrent task."""
    # The protocol docstring promises asyncio.sleep semantics, which yield to
    # the event loop exactly once even for non-positive durations.
    clock = RealClock()
    progress: list[int] = []

    async def side_task() -> None:
        for i in range(5):
            progress.append(i)
            await asyncio.sleep(0)

    task = asyncio.ensure_future(side_task())
    for _ in range(10):
        await clock.sleep_ns(0)
        await clock.sleep_ns(-5)
    await task
    assert progress == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_sim_clock_nonpositive_sleep_yields_to_event_loop() -> None:
    """C4: the SimClock zero-duration and already-crossed fast paths must yield once rather than return synchronously."""
    clock = SimClock()
    clock.advance_to(10_000)
    progress: list[int] = []

    async def side_task() -> None:
        for i in range(3):
            progress.append(i)
            await asyncio.sleep(0)

    task = asyncio.ensure_future(side_task())
    for _ in range(5):
        await clock.sleep_ns(0)
    await task
    assert progress == [0, 1, 2]
    assert not clock.has_sleepers()


@pytest.mark.asyncio
async def test_real_clock_sleep_ns_waits_at_least_requested(time_traveler) -> None:
    """A non-trivial sleep waits the requested duration.

    Takes the ``time_traveler`` fixture so the unit suite's instant-sleep
    patch stands down and ``asyncio.sleep`` advances looptime's virtual clock
    -- otherwise every sleep returns in microseconds and the assertion is
    vacuous. Virtual time has no scheduling slop, so this pins the EXACT
    duration rather than a lower bound.
    """
    want_ns = 5_000_000  # 5 ms
    with time_traveler.sleeps_for(want_ns / 1e9):
        await RealClock().sleep_ns(want_ns)


@pytest.mark.asyncio
async def test_real_clock_sleep_ns_zero_returns_fast() -> None:
    """Zero / negative durations return promptly without arming a timer."""
    started = time.perf_counter_ns()
    clock = RealClock()
    await clock.sleep_ns(0)
    await clock.sleep_ns(-1)
    assert time.perf_counter_ns() - started < 50_000_000


@pytest.mark.asyncio
async def test_sim_clock_concurrent_sleeps_overlap() -> None:
    """Concurrent sleeps OVERLAP; they do not sum.

    The defining property of a deadline-based sim clock, and the one a naive
    ``now += delay`` model gets wrong: three sleepers of 10/20/30ns starting
    together must finish at 30ns (the longest), not 60ns (their sum), each
    waking at its own deadline.

    Pinned here because a bespoke clock in the graph tests once got this wrong
    and reported a 40s replay as 100s, which read as a pacing regression in
    production code that had none.
    """
    clock = SimClock()
    woke_at: dict[int, int] = {}

    async def sleeper(duration_ns: int) -> None:
        await clock.sleep_ns(duration_ns)
        woke_at[duration_ns] = clock.perf_ns()

    tasks = [asyncio.ensure_future(sleeper(d)) for d in (10, 20, 30)]
    await asyncio.sleep(0)

    while clock.has_sleepers():
        next_ns = clock.next_event_time()
        assert next_ns is not None
        clock.advance_to(next_ns)
        await asyncio.sleep(0)

    await asyncio.gather(*tasks)
    assert clock.perf_ns() == 30, "concurrent sleeps summed instead of overlapping"
    assert woke_at == {10: 10, 20: 20, 30: 30}
