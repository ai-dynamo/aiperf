# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the AIPerf clock abstraction.

Covers the unit-conversion mixin, the stateless ``WallClock``, and the
``VirtualClock`` discrete-event semantics: monotonic ``advance_to``, the
fast-path for already-crossed deadlines, deterministic wake order by
``(deadline, insertion_id)``, and the ``peek_min_waiter_ns`` /
``set_on_waiter_parked`` driver hooks.
"""

import asyncio

import pytest

from aiperf.common.clock import AIPerfClock, VirtualClock, WallClock


def test_wallclock_satisfies_protocol():
    assert isinstance(WallClock(), AIPerfClock)


def test_virtualclock_satisfies_protocol():
    assert isinstance(VirtualClock(), AIPerfClock)


def test_unit_conversions_share_the_ns_layer():
    clock = VirtualClock()
    clock._now_ns = 2_500_000_000  # 2.5s
    assert clock.now_ns() == 2_500_000_000
    assert clock.now() == pytest.approx(2.5)
    assert clock.now_ms() == pytest.approx(2_500.0)


@pytest.mark.asyncio
async def test_wallclock_nonpositive_sleep_is_immediate():
    clock = WallClock()
    # Must not raise or hang; negative/zero durations return immediately.
    await clock.sleep_ns(0)
    await clock.sleep_ns(-5)
    await clock.sleep(-1.0)


@pytest.mark.asyncio
async def test_virtualclock_advance_wakes_crossed_sleeper():
    clock = VirtualClock()
    woke_at: list[int] = []

    async def sleeper() -> None:
        await clock.sleep_ns(1_000)
        woke_at.append(clock.now_ns())

    task = asyncio.ensure_future(sleeper())
    await asyncio.sleep(0)  # let the sleeper park
    assert clock.has_waiters()
    assert clock.peek_min_waiter_ns() == 1_000

    await clock.advance_to(1_000)
    await task
    assert woke_at == [1_000]
    assert not clock.has_waiters()


@pytest.mark.asyncio
async def test_virtualclock_advance_is_monotonic():
    clock = VirtualClock()
    await clock.advance_to(5_000)
    assert clock.now_ns() == 5_000
    # Backwards / equal advances are silently ignored.
    await clock.advance_to(4_000)
    assert clock.now_ns() == 5_000
    await clock.advance_to(5_000)
    assert clock.now_ns() == 5_000


@pytest.mark.asyncio
async def test_virtualclock_fast_path_for_already_crossed_deadline():
    clock = VirtualClock()
    await clock.advance_to(10_000)
    # Deadline already in the past -> returns immediately, never parks.
    await clock.sleep_until_ns(5_000)
    assert not clock.has_waiters()


@pytest.mark.asyncio
async def test_virtualclock_wakes_in_deadline_then_insertion_order():
    clock = VirtualClock()
    order: list[str] = []

    async def sleeper(name: str, deadline_ns: int) -> None:
        await clock.sleep_until_ns(deadline_ns)
        order.append(name)

    # Park three sleepers: two share a deadline (must wake in registration
    # order), one is later.
    t_a = asyncio.ensure_future(sleeper("a", 1_000))
    await asyncio.sleep(0)
    t_b = asyncio.ensure_future(sleeper("b", 1_000))
    await asyncio.sleep(0)
    t_c = asyncio.ensure_future(sleeper("c", 2_000))
    await asyncio.sleep(0)

    assert clock.peek_min_waiter_ns() == 1_000

    # One advance crossing both 1_000 deadlines wakes a then b (insertion order).
    await clock.advance_to(1_500)
    await asyncio.gather(t_a, t_b)
    assert order == ["a", "b"]

    await clock.advance_to(2_000)
    await t_c
    assert order == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_virtualclock_on_waiter_parked_callback_fires():
    clock = VirtualClock()
    parked: list[int] = []
    clock.set_on_waiter_parked(lambda: parked.append(1))

    async def sleeper() -> None:
        await clock.sleep_ns(500)

    task = asyncio.ensure_future(sleeper())
    await asyncio.sleep(0)
    assert parked == [1]  # callback fired exactly once when the waiter parked

    await clock.advance_to(500)
    await task


@pytest.mark.asyncio
async def test_virtualclock_peek_returns_none_when_idle():
    clock = VirtualClock()
    assert clock.peek_min_waiter_ns() is None
    assert not clock.has_waiters()


@pytest.mark.asyncio
async def test_wallclock_nonpositive_sleep_yields_to_event_loop():
    """C4: a tight loop of sleep_ns(<=0) must not starve a concurrent task.

    The protocol docstring promises asyncio.sleep semantics, which yield to
    the event loop exactly once even for non-positive durations.
    """
    clock = WallClock()
    progress: list[int] = []

    async def side_task() -> None:
        for i in range(5):
            progress.append(i)
            await asyncio.sleep(0)

    task = asyncio.ensure_future(side_task())
    for _ in range(10):
        await clock.sleep_ns(0)
        await clock.sleep_ns(-5)
        await clock.sleep_until_ns(clock.now_ns() - 1_000)
    await task
    assert progress == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_virtualclock_nonpositive_and_crossed_sleeps_yield_to_event_loop():
    """C4: VirtualClock fast paths (zero duration, already-crossed deadline)
    must also yield once instead of returning synchronously."""
    clock = VirtualClock()
    await clock.advance_to(10_000)
    progress: list[int] = []

    async def side_task() -> None:
        for i in range(3):
            progress.append(i)
            await asyncio.sleep(0)

    task = asyncio.ensure_future(side_task())
    for _ in range(5):
        await clock.sleep_ns(0)
        await clock.sleep_until_ns(5_000)  # already crossed
    await task
    assert progress == [0, 1, 2]
    assert not clock.has_waiters()


@pytest.mark.asyncio
async def test_virtualclock_cancelled_sleeper_is_reaped():
    """C5: cancelling a parked sleeper must remove its phantom deadline so a
    driver pump cannot fast-forward sim time to a deadline nobody waits on."""
    clock = VirtualClock()
    task = asyncio.ensure_future(clock.sleep_until_ns(1_000))
    await asyncio.sleep(0)  # let the sleeper park
    assert clock.has_waiters()
    assert clock.peek_min_waiter_ns() == 1_000

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not clock.has_waiters()
    assert clock.peek_min_waiter_ns() is None


@pytest.mark.asyncio
async def test_virtualclock_cancelled_sleeper_does_not_mask_live_waiter():
    """C5: a cancelled earlier deadline must neither be reported by peek nor
    disturb the wake of a live later waiter."""
    clock = VirtualClock()
    dead = asyncio.ensure_future(clock.sleep_until_ns(1_000))
    await asyncio.sleep(0)
    woke_at: list[int] = []

    async def live_sleeper() -> None:
        await clock.sleep_until_ns(2_000)
        woke_at.append(clock.now_ns())

    live = asyncio.ensure_future(live_sleeper())
    await asyncio.sleep(0)

    dead.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dead

    assert clock.peek_min_waiter_ns() == 2_000
    # Advancing across the dead entry's deadline must not error or wake it.
    await clock.advance_to(2_000)
    await live
    assert woke_at == [2_000]
    assert not clock.has_waiters()
