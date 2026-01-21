# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.timing.concurrency import (
    ConcurrencyManager,
    ConcurrencyStats,
    DynamicConcurrencyLimit,
    GlobalPhaseConcurrencyLimiter,
)

P, W = CreditPhase.PROFILING, CreditPhase.WARMUP


async def _cancel(t: asyncio.Task) -> None:
    t.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await t


class TestDynamicConcurrencyLimit:
    def test_initial_state(self) -> None:
        lim = DynamicConcurrencyLimit()
        assert lim.current_limit == 0 and lim.debt == 0 and lim.effective_slots == 0

    def test_set_limit_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            DynamicConcurrencyLimit().set_limit(-1)

    @pytest.mark.parametrize("init,final,exp_slots", [(0, 10, 10), (10, 25, 25), (50, 25, 25), (10, 0, 0), (100, 75, 75)])  # fmt: skip
    def test_set_limit_no_inflight(self, init: int, final: int, exp_slots: int) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(init)
        lim.set_limit(final)
        assert (
            lim.current_limit == final
            and lim.effective_slots == exp_slots
            and lim.debt == 0
        )

    def test_set_same_limit_noop(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(10)
        lim.set_limit(10)
        assert lim.current_limit == 10 and lim.debt == 0 and lim.effective_slots == 10

    @pytest.mark.asyncio
    async def test_acquire_succeeds_with_permits(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(1)
        await asyncio.wait_for(lim.acquire(), timeout=0.1)
        assert lim.effective_slots == 0

    @pytest.mark.asyncio
    async def test_acquire_blocks_without_permits(self) -> None:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(DynamicConcurrencyLimit(0).acquire(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_release_frees_permit(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(1)
        await lim.acquire()
        assert lim.effective_slots == 0
        lim.release()
        assert lim.effective_slots == 1

    def test_release_without_acquire(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(10)
        lim.release()
        assert lim.effective_slots == 11

    @pytest.mark.asyncio
    @pytest.mark.parametrize("acq,dec,inc,exp_debt,exp_slots", [(50, 25, 60, 0, 10), (50, 25, 35, 15, 0), (50, 25, 50, 0, 0)])  # fmt: skip
    async def test_debt_cancellation(
        self, acq: int, dec: int, inc: int, exp_debt: int, exp_slots: int
    ) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(acq)
        for _ in range(acq):
            await lim.acquire()
        lim.set_limit(dec)
        assert lim.debt == acq - dec
        lim.set_limit(inc)
        assert (
            lim.current_limit == inc
            and lim.debt == exp_debt
            and lim.effective_slots == exp_slots
        )

    @pytest.mark.asyncio
    async def test_increase_wakes_waiters(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(0)
        acquired: list[int] = []

        async def waiter(i: int) -> None:
            await lim.acquire()
            acquired.append(i)

        tasks = [asyncio.create_task(waiter(i)) for i in range(3)]
        await asyncio.sleep(0.05)
        assert len(acquired) == 0
        lim.set_limit(3)
        await asyncio.sleep(0.05)
        assert len(acquired) == 3
        for t in tasks:
            await _cancel(t)

    @pytest.mark.asyncio
    async def test_decrease_with_inflight_creates_debt(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(50)
        for _ in range(50):
            await lim.acquire()
        lim.set_limit(25)
        assert lim.debt == 25 and lim.effective_slots == 0

    @pytest.mark.asyncio
    async def test_decrease_partial_drain_partial_debt(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(50)
        for _ in range(40):
            await lim.acquire()
        lim.set_limit(25)
        assert lim.debt == 15 and lim.effective_slots == 0

    @pytest.mark.asyncio
    async def test_release_absorbs_debt(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(50)
        for _ in range(50):
            await lim.acquire()
        lim.set_limit(25)
        lim.release()
        assert lim.debt == 24

    @pytest.mark.asyncio
    async def test_releases_drain_debt_then_free(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(5)
        for _ in range(5):
            await lim.acquire()
        lim.set_limit(3)
        assert lim.debt == 2 and lim.effective_slots == 0
        lim.release()
        lim.release()
        assert lim.debt == 0 and lim.effective_slots == 0
        lim.release()
        assert lim.effective_slots == 1

    @pytest.mark.asyncio
    async def test_set_same_limit_with_debt(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(50)
        for _ in range(50):
            await lim.acquire()
        lim.set_limit(25)
        lim.set_limit(25)
        assert lim.debt == 25

    @pytest.mark.asyncio
    async def test_large_debt_small_increase(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(1000)
        for _ in range(1000):
            await lim.acquire()
        lim.set_limit(0)
        lim.set_limit(1)
        assert lim.debt == 999 and lim.effective_slots == 0

    @pytest.mark.asyncio
    async def test_debt_exactly_equals_releases(self) -> None:
        lim = DynamicConcurrencyLimit(10)
        for _ in range(10):
            await lim.acquire()
        lim.set_limit(5)
        for _ in range(5):
            lim.release()
        assert lim.debt == 0 and lim.effective_slots == 0

    @pytest.mark.asyncio
    async def test_concurrency_ramp_up(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(10)
        for _ in range(10):
            await lim.acquire()
        lim.set_limit(25)
        assert lim.effective_slots == 15
        lim.set_limit(50)
        assert lim.effective_slots == 40
        lim.set_limit(100)
        assert lim.effective_slots == 90

    @pytest.mark.asyncio
    async def test_seamless_transition_with_drain(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(10)
        for _ in range(10):
            await lim.acquire()
        lim.set_limit(25)
        for _ in range(15):
            await lim.acquire()
        for _ in range(10):
            lim.release()
        assert lim.effective_slots == 10

    def test_oscillating_limits_immediate_drain(self) -> None:
        lim = DynamicConcurrencyLimit()
        for val, exp in [(100, 100), (50, 50), (75, 75), (25, 25), (100, 100)]:
            lim.set_limit(val)
            assert lim.debt == 0 and lim.effective_slots == exp

    @pytest.mark.asyncio
    async def test_oscillating_limits_with_inflight(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(100)
        for _ in range(100):
            await lim.acquire()
        for val, exp_debt in [(50, 50), (75, 25), (25, 75), (100, 0)]:
            lim.set_limit(val)
            assert lim.debt == exp_debt

    @pytest.mark.asyncio
    async def test_multiple_waiters_single_release(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(0)
        cnt = 0

        async def waiter() -> None:
            nonlocal cnt
            await lim.acquire()
            cnt += 1

        tasks = [asyncio.create_task(waiter()) for _ in range(5)]
        await asyncio.sleep(0.05)
        assert cnt == 0
        lim.set_limit(1)
        await asyncio.sleep(0.05)
        assert cnt == 1
        lim.set_limit(3)
        await asyncio.sleep(0.05)
        assert cnt == 3
        for t in tasks:
            await _cancel(t)

    @pytest.mark.asyncio
    async def test_rapid_acquire_release(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(10)

        async def worker(n: int) -> int:
            c = 0
            for _ in range(n):
                await lim.acquire()
                c += 1
                await asyncio.sleep(0)
                lim.release()
            return c

        results = await asyncio.gather(*[worker(100) for _ in range(5)])
        assert sum(results) == 500 and lim.effective_slots == 10

    @pytest.mark.asyncio
    async def test_decrease_immediately_enforces_limit(self) -> None:
        lim = DynamicConcurrencyLimit()
        lim.set_limit(50)
        lim.set_limit(25)
        assert lim.effective_slots == 25 and lim.debt == 0
        for _ in range(25):
            await lim.acquire()
        task = asyncio.create_task(lim.acquire())
        await asyncio.sleep(0.05)
        assert not task.done()
        await _cancel(task)


class TestGlobalPhaseConcurrencyLimiter:
    def test_initial_state_disabled(self) -> None:
        assert not GlobalPhaseConcurrencyLimiter().enabled

    @pytest.mark.parametrize("limit,expected", [(10, True), (None, False)])  # fmt: skip
    def test_configure_enables_limiter(self, limit: int | None, expected: bool) -> None:
        lim = GlobalPhaseConcurrencyLimiter()
        lim.configure_for_phase(P, 10)
        lim.configure_for_phase(W, limit)
        assert lim.enabled == expected

    @pytest.mark.asyncio
    async def test_acquire_requires_configured_phase(self) -> None:
        lim = GlobalPhaseConcurrencyLimiter()
        lim.configure_for_phase(P, 10)
        with pytest.raises(ValueError, match="not configured"):
            await lim.acquire(W, lambda: True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("can_proceed,expected", [(True, True), (False, False)])  # fmt: skip
    async def test_acquire_result(self, can_proceed: bool, expected: bool) -> None:
        lim = GlobalPhaseConcurrencyLimiter()
        lim.configure_for_phase(P, 5)
        assert await lim.acquire(P, lambda: can_proceed) == expected

    def test_release_requires_configured_phase(self) -> None:
        lim = GlobalPhaseConcurrencyLimiter()
        lim.configure_for_phase(P, 10)
        with pytest.raises(ValueError, match="not configured"):
            lim.release(W)

    @pytest.mark.asyncio
    async def test_multiple_phases_independent(self) -> None:
        lim = GlobalPhaseConcurrencyLimiter()
        lim.configure_for_phase(W, 5)
        lim.configure_for_phase(P, 10)
        for _ in range(5):
            await lim.acquire(W, lambda: True)
        assert await lim.acquire(P, lambda: True) is True

    @pytest.mark.asyncio
    async def test_held_slots_tracking(self) -> None:
        lim = GlobalPhaseConcurrencyLimiter()
        lim.configure_for_phase(P, 10)
        assert lim.get_held_slots(P) == 0
        await lim.acquire(P, lambda: True)
        await lim.acquire(P, lambda: True)
        assert lim.get_held_slots(P) == 2
        lim.release(P)
        assert lim.get_held_slots(P) == 1

    def test_unconfigured_phase_held_slots_zero(self) -> None:
        assert GlobalPhaseConcurrencyLimiter().get_held_slots(P) == 0

    @pytest.mark.asyncio
    async def test_stats_tracking(self) -> None:
        lim = GlobalPhaseConcurrencyLimiter()
        lim.configure_for_phase(P, 10)
        await lim.acquire(P, lambda: True)
        await lim.acquire(P, lambda: True)
        lim.release(P)
        assert (
            lim.global_stats.acquire_count == 2 and lim.global_stats.release_count == 1
        )
        ps = lim.get_phase_stats(P)
        assert ps is not None and ps.acquire_count == 2 and ps.release_count == 1

    def test_unconfigured_phase_stats_none(self) -> None:
        assert GlobalPhaseConcurrencyLimiter().get_phase_stats(P) is None


class TestConcurrencyManager:
    def test_initial_state_disabled(self) -> None:
        m = ConcurrencyManager()
        assert not m._session_limiter.enabled and not m._prefill_limiter.enabled

    @pytest.mark.parametrize("conc,prefill,sess_en,pre_en", [(10, None, True, False), (None, 5, False, True), (10, 5, True, True)])  # fmt: skip
    def test_configure_enables_limiters(
        self, conc: int | None, prefill: int | None, sess_en: bool, pre_en: bool
    ) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, conc, prefill)
        assert (
            m._session_limiter.enabled == sess_en
            and m._prefill_limiter.enabled == pre_en
        )

    @pytest.mark.asyncio
    async def test_acquire_session_slot_disabled_calls_check(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None)
        called = False

        def chk() -> bool:
            nonlocal called
            called = True
            return True

        assert await m.acquire_session_slot(P, chk) is True and called

    @pytest.mark.asyncio
    @pytest.mark.parametrize("can_proceed,expected", [(True, True), (False, False)])  # fmt: skip
    async def test_acquire_session_slot_enabled(
        self, can_proceed: bool, expected: bool
    ) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, 5, None)
        assert await m.acquire_session_slot(P, lambda: can_proceed) == expected

    def test_release_session_slot_disabled_noop(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None)
        m.release_session_slot(P)

    @pytest.mark.asyncio
    async def test_release_session_slot_enabled(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, 1, None)
        await m.acquire_session_slot(P, lambda: True)
        task = asyncio.create_task(m.acquire_session_slot(P, lambda: True))
        await asyncio.sleep(0.05)
        assert not task.done()
        m.release_session_slot(P)
        await asyncio.sleep(0.05)
        assert task.done()
        await _cancel(task)

    @pytest.mark.asyncio
    async def test_acquire_prefill_slot_disabled_calls_check(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None)
        called = False

        def chk() -> bool:
            nonlocal called
            called = True
            return True

        assert await m.acquire_prefill_slot(P, chk) is True and called

    @pytest.mark.asyncio
    async def test_acquire_prefill_slot_enabled(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, 5)
        assert await m.acquire_prefill_slot(P, lambda: True) is True

    def test_release_prefill_slot_disabled_noop(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None)
        m.release_prefill_slot(P)

    @pytest.mark.asyncio
    async def test_release_stuck_slots_returns_counts(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, 10, 5)
        for _ in range(3):
            await m.acquire_session_slot(P, lambda: True)
        for _ in range(2):
            await m.acquire_prefill_slot(P, lambda: True)
        assert m.release_stuck_slots(P) == (3, 2)

    def test_release_stuck_slots_disabled_returns_zero(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None)
        assert m.release_stuck_slots(P) == (0, 0)

    def test_get_session_stats_disabled_returns_none(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None)
        assert m.get_session_stats() is None and m.get_session_stats(P) is None

    @pytest.mark.asyncio
    async def test_get_session_stats_enabled(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, 10, None)
        await m.acquire_session_slot(P, lambda: True)
        gs, ps = m.get_session_stats(), m.get_session_stats(P)
        assert gs is not None and gs.acquire_count == 1
        assert ps is not None and ps.acquire_count == 1

    @pytest.mark.asyncio
    async def test_set_session_limit_updates_limits(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, 5, None)
        for _ in range(5):
            await m.acquire_session_slot(P, lambda: True)
        m.set_session_limit(P, 10)
        assert await m.acquire_session_slot(P, lambda: True) is True

    def test_set_session_limit_disabled_noop(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None)
        m.set_session_limit(P, 10)

    @pytest.mark.asyncio
    async def test_set_prefill_limit_updates_limits(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, 3)
        for _ in range(3):
            await m.acquire_prefill_slot(P, lambda: True)
        m.set_prefill_limit(P, 5)
        assert await m.acquire_prefill_slot(P, lambda: True) is True

    def test_set_prefill_limit_disabled_noop(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None)
        m.set_prefill_limit(P, 10)


class TestConcurrencyStats:
    def test_default_values(self) -> None:
        s = ConcurrencyStats()
        assert s.acquire_count == 0 and s.release_count == 0 and s.wait_count == 0

    def test_custom_values(self) -> None:
        s = ConcurrencyStats(acquire_count=10, release_count=5, wait_count=2)
        assert s.acquire_count == 10 and s.release_count == 5 and s.wait_count == 2
