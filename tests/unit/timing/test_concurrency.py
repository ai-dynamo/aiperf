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

    def test_release_without_acquire_is_noop(self) -> None:
        """Spurious release on a full pool is refused, not absorbed.

        Previously this inflated the pool above its configured limit because
        asyncio.Semaphore has no upper bound; release() now guards against
        callers that double-release or release without a matching acquire.
        """
        lim = DynamicConcurrencyLimit()
        lim.set_limit(10)
        lim.release()
        assert lim.effective_slots == 10

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
    async def test_acquire_request_slot_disabled_calls_check(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None)
        called = False

        def chk() -> bool:
            nonlocal called
            called = True
            return True

        assert await m.acquire_request_slot(P, chk) is True and called

    @pytest.mark.asyncio
    async def test_acquire_request_slot_enabled(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None, request_concurrency=5)
        assert await m.acquire_request_slot(P, lambda: True) is True

    def test_release_request_slot_disabled_noop(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None)
        m.release_request_slot(P)

    @pytest.mark.asyncio
    async def test_request_limiter_blocks_at_limit(self) -> None:
        """Acquiring up to the limit succeeds non-blocking; further try_acquire fails until release."""
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None, request_concurrency=2)
        assert m.try_acquire_request_slot(P, lambda: True) is True
        assert m.try_acquire_request_slot(P, lambda: True) is True
        assert m.try_acquire_request_slot(P, lambda: True) is False
        m.release_request_slot(P)
        assert m.try_acquire_request_slot(P, lambda: True) is True

    @pytest.mark.asyncio
    async def test_release_stuck_slots_returns_counts(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, 10, 5, request_concurrency=4)
        for _ in range(3):
            await m.acquire_session_slot(P, lambda: True)
        for _ in range(2):
            await m.acquire_request_slot(P, lambda: True)
        for _ in range(2):
            await m.acquire_prefill_slot(P, lambda: True)
        assert m.release_stuck_slots(P) == (3, 2, 2)

    def test_release_stuck_slots_disabled_returns_zero(self) -> None:
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None)
        assert m.release_stuck_slots(P) == (0, 0, 0)

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


class TestRequestConcurrencyAdversarial:
    """Adversarial tests for the request concurrency dimension.

    Cover concurrent races, slot leaks under stop-condition rejection,
    interaction with the other two dimensions (most-restrictive wins),
    phase isolation in stuck-slot release, and capacity recovery.
    """

    @pytest.mark.asyncio
    async def test_blocking_acquires_never_exceed_limit(self) -> None:
        """N coroutines racing for K<N slots: at any moment, at most K hold a slot."""
        LIMIT, RACERS = 3, 10
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None, request_concurrency=LIMIT)

        in_flight = 0
        max_in_flight = 0
        gate = asyncio.Event()

        async def racer() -> None:
            nonlocal in_flight, max_in_flight
            await m.acquire_request_slot(P, lambda: True)
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await gate.wait()
            in_flight -= 1
            m.release_request_slot(P)

        tasks = [asyncio.create_task(racer()) for _ in range(RACERS)]
        # Give the first wave time to acquire and block on the gate
        for _ in range(20):
            await asyncio.sleep(0)
        assert in_flight == LIMIT
        assert max_in_flight == LIMIT
        # Releasing the gate lets each finisher hand off to a waiter
        gate.set()
        await asyncio.gather(*tasks)
        assert in_flight == 0
        assert max_in_flight == LIMIT  # never exceeded across the whole run

    @pytest.mark.asyncio
    async def test_stop_condition_after_acquire_releases_slot(self) -> None:
        """can_proceed_fn returning False post-acquire must NOT leak a slot."""
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None, request_concurrency=1)

        # First call returns True (let the acquire complete the global step),
        # second call returns False (rejecting after the phase step) — the
        # limiter must release everything it took.
        toggle = iter([True, False])
        assert await m.acquire_request_slot(P, lambda: next(toggle)) is False

        # If the slot leaked, this acquire would block forever; it must succeed.
        assert (
            await asyncio.wait_for(m.acquire_request_slot(P, lambda: True), timeout=0.5)
            is True
        )
        m.release_request_slot(P)

    @pytest.mark.asyncio
    async def test_most_restrictive_dimension_wins(self) -> None:
        """With session=10, request=2, prefill=10 — only 2 turns can be in flight."""
        m = ConcurrencyManager()
        m.configure_for_phase(
            P, concurrency=10, prefill_concurrency=10, request_concurrency=2
        )

        async def issue_turn() -> bool:
            if not await m.acquire_session_slot(P, lambda: True):
                return False
            if not await m.acquire_request_slot(P, lambda: True):
                m.release_session_slot(P)
                return False
            if not await m.acquire_prefill_slot(P, lambda: True):
                m.release_request_slot(P)
                m.release_session_slot(P)
                return False
            return True

        # Two turns can land immediately; the third must block on request limit.
        assert await issue_turn()
        assert await issue_turn()
        third = asyncio.create_task(issue_turn())
        await asyncio.sleep(0.05)
        assert not third.done(), "third turn should be blocked on request limit"

        # Releasing one request slot unblocks exactly one waiter.
        m.release_request_slot(P)
        assert await asyncio.wait_for(third, timeout=0.5)

    @pytest.mark.asyncio
    async def test_release_stuck_slots_restores_capacity(self) -> None:
        """After stuck-slot release, the limiter accepts the full original limit again."""
        LIMIT = 4
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None, request_concurrency=LIMIT)

        # Fill to the limit, then "lose" the slots (simulating credits that never return).
        for _ in range(LIMIT):
            await m.acquire_request_slot(P, lambda: True)
        assert m.try_acquire_request_slot(P, lambda: True) is False

        released = m.release_stuck_slots(P)
        assert released == (0, LIMIT, 0)

        # Capacity restored — should accept LIMIT new acquires immediately.
        for i in range(LIMIT):
            assert m.try_acquire_request_slot(P, lambda: True) is True, f"slot {i}"
        assert m.try_acquire_request_slot(P, lambda: True) is False

    @pytest.mark.asyncio
    async def test_release_stuck_slots_phase_scoped(self) -> None:
        """release_stuck_slots(WARMUP) frees only WARMUP-held phase slots.

        Note: the global limit is intentionally shared across phases (the last
        configure_for_phase resets it), so this test only stays within the
        shared budget. The semantic we verify is: phase-scoped slot accounting
        — release_stuck_slots(W) does not pop a slot from the PROFILING phase
        semaphore, only the WARMUP one.
        """
        m = ConcurrencyManager()
        m.configure_for_phase(W, None, None, request_concurrency=2)
        m.configure_for_phase(P, None, None, request_concurrency=2)

        # 1 in W, 1 in P — total 2, fits the (now P-set) global cap of 2.
        await m.acquire_request_slot(W, lambda: True)
        await m.acquire_request_slot(P, lambda: True)

        # Release WARMUP only — frees its phase slot AND its global slot.
        assert m.release_stuck_slots(W) == (0, 1, 0)

        # PROFILING's held slot is untouched: it still holds 1.
        assert m.try_acquire_request_slot(P, lambda: True) is True  # 2nd P slot
        # And now the (shared) global is full again.
        assert m.try_acquire_request_slot(P, lambda: True) is False

    @pytest.mark.asyncio
    async def test_extra_release_is_refused_not_inflated(self) -> None:
        """A double-release must NOT inflate the pool above its configured limit.

        asyncio.Semaphore has no upper bound, so DynamicConcurrencyLimit.release()
        explicitly guards against spurious releases; the third release here is
        a no-op (and emits a warning) rather than freeing a phantom slot.
        """
        m = ConcurrencyManager()
        m.configure_for_phase(P, None, None, request_concurrency=2)

        await m.acquire_request_slot(P, lambda: True)
        await m.acquire_request_slot(P, lambda: True)
        m.release_request_slot(P)
        m.release_request_slot(P)
        m.release_request_slot(P)  # spurious — pool already at full capacity

        # Configured limit is 2 → exactly two slots should be acquirable.
        assert m.try_acquire_request_slot(P, lambda: True) is True
        assert m.try_acquire_request_slot(P, lambda: True) is True
        assert m.try_acquire_request_slot(P, lambda: True) is False


class TestConcurrencyStats:
    def test_default_values(self) -> None:
        s = ConcurrencyStats()
        assert s.acquire_count == 0 and s.release_count == 0 and s.wait_count == 0

    def test_custom_values(self) -> None:
        s = ConcurrencyStats(acquire_count=10, release_count=5, wait_count=2)
        assert s.acquire_count == 10 and s.release_count == 5 and s.wait_count == 2
