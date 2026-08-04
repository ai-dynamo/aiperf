# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import math
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from scipy import stats

from aiperf.common import random_generator as rng
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.intervals import (
    ConcurrencyBurstIntervalGenerator,
    ConstantIntervalGenerator,
    IntervalGeneratorConfig,
    PoissonIntervalGenerator,
)
from aiperf.timing.strategies.request_rate import RequestRateStrategy
from tests.unit.timing.conftest import OrchestratorHarness, get_session_stats


@pytest.mark.skip(
    reason="Timing tests require time_traveler_no_patch_sleep and proper "
    "synchronization between time.time_ns(), looptime, and time_traveler patches."
)
@pytest.mark.asyncio
@pytest.mark.slow
class TestPoissonDistribution:
    async def run_poisson(
        self, rate, count, seed, create_orchestrator_harness
    ) -> list[int]:
        rng.reset()
        rng.init(seed)
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", 1) for i in range(count)],
            request_count=count,
            request_rate=rate,
            random_seed=seed,
        )
        await h.run_with_auto_return()
        return [c.issued_at_ns for c in h.sent_credits]

    async def run_event_counts(self, rate, count, seed, interval, harness_factory):
        ts = await self.run_poisson(rate, count, seed, harness_factory)
        ts_sec = np.array(ts) / NANOS_PER_SECOND
        start, end = ts_sec[0], ts_sec[-1]
        n_intervals = int((end - start) / interval)
        counts = [
            np.sum(
                (ts_sec >= start + i * interval) & (ts_sec < start + (i + 1) * interval)
            )
            for i in range(n_intervals)
        ]
        return np.mean(counts), np.array(counts)

    async def test_exponential_distribution(
        self, create_orchestrator_harness, time_traveler
    ) -> None:
        ts = await self.run_poisson(200.0, 10_000, 42, create_orchestrator_harness)
        expected_mean = 1.0 / 200.0
        assert len(ts) == 10_000
        intervals = (
            np.array([ts[i] - ts[i - 1] for i in range(1, len(ts))]) / NANOS_PER_SECOND
        )
        actual_mean, actual_std = np.mean(intervals), np.std(intervals)
        assert abs(actual_mean - expected_mean) < expected_mean * 0.2
        assert abs(actual_std - expected_mean) < expected_mean * 0.3
        assert abs(actual_std / actual_mean - 1.0) < 0.2
        prop_below = np.sum(intervals < actual_mean) / len(intervals)
        assert abs(prop_below - (1 - math.exp(-1))) < 0.1

    async def test_independence(
        self, create_orchestrator_harness, time_traveler
    ) -> None:
        ts = await self.run_poisson(150.0, 10_000, 123, create_orchestrator_harness)
        intervals = (
            np.array([ts[i] - ts[i - 1] for i in range(1, len(ts))]) / NANOS_PER_SECOND
        )
        corr = np.corrcoef(intervals[:-1], intervals[1:])[0, 1]
        assert abs(corr) < 0.2

    async def test_event_counts_mean(
        self, create_orchestrator_harness, time_traveler
    ) -> None:
        mean, _ = await self.run_event_counts(
            1_000, 25_000, 789, 0.5, create_orchestrator_harness
        )
        assert abs(mean - 500) < 150

    async def test_event_counts_variance(
        self, create_orchestrator_harness, time_traveler
    ) -> None:
        mean, counts = await self.run_event_counts(
            1_000, 25_000, 789, 0.5, create_orchestrator_harness
        )
        assert abs(np.var(counts) - mean) < mean * 0.4

    async def test_dispersion_index(
        self, create_orchestrator_harness, time_traveler
    ) -> None:
        mean, counts = await self.run_event_counts(
            1_000, 25_000, 789, 0.5, create_orchestrator_harness
        )
        assert abs(np.var(counts) / mean - 1.0) < 0.4

    async def test_ks_test(self, create_orchestrator_harness, time_traveler) -> None:
        mean, counts = await self.run_event_counts(
            1_000, 25_000, 789, 0.5, create_orchestrator_harness
        )
        max_count = int(np.max(counts))
        theoretical = stats.poisson.pmf(np.arange(max_count + 1), mean)
        unique, freq = np.unique(counts, return_counts=True)
        empirical = np.zeros(max_count + 1)
        empirical[unique] = freq / len(counts)
        ks = np.max(np.abs(np.cumsum(empirical) - np.cumsum(theoretical)))
        assert ks < 1.36 / np.sqrt(len(counts))

    async def test_ratio_property(
        self, create_orchestrator_harness, time_traveler
    ) -> None:
        mean, counts = await self.run_event_counts(
            1_000, 25_000, 789, 0.5, create_orchestrator_harness
        )
        unique, freq = np.unique(counts, return_counts=True)
        probs = freq / len(counts)
        ratios = []
        for i in range(len(unique) - 1):
            if probs[i] > 0.01 and probs[i + 1] > 0.01:
                expected = mean / (unique[i] + 1)
                if expected > 0.1:
                    ratios.append(abs(probs[i + 1] / probs[i] - expected) / expected)
        assert np.mean(ratios) < 0.5


@pytest.mark.asyncio
class TestMaxConcurrency:
    @pytest.mark.parametrize(
        "concurrency,rate,has_stats",
        [(5, None, True), (10, None, True), (None, 10.0, False)],
    )  # fmt: skip
    async def test_session_stats_tracked_only_with_concurrency_limit(
        self,
        create_orchestrator_harness,
        time_traveler,
        concurrency,
        rate,
        has_stats,
    ) -> None:
        """ConcurrencyStats are only tracked when a concurrency limit is set."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", 1) for i in range(10)],
            request_count=10,
            concurrency=concurrency,
            request_rate=rate,
        )
        await h.run_with_auto_return()
        s = get_session_stats(h.orchestrator)
        assert (s is not None) == has_stats

    async def test_all_requests_acquire_concurrency_slot(
        self, create_orchestrator_harness, time_traveler
    ) -> None:
        """Each request acquires and releases a concurrency slot."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", 1) for i in range(5)],
            request_count=5,
            concurrency=3,
        )
        await h.run_with_auto_return()
        s = get_session_stats(h.orchestrator)
        assert s is not None
        assert s.acquire_count == 5
        assert s.release_count == 5

    async def test_no_wait_when_concurrency_exceeds_requests(
        self, create_orchestrator_harness, time_traveler
    ) -> None:
        """No waits occur when concurrency limit exceeds request count."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", 1) for i in range(5)],
            request_count=5,
            concurrency=10,
        )
        await h.run_with_auto_return()
        s = get_session_stats(h.orchestrator)
        assert s is not None
        assert s.wait_count == 0


@pytest.mark.asyncio
async def test_request_rate_update_wakes_pending_sleep() -> None:
    class FakeIntervalGenerator:
        def __init__(self, config):
            self.rate = config.request_rate

        def next_interval(self) -> float:
            return 10.0 if self.rate == 0.1 else 0.05

        def set_rate(self, new_rate: float) -> None:
            self.rate = new_rate

    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = int(time.perf_counter() * NANOS_PER_SECOND)
    conversation_source = MagicMock()
    conversation_source.next.return_value.build_first_turn.return_value = object()
    credit_issuer = MagicMock()
    credit_issuer.try_issue_credit = AsyncMock(return_value=False)
    stop_checker = MagicMock()
    stop_checker.can_start_new_session.return_value = True

    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=0.1,
        total_expected_requests=1,
    )
    with patch(
        "aiperf.timing.strategies.request_rate.plugins.get_class",
        return_value=FakeIntervalGenerator,
    ):
        strategy = RequestRateStrategy(
            config=config,
            conversation_source=conversation_source,
            scheduler=MagicMock(),
            stop_checker=stop_checker,
            credit_issuer=credit_issuer,
            lifecycle=lifecycle,
        )

    task = asyncio.create_task(strategy.execute_phase())
    await asyncio.sleep(0)

    strategy.set_request_rate(100.0)
    await asyncio.wait_for(task, timeout=1.0)

    credit_issuer.try_issue_credit.assert_awaited_once()


@pytest.mark.asyncio
async def test_high_res_pacer_rate_update_reschedules_before_issuing_credit() -> None:
    class FakeIntervalGenerator:
        def __init__(self, config) -> None:
            self.rate = config.request_rate

        def next_interval(self) -> float:
            return 0.05 if self.rate == 0.1 else 0.001

        def set_rate(self, new_rate: float) -> None:
            self.rate = new_rate

    class FakePacer:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = False
            self.deadlines: list[float] = []

        async def sleep_until(self, deadline_perf_s: float) -> None:
            self.deadlines.append(deadline_perf_s)
            if len(self.deadlines) == 1:
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

        def close(self) -> None:
            pass

    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = int(time.perf_counter() * NANOS_PER_SECOND)
    conversation_source = MagicMock()
    conversation_source.next.return_value.build_first_turn.return_value = object()
    credit_issuer = MagicMock()
    credit_issuer.try_issue_credit = AsyncMock(return_value=False)
    stop_checker = MagicMock()
    stop_checker.can_start_new_session.return_value = True
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=0.1,
        total_expected_requests=1,
    )
    with patch(
        "aiperf.timing.strategies.request_rate.plugins.get_class",
        return_value=FakeIntervalGenerator,
    ):
        strategy = RequestRateStrategy(
            config=config,
            conversation_source=conversation_source,
            scheduler=MagicMock(),
            stop_checker=stop_checker,
            credit_issuer=credit_issuer,
            lifecycle=lifecycle,
        )

    pacer = FakePacer()
    with patch.object(strategy, "_create_high_res_pacer", return_value=pacer):
        task = asyncio.create_task(strategy.execute_phase())
        await pacer.started.wait()
        strategy.set_request_rate(1000.0)
        await asyncio.wait_for(task, timeout=1.0)

    assert pacer.cancelled
    assert len(pacer.deadlines) == 2
    assert pacer.deadlines[1] < pacer.deadlines[0]
    credit_issuer.try_issue_credit.assert_awaited_once()


@pytest.mark.asyncio
async def test_high_res_pacer_os_error_falls_back_to_event_loop_timers() -> None:
    class FakeIntervalGenerator:
        def __init__(self, config) -> None:
            self.rate = config.request_rate

        def next_interval(self) -> float:
            return 0.001

    class FailingPacer:
        def __init__(self) -> None:
            self.closed = False

        async def sleep_until(self, _deadline_perf_s: float) -> None:
            raise OSError("timerfd failed")

        def close(self) -> None:
            self.closed = True

    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = int(time.perf_counter() * NANOS_PER_SECOND)
    conversation_source = MagicMock()
    conversation_source.next.return_value.build_first_turn.return_value = object()
    credit_issuer = MagicMock()
    credit_issuer.try_issue_credit = AsyncMock(return_value=False)
    stop_checker = MagicMock()
    stop_checker.can_start_new_session.return_value = True
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=1000.0,
        total_expected_requests=1,
    )
    with patch(
        "aiperf.timing.strategies.request_rate.plugins.get_class",
        return_value=FakeIntervalGenerator,
    ):
        strategy = RequestRateStrategy(
            config=config,
            conversation_source=conversation_source,
            scheduler=MagicMock(),
            stop_checker=stop_checker,
            credit_issuer=credit_issuer,
            lifecycle=lifecycle,
        )

    pacer = FailingPacer()
    with patch.object(strategy, "_create_high_res_pacer", return_value=pacer):
        await asyncio.wait_for(strategy.execute_phase(), timeout=1.0)

    assert pacer.closed
    credit_issuer.try_issue_credit.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_phase_cancelled_mid_pace_drains_pacer_and_closes() -> None:
    """A cancel (even a repeated one) during active pacing must not orphan the pacer.

    ``_wait_for_pacer_or_rate_update`` cancels its two child tasks in a
    ``finally`` block and then awaits them. A second ``Task.cancel()`` — eager
    cancellation, or an external shutdown racing a first cancel — sets
    ``_must_cancel`` and would otherwise interrupt that await, abandoning the
    pacer's in-flight sleep as a pending task ("Task was destroyed but it is
    pending!") and leaving its tick event stale for the next caller. The
    shielded gather keeps the drain running to completion regardless.
    """

    class FakeIntervalGenerator:
        def __init__(self, config) -> None:
            self.rate = config.request_rate

        def next_interval(self) -> float:
            return 0.001

    class FakePacer:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.sleep_cancelled = False
            self.closed = False

        async def sleep_until(self, _deadline_perf_s: float) -> None:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.sleep_cancelled = True
                raise

        def close(self) -> None:
            self.closed = True

    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = int(time.perf_counter() * NANOS_PER_SECOND)
    conversation_source = MagicMock()
    conversation_source.next.return_value.build_first_turn.return_value = object()
    credit_issuer = MagicMock()
    credit_issuer.try_issue_credit = AsyncMock(return_value=False)
    stop_checker = MagicMock()
    stop_checker.can_start_new_session.return_value = True
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=1000.0,
        total_expected_requests=1,
    )
    with patch(
        "aiperf.timing.strategies.request_rate.plugins.get_class",
        return_value=FakeIntervalGenerator,
    ):
        strategy = RequestRateStrategy(
            config=config,
            conversation_source=conversation_source,
            scheduler=MagicMock(),
            stop_checker=stop_checker,
            credit_issuer=credit_issuer,
            lifecycle=lifecycle,
        )

    pacer = FakePacer()
    with patch.object(strategy, "_create_high_res_pacer", return_value=pacer):
        task = asyncio.create_task(strategy.execute_phase())
        await pacer.started.wait()
        # Two cancels: the second sets ``_must_cancel``, so the cleanup await is
        # re-cancelled before the child tasks have been drained.
        task.cancel()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The shielded gather keeps running after the outer cancel lands; give
        # it the loop iterations it needs to finish draining.
        for _ in range(10):
            if pacer.sleep_cancelled:
                break
            await asyncio.sleep(0)

    assert pacer.sleep_cancelled, "pacer sleep was orphaned instead of drained"
    assert pacer.closed, "pacer was not closed on the cancellation path"
    credit_issuer.try_issue_credit.assert_not_awaited()


def _build_minimal_strategy() -> RequestRateStrategy:
    class FakeIntervalGenerator:
        def __init__(self, config):
            self.rate = config.request_rate

        def next_interval(self) -> float:
            return 0.01

    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=100.0,
        total_expected_requests=1,
    )
    with patch(
        "aiperf.timing.strategies.request_rate.plugins.get_class",
        return_value=FakeIntervalGenerator,
    ):
        return RequestRateStrategy(
            config=config,
            conversation_source=MagicMock(),
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=MagicMock(),
            lifecycle=MagicMock(),
        )


class TestHighResPacerFactory:
    """Tests for the high-resolution pacer selection factory."""

    def test_returns_none_when_disabled(self) -> None:
        """No pacer is created when HIGH_RES_TIMER is off."""
        strategy = _build_minimal_strategy()
        with patch(
            "aiperf.timing.strategies.request_rate.Environment.TIMING.HIGH_RES_TIMER",
            False,
        ):
            assert strategy._create_high_res_pacer() is None

    async def test_returns_pacer_when_enabled(self) -> None:
        """A pacer is created (and closes cleanly) when HIGH_RES_TIMER is on.

        Must run inside an event loop: both pacers capture the running loop.
        """
        strategy = _build_minimal_strategy()
        with patch(
            "aiperf.timing.strategies.request_rate.Environment.TIMING.HIGH_RES_TIMER",
            True,
        ):
            pacer = strategy._create_high_res_pacer()
        assert pacer is not None
        pacer.close()


class TestConcurrencyBurstGenerator:
    def test_returns_zero(self) -> None:
        cfg = IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.CONCURRENCY_BURST, request_rate=None
        )
        gen = ConcurrencyBurstIntervalGenerator(cfg)
        for _ in range(10):
            assert gen.next_interval() == 0

    def test_rate_is_zero(self) -> None:
        gen = ConcurrencyBurstIntervalGenerator(
            IntervalGeneratorConfig(arrival_pattern=ArrivalPattern.CONCURRENCY_BURST)
        )
        assert gen.rate == 0.0

    def test_set_rate_noop(self) -> None:
        gen = ConcurrencyBurstIntervalGenerator(
            IntervalGeneratorConfig(arrival_pattern=ArrivalPattern.CONCURRENCY_BURST)
        )
        gen.set_rate(100.0)
        assert gen.rate == 0.0


class TestPoissonGenerator:
    def test_none_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            PoissonIntervalGenerator(
                IntervalGeneratorConfig(
                    arrival_pattern=ArrivalPattern.POISSON, request_rate=None
                )
            )

    @pytest.mark.parametrize("rate", [0, -1, -5.0, -100.5, 0.0])
    def test_invalid_rate_raises(self, rate) -> None:
        with pytest.raises(ValueError):
            PoissonIntervalGenerator(
                IntervalGeneratorConfig(
                    arrival_pattern=ArrivalPattern.POISSON, request_rate=rate
                )
            )

    @pytest.mark.parametrize("rate", [0.1, 1.0, 10.5, 100, 1000])
    def test_valid_rate(self, rate) -> None:
        rng.reset()
        rng.init(42)
        gen = PoissonIntervalGenerator(
            IntervalGeneratorConfig(
                arrival_pattern=ArrivalPattern.POISSON, request_rate=rate
            )
        )
        for _ in range(10):
            assert gen.next_interval() > 0


class TestConstantGenerator:
    def test_none_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            ConstantIntervalGenerator(
                IntervalGeneratorConfig(
                    arrival_pattern=ArrivalPattern.CONSTANT, request_rate=None
                )
            )

    @pytest.mark.parametrize("rate", [0, -1, -5.0, -100.5, 0.0])
    def test_invalid_rate_raises(self, rate) -> None:
        with pytest.raises(ValueError):
            ConstantIntervalGenerator(
                IntervalGeneratorConfig(
                    arrival_pattern=ArrivalPattern.CONSTANT, request_rate=rate
                )
            )

    @pytest.mark.parametrize("rate", [0.1, 1.0, 10.5, 100, 1000])
    def test_valid_rate(self, rate) -> None:
        gen = ConstantIntervalGenerator(
            IntervalGeneratorConfig(
                arrival_pattern=ArrivalPattern.CONSTANT, request_rate=rate
            )
        )
        expected = 1.0 / rate
        for _ in range(10):
            assert gen.next_interval() == expected


@pytest.mark.asyncio
class TestConstantArrival:
    async def test_constant_rate_with_concurrency_limit(
        self, create_orchestrator_harness, time_traveler
    ) -> None:
        """Constant arrival pattern works correctly with concurrency limiting."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", 1) for i in range(2)],
            request_rate=1.0,
            request_count=2,
            arrival_pattern=ArrivalPattern.CONSTANT,
            concurrency=1,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == 2
