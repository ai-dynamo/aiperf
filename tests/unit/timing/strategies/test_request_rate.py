# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common import random_generator as rng
from aiperf.plugin.enums import ArrivalPattern
from aiperf.timing.intervals import (
    ConcurrencyBurstIntervalGenerator,
    ConstantIntervalGenerator,
    IntervalGeneratorConfig,
    PoissonIntervalGenerator,
)
from tests.unit.timing.conftest import OrchestratorHarness, get_session_stats


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
