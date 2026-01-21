# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common import random_generator as rng
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.timing.config import RequestCancellationConfig
from aiperf.timing.request_cancellation import RequestCancellationSimulator


def mk_cfg(rate: float | None = None, delay: float = 0.0) -> RequestCancellationConfig:
    return RequestCancellationConfig(rate=rate, delay=delay)


def mk_sim(
    rate: float | None = None, delay: float = 0.0
) -> RequestCancellationSimulator:
    return RequestCancellationSimulator(mk_cfg(rate, delay))


class TestRequestCancellation:
    @pytest.mark.parametrize("rate,expected", [
        (None, False),
        (0.0, False),
        (0.1, True),
        (50.0, True),
        (100.0, True),
    ])  # fmt: skip
    def test_enabled_status(self, rate: float | None, expected: bool) -> None:
        assert mk_sim(rate).is_cancellation_enabled == expected

    def test_zero_rate_never_cancels(self) -> None:
        sim = mk_sim(rate=0.0, delay=1.0)
        for _ in range(100):
            assert sim.next_cancellation_delay_ns() is None

    def test_full_rate_always_cancels(self) -> None:
        sim = mk_sim(rate=100.0, delay=1.0)
        for _ in range(100):
            assert sim.next_cancellation_delay_ns() == NANOS_PER_SECOND

    def test_50_percent_rate_cancels_approximately_half(self) -> None:
        rng.reset()
        rng.init(42)
        sim = mk_sim(rate=50.0, delay=1.0)
        cnt = sum(1 for _ in range(100) if sim.next_cancellation_delay_ns() is not None)
        assert 30 <= cnt <= 70

    @pytest.mark.parametrize("delay,expected_ns", [
        (0.0, 0),
        (0.5, int(0.5 * NANOS_PER_SECOND)),
        (1.0, NANOS_PER_SECOND),
        (2.5, int(2.5 * NANOS_PER_SECOND)),
        (10.0, int(10.0 * NANOS_PER_SECOND)),
    ])  # fmt: skip
    def test_delay_conversion(self, delay: float, expected_ns: int) -> None:
        assert (
            mk_sim(rate=100.0, delay=delay).next_cancellation_delay_ns() == expected_ns
        )

    @pytest.mark.parametrize("phase,expect_cancel", [
        (CreditPhase.WARMUP, False),
        (CreditPhase.PROFILING, True),
        (None, True),
    ])  # fmt: skip
    def test_phase_behavior(
        self, phase: CreditPhase | None, expect_cancel: bool
    ) -> None:
        sim = mk_sim(rate=100.0, delay=1.0)
        result = sim.next_cancellation_delay_ns(phase=phase)
        if expect_cancel:
            assert result == NANOS_PER_SECOND
        else:
            assert result is None

    def test_warmup_does_not_consume_rng(self) -> None:
        """Verify warmup phase doesn't advance the RNG state."""
        rng.reset()
        rng.init(42)
        sim = mk_sim(rate=50.0, delay=1.0)
        # Call during warmup multiple times - should not consume RNG
        for _ in range(10):
            sim.next_cancellation_delay_ns(phase=CreditPhase.WARMUP)
        # First profiling call
        r1 = sim.next_cancellation_delay_ns(phase=CreditPhase.PROFILING)

        # Reset and create new simulator - skip warmup calls
        rng.reset()
        rng.init(42)
        sim2 = mk_sim(rate=50.0, delay=1.0)
        # First profiling call without warmup calls
        r2 = sim2.next_cancellation_delay_ns(phase=CreditPhase.PROFILING)

        assert r1 == r2

    def test_same_seed_produces_reproducible_sequence(self) -> None:
        """Verify same seed produces identical cancellation decisions."""
        rng.reset()
        rng.init(42)
        sim1 = mk_sim(rate=50.0, delay=1.0)
        d1 = [sim1.next_cancellation_delay_ns() for _ in range(50)]

        rng.reset()
        rng.init(42)
        sim2 = mk_sim(rate=50.0, delay=1.0)
        d2 = [sim2.next_cancellation_delay_ns() for _ in range(50)]

        assert d1 == d2
