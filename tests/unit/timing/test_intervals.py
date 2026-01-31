# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import statistics

import pytest

from aiperf.plugin import plugins
from aiperf.plugin.enums import ArrivalPattern, PluginType
from aiperf.timing.intervals import (
    ConcurrencyBurstIntervalGenerator,
    ConstantIntervalGenerator,
    GammaIntervalGenerator,
    IntervalGeneratorConfig,
    PoissonIntervalGenerator,
)


def cfg(
    pattern: ArrivalPattern, rate: float | None = None, smooth: float | None = None
) -> IntervalGeneratorConfig:
    return IntervalGeneratorConfig(
        arrival_pattern=pattern, request_rate=rate, arrival_smoothness=smooth
    )


class TestPoissonIntervalGenerator:
    def test_init_valid_rate(self):
        gen = PoissonIntervalGenerator(cfg(ArrivalPattern.POISSON, rate=10.0))
        assert gen.rate == 10.0

    @pytest.mark.parametrize("rate", [0.0, -5.0, None])
    def test_init_raises_on_invalid_rate(self, rate):
        with pytest.raises(ValueError, match="must be set and greater than 0"):
            PoissonIntervalGenerator(cfg(ArrivalPattern.POISSON, rate=rate))

    def test_next_interval_positive(self):
        gen = PoissonIntervalGenerator(cfg(ArrivalPattern.POISSON, rate=10.0))
        assert all(gen.next_interval() > 0 for _ in range(100))

    def test_next_interval_average_matches_rate(self):
        gen = PoissonIntervalGenerator(cfg(ArrivalPattern.POISSON, rate=100.0))
        intervals = [gen.next_interval() for _ in range(10000)]
        assert abs(statistics.mean(intervals) - 0.01) / 0.01 < 0.15

    def test_set_rate_updates(self):
        gen = PoissonIntervalGenerator(cfg(ArrivalPattern.POISSON, rate=10.0))
        gen.set_rate(50.0)
        assert gen.rate == 50.0

    def test_set_rate_affects_intervals(self):
        gen = PoissonIntervalGenerator(cfg(ArrivalPattern.POISSON, rate=10.0))
        low = [gen.next_interval() for _ in range(1000)]
        gen.set_rate(100.0)
        high = [gen.next_interval() for _ in range(1000)]
        assert statistics.mean(high) < statistics.mean(low)

    @pytest.mark.parametrize("rate", [0.0, -5.0])
    def test_set_rate_raises_on_invalid(self, rate):
        gen = PoissonIntervalGenerator(cfg(ArrivalPattern.POISSON, rate=10.0))
        with pytest.raises(ValueError, match="must be > 0"):
            gen.set_rate(rate)


class TestGammaIntervalGenerator:
    def test_init_valid_rate(self):
        gen = GammaIntervalGenerator(cfg(ArrivalPattern.GAMMA, rate=10.0))
        assert gen.rate == 10.0

    def test_init_defaults_smoothness_to_one(self):
        gen = GammaIntervalGenerator(cfg(ArrivalPattern.GAMMA, rate=10.0, smooth=None))
        assert gen.smoothness == 1.0

    def test_init_custom_smoothness(self):
        gen = GammaIntervalGenerator(cfg(ArrivalPattern.GAMMA, rate=10.0, smooth=2.5))
        assert gen.smoothness == 2.5

    def test_init_raises_on_invalid_rate(self):
        with pytest.raises(ValueError, match="must be set and greater than 0"):
            GammaIntervalGenerator(cfg(ArrivalPattern.GAMMA, rate=0.0))

    def test_next_interval_positive(self):
        gen = GammaIntervalGenerator(cfg(ArrivalPattern.GAMMA, rate=10.0, smooth=2.0))
        assert all(gen.next_interval() > 0 for _ in range(100))

    def test_next_interval_average_matches_rate(self):
        gen = GammaIntervalGenerator(cfg(ArrivalPattern.GAMMA, rate=100.0, smooth=1.0))
        intervals = [gen.next_interval() for _ in range(10000)]
        assert abs(statistics.mean(intervals) - 0.01) / 0.01 < 0.15

    @pytest.mark.parametrize("smooth,expected_cv", [(1.0, 1.0), (4.0, 0.5), (9.0, 0.333), (0.25, 2.0), (0.5, 1.414), (16.0, 0.25)])  # fmt: skip
    def test_cv_matches_gamma_formula(self, smooth: float, expected_cv: float):
        gen = GammaIntervalGenerator(
            cfg(ArrivalPattern.GAMMA, rate=100.0, smooth=smooth)
        )
        intervals = [gen.next_interval() for _ in range(10000)]
        cv = statistics.stdev(intervals) / statistics.mean(intervals)
        assert abs(cv - expected_cv) < expected_cv * 0.20

    def test_cv_monotonically_decreases(self):
        smoothness_vals = [0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 16.0, 25.0]
        cvs = []
        for s in smoothness_vals:
            gen = GammaIntervalGenerator(
                cfg(ArrivalPattern.GAMMA, rate=100.0, smooth=s)
            )
            intervals = [gen.next_interval() for _ in range(5000)]
            cvs.append(statistics.stdev(intervals) / statistics.mean(intervals))
        for i in range(1, len(cvs)):
            assert cvs[i] < cvs[i - 1]

    def test_set_rate_updates(self):
        gen = GammaIntervalGenerator(cfg(ArrivalPattern.GAMMA, rate=10.0, smooth=2.0))
        gen.set_rate(100.0)
        assert gen.rate == 100.0
        intervals = [gen.next_interval() for _ in range(5000)]
        assert abs(statistics.mean(intervals) - 0.01) / 0.01 < 0.20


class TestConstantIntervalGenerator:
    def test_init_valid_rate(self):
        gen = ConstantIntervalGenerator(cfg(ArrivalPattern.CONSTANT, rate=10.0))
        assert gen.rate == 10.0

    def test_init_raises_on_invalid_rate(self):
        with pytest.raises(ValueError, match="must be set and greater than 0"):
            ConstantIntervalGenerator(cfg(ArrivalPattern.CONSTANT, rate=0.0))

    def test_next_interval_fixed(self):
        gen = ConstantIntervalGenerator(cfg(ArrivalPattern.CONSTANT, rate=10.0))
        assert all(gen.next_interval() == 0.1 for _ in range(100))

    def test_set_rate_updates(self):
        gen = ConstantIntervalGenerator(cfg(ArrivalPattern.CONSTANT, rate=10.0))
        assert gen.next_interval() == 0.1
        gen.set_rate(50.0)
        assert gen.rate == 50.0 and gen.next_interval() == 0.02

    def test_set_rate_raises_on_invalid(self):
        gen = ConstantIntervalGenerator(cfg(ArrivalPattern.CONSTANT, rate=10.0))
        with pytest.raises(ValueError, match="must be > 0"):
            gen.set_rate(0.0)


class TestConcurrencyBurstIntervalGenerator:
    def test_rate_always_zero(self):
        gen = ConcurrencyBurstIntervalGenerator(cfg(ArrivalPattern.CONCURRENCY_BURST))
        assert gen.rate == 0.0

    def test_next_interval_returns_zero(self):
        gen = ConcurrencyBurstIntervalGenerator(cfg(ArrivalPattern.CONCURRENCY_BURST))
        assert all(gen.next_interval() == 0 for _ in range(100))

    def test_set_rate_noop(self):
        gen = ConcurrencyBurstIntervalGenerator(cfg(ArrivalPattern.CONCURRENCY_BURST))
        for r in [100.0, 0.0, -5.0]:
            gen.set_rate(r)
        assert gen.rate == 0.0 and gen.next_interval() == 0


class TestIntervalGeneratorPlugin:
    @pytest.mark.parametrize("pattern,cls", [(ArrivalPattern.POISSON, PoissonIntervalGenerator), (ArrivalPattern.GAMMA, GammaIntervalGenerator), (ArrivalPattern.CONSTANT, ConstantIntervalGenerator), (ArrivalPattern.CONCURRENCY_BURST, ConcurrencyBurstIntervalGenerator)])  # fmt: skip
    def test_plugin_creates_correct_type(self, pattern: ArrivalPattern, cls: type):
        rate = 10.0 if pattern != ArrivalPattern.CONCURRENCY_BURST else None
        GeneratorClass = plugins.get_class(PluginType.ARRIVAL_PATTERN, pattern)
        assert isinstance(GeneratorClass(config=cfg(pattern, rate=rate)), cls)


class TestEdgeCases:
    @pytest.mark.parametrize("rate,expected", [(1_000_000.0, 1e-6), (0.001, 1000.0)])
    def test_extreme_rates(self, rate, expected):
        gen = ConstantIntervalGenerator(cfg(ArrivalPattern.CONSTANT, rate=rate))
        assert gen.next_interval() == expected
