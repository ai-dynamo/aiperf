# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
from pydantic import TypeAdapter
from pytest import param

from aiperf.common import random_generator as rng
from aiperf.config.distributions import (
    PercentileDistribution,
    SamplingDistribution,
)

ADAPTER = TypeAdapter(SamplingDistribution)


def _percentile(sorted_vals: list[float], q: float) -> float:
    idx = min(int(q * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


class TestDiscriminator:
    @pytest.mark.parametrize(
        "payload",
        [
            param({"p50": 50000, "p99": 400000}, id="p50_p99"),
            param({"p50": 50000, "p99": 400000, "mean": 60000}, id="with_mean"),
            param({"type": "percentile", "p50": 100, "p99": 500}, id="explicit_type"),
        ],
    )  # fmt: skip
    def test_discriminator_p50_key_selects_percentile(self, payload: dict) -> None:
        dist = ADAPTER.validate_python(payload)
        assert isinstance(dist, PercentileDistribution)

    def test_discriminator_mean_with_p50_not_normal(self) -> None:
        # "mean" alone maps to Normal; p50 presence must win over mean.
        dist = ADAPTER.validate_python({"mean": 60000, "p50": 50000, "p99": 400000})
        assert isinstance(dist, PercentileDistribution)


class TestValidation:
    @pytest.mark.parametrize(
        "payload,match",
        [
            param({"p50": 400000, "p99": 50000}, "p99", id="p99_below_p50"),
            param({"p50": 100, "p99": 100}, "p99", id="p99_equal_p50"),
            param({"p50": 50000, "p99": 400000, "mean": 40000}, "mean", id="mean_below_p50"),
            param({"p50": 50000, "p99": 400000, "mean": 399999}, "infeasible|mean", id="mean_infeasible_high"),
            param({"p50": 0, "p99": 100}, "greater than 0", id="p50_zero"),
        ],
    )  # fmt: skip
    def test_invalid_targets_raise(self, payload: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ADAPTER.validate_python(payload)

    def test_non_finite_rejected(self) -> None:
        with pytest.raises(ValueError):
            ADAPTER.validate_python({"p50": 50000, "p99": float("inf")})

    @pytest.mark.parametrize(
        "p50,p99",
        [
            param(1, 1e40, id="overflow_ratio"),
            param(100, 1e35, id="inf_implied_mean"),
        ],
    )  # fmt: skip
    def test_extreme_ratio_rejected(self, p50: float, p99: float) -> None:
        with pytest.raises(ValueError, match="too extreme|finite"):
            PercentileDistribution(p50=p50, p99=p99)


class TestLogNormalMode:
    """p50 + p99 without mean fits a log-normal exactly."""

    def test_expected_value_is_lognormal_implied_mean(self) -> None:
        dist = PercentileDistribution(p50=50000, p99=400000)
        sigma = math.log(400000 / 50000) / 2.3263478740408408
        assert dist.expected_value == pytest.approx(
            50000 * math.exp(sigma**2 / 2), rel=1e-9
        )

    def test_sampled_percentiles_match_targets(self) -> None:
        dist = PercentileDistribution(p50=50000, p99=400000)
        r = rng.derive("test.percentile.lognormal")
        vals = sorted(dist.sample(r) for _ in range(100_000))
        assert _percentile(vals, 0.50) == pytest.approx(50000, rel=0.03)
        assert _percentile(vals, 0.99) == pytest.approx(400000, rel=0.05)


class TestMixtureMode:
    """p50 + p99 + mean solves a two-component mixture (the headline use-case)."""

    def test_headline_use_case_hits_all_three_targets(self) -> None:
        dist = PercentileDistribution(p50=50000, p99=400000, mean=60000)
        r = rng.derive("test.percentile.mixture")
        vals = sorted(dist.sample(r) for _ in range(200_000))
        assert _percentile(vals, 0.50) == pytest.approx(50000, rel=0.03)
        assert _percentile(vals, 0.99) == pytest.approx(400000, rel=0.05)
        assert sum(vals) / len(vals) == pytest.approx(60000, rel=0.03)

    def test_expected_value_returns_configured_mean(self) -> None:
        dist = PercentileDistribution(p50=50000, p99=400000, mean=60000)
        assert dist.expected_value == pytest.approx(60000)

    def test_mean_matching_lognormal_uses_lognormal(self) -> None:
        sigma = math.log(8.0) / 2.3263478740408408
        implied = 50000 * math.exp(sigma**2 / 2)
        dist = PercentileDistribution(p50=50000, p99=400000, mean=implied)
        r = rng.derive("test.percentile.implied")
        vals = sorted(dist.sample(r) for _ in range(100_000))
        assert _percentile(vals, 0.50) == pytest.approx(50000, rel=0.03)

    @pytest.mark.parametrize(
        "p50,p99,mean",
        [
            param(50000, 400000, 60000, id="headline"),
            param(50000, 400000, 100000, id="mid_mean"),
            param(128, 8192, 512, id="small_tokens"),
            param(1000, 2000, 1100, id="tight_spread"),
        ],
    )  # fmt: skip
    def test_solver_various_targets_all_hit(
        self, p50: float, p99: float, mean: float
    ) -> None:
        dist = PercentileDistribution(p50=p50, p99=p99, mean=mean)
        r = rng.derive(f"test.percentile.{p50}.{mean}")
        vals = sorted(dist.sample(r) for _ in range(200_000))
        assert _percentile(vals, 0.50) == pytest.approx(p50, rel=0.05)
        assert _percentile(vals, 0.99) == pytest.approx(p99, rel=0.06)
        assert sum(vals) / len(vals) == pytest.approx(mean, rel=0.04)


class TestComposition:
    def test_min_max_clamps_compose(self) -> None:
        dist = ADAPTER.validate_python(
            {"p50": 50000, "p99": 400000, "mean": 60000, "max": 600000, "min": 1000}
        )
        r = rng.derive("test.percentile.clamp")
        vals = [dist.sample(r) for _ in range(50_000)]
        assert max(vals) <= 600000
        assert min(vals) >= 1000

    def test_sample_int_returns_positive_ints(self) -> None:
        dist = PercentileDistribution(p50=50, p99=400)
        r = rng.derive("test.percentile.int")
        vals = [dist.sample_int(r) for _ in range(1000)]
        assert all(isinstance(v, int) and v >= 1 for v in vals)

    def test_serialization_round_trip(self) -> None:
        dist = PercentileDistribution(p50=50000, p99=400000, mean=60000)
        again = ADAPTER.validate_python(dist.model_dump())
        r = rng.derive("test.percentile.roundtrip")
        assert isinstance(again, PercentileDistribution)
        assert again.sample(r) > 0

    def test_repr_shows_targets(self) -> None:
        assert "percentile" in repr(PercentileDistribution(p50=100, p99=500))
