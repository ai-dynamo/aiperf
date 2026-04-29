# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for distributions module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aiperf.dataset.agentic_code_gen.distributions import (
    fit_from_samples,
    lognormal_from_mean_median,
    sample_lognormal,
    sample_mixture_delay,
)
from aiperf.dataset.agentic_code_gen.models import LognormalParams, MixtureDelayConfig


class TestLognormalFromMeanMedian:
    def test_computes_mu_from_median(self) -> None:
        params = lognormal_from_mean_median(mean=67_000, median=54_000)
        assert params.mu == pytest.approx(math.log(54_000), rel=1e-6)

    def test_computes_sigma_from_ratio(self) -> None:
        params = lognormal_from_mean_median(mean=67_000, median=54_000)
        expected_sigma = math.sqrt(2.0 * math.log(67_000 / 54_000))
        assert params.sigma == pytest.approx(expected_sigma, rel=1e-6)

    def test_stores_mean_and_median(self) -> None:
        params = lognormal_from_mean_median(mean=600, median=350)
        assert params.mean == 600
        assert params.median == 350

    def test_equal_mean_median_gives_zero_sigma(self) -> None:
        params = lognormal_from_mean_median(mean=100, median=100)
        assert params.sigma == 0.0

    def test_negative_mean_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            lognormal_from_mean_median(mean=-1, median=100)

    def test_mean_less_than_median_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= median"):
            lognormal_from_mean_median(mean=50, median=100)

    @pytest.mark.parametrize(
        "mean,median",
        [
            (67_000, 54_000),
            (4_500, 2_100),
            (600, 350),
            (3_000, 2_000),
            (45_000, 30_000),
        ],
    )
    def test_plan_table_values(self, mean: int, median: int) -> None:
        params = lognormal_from_mean_median(mean=mean, median=median)
        assert params.mu == pytest.approx(math.log(median), rel=1e-3)
        assert params.mean == mean
        assert params.median == median


class TestLognormalParamsAutoCompute:
    def test_mu_sigma_computed_from_mean_median(self) -> None:
        params = LognormalParams(mean=67_000, median=54_000)
        assert params.mu == pytest.approx(math.log(54_000), rel=1e-6)
        expected_sigma = math.sqrt(2.0 * math.log(67_000 / 54_000))
        assert params.sigma == pytest.approx(expected_sigma, rel=1e-6)

    def test_explicit_mu_sigma_preserved(self) -> None:
        params = LognormalParams(mu=10.0, sigma=0.5, mean=67_000, median=54_000)
        assert params.mu == 10.0
        assert params.sigma == 0.5

    def test_equal_mean_median_gives_zero_sigma(self) -> None:
        params = LognormalParams(mean=100, median=100)
        assert params.sigma == 0.0

    def test_mean_less_than_median_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= median"):
            LognormalParams(mean=50, median=100)

    def test_explicit_mu_sigma_still_validates_mean_median(self) -> None:
        with pytest.raises(ValueError, match="must be >= median"):
            LognormalParams(mu=1.0, sigma=0.1, mean=50, median=100)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mu": 1.0},
            {"sigma": 0.1},
        ],
    )
    def test_partial_mu_sigma_raises(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="supplied as a pair"):
            LognormalParams(mean=100, median=100, **kwargs)

    def test_non_finite_explicit_mu_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="mu must be finite"):
            LognormalParams(mu=math.inf, sigma=0.1, mean=100, median=100)
        with pytest.raises(ValueError, match="sigma must be finite"):
            LognormalParams(mu=1.0, sigma=math.inf, mean=100, median=100)

    def test_min_greater_than_max_raises(self) -> None:
        with pytest.raises(ValueError, match="must be <= max"):
            LognormalParams(mean=100, median=100, min=10, max=5)

    def test_roundtrip_with_lognormal_from_mean_median(self) -> None:
        explicit = lognormal_from_mean_median(mean=4500, median=2100)
        auto = LognormalParams(mean=4500, median=2100)
        assert auto.mu == pytest.approx(explicit.mu, rel=1e-9)
        assert auto.sigma == pytest.approx(explicit.sigma, rel=1e-9)


class TestFitFromSamples:
    def test_recovers_known_distribution(self) -> None:
        rng = np.random.default_rng(42)
        true_mu, true_sigma = 7.0, 0.5
        samples = rng.lognormal(true_mu, true_sigma, size=10_000)
        params = fit_from_samples(samples)
        assert params.mu == pytest.approx(true_mu, abs=0.05)
        assert params.sigma == pytest.approx(true_sigma, abs=0.05)

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            fit_from_samples(np.array([1.0]))

    def test_filters_non_positive(self) -> None:
        samples = np.array([0, -1, 10, 20, 30])
        params = fit_from_samples(samples)
        assert params.mean > 0


class TestSampleLognormal:
    def test_returns_correct_shape(self) -> None:
        params = lognormal_from_mean_median(mean=600, median=350)
        rng = np.random.default_rng(42)
        samples = sample_lognormal(params, rng, size=100)
        assert samples.shape == (100,)

    def test_clipping_works(self) -> None:
        params = LognormalParams(mean=600, median=350, max=3750)
        rng = np.random.default_rng(42)
        samples = sample_lognormal(params, rng, size=1000, clip_min=30)
        assert samples.min() >= 30
        assert samples.max() <= 3750


class TestSampleMixtureDelay:
    def test_returns_correct_shape(self) -> None:
        config = MixtureDelayConfig(
            agentic_fraction=0.7,
            agentic_delay=lognormal_from_mean_median(3_000, 2_000),
            human_delay=lognormal_from_mean_median(45_000, 30_000),
        )
        rng = np.random.default_rng(42)
        samples = sample_mixture_delay(config, rng, size=1000)
        assert samples.shape == (1000,)

    def test_bimodal_distribution(self) -> None:
        config = MixtureDelayConfig(
            agentic_fraction=0.7,
            agentic_delay=lognormal_from_mean_median(3_000, 2_000),
            human_delay=lognormal_from_mean_median(45_000, 30_000),
        )
        rng = np.random.default_rng(42)
        samples = sample_mixture_delay(config, rng, size=10_000)
        fast = samples[samples < 10_000]
        slow = samples[samples >= 10_000]
        assert len(fast) > len(slow)
        assert len(fast) / len(samples) == pytest.approx(0.7, abs=0.1)

    def test_all_agentic(self) -> None:
        config = MixtureDelayConfig(
            agentic_fraction=1.0,
            agentic_delay=lognormal_from_mean_median(3_000, 2_000),
            human_delay=lognormal_from_mean_median(45_000, 30_000),
        )
        rng = np.random.default_rng(42)
        samples = sample_mixture_delay(config, rng, size=1000)
        assert float(np.median(samples)) < 10_000

    def test_component_and_mixture_max_bounds_limit_samples(self) -> None:
        config = MixtureDelayConfig(
            agentic_fraction=0.5,
            agentic_delay=LognormalParams(mean=3_000, median=2_000, max=4_000),
            human_delay=LognormalParams(mean=45_000, median=30_000, max=50_000),
            max=10_000,
        )
        rng = np.random.default_rng(42)
        samples = sample_mixture_delay(config, rng, size=1000)
        assert samples.max() <= 10_000
