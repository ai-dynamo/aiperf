# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for distributions module."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pytest import param

from aiperf.dataset.agentic_code_gen.distributions import (
    fit_from_samples,
    lognormal_from_mean_median,
    sample_lognormal,
    sample_mixture_delay,
    sample_weibull,
)
from aiperf.dataset.agentic_code_gen.models import (
    LognormalParams,
    MixtureDelayConfig,
    WeibullParams,
)


def _weibull_mean_for(shape: float, median: float) -> float:
    """Real-space mean of a Weibull with the given shape and median."""
    scale = median / math.log(2.0) ** (1.0 / shape)
    return scale * math.gamma(1.0 + 1.0 / shape)


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

    @pytest.mark.parametrize(
        "field",
        ["mu", "sigma", "mean", "median", "min", "max"],
    )
    def test_non_finite_field_raises(self, field: str) -> None:
        # Every field is FiniteFloat, so rejection happens at the field
        # validator rather than in the model validator. gt=0.0 does not
        # exclude inf, which is the reason the annotation is needed: without
        # it LognormalParams(mean=inf, median=1.0) was accepted and derived
        # sigma=inf.
        kwargs: dict[str, float] = {"mean": 100.0, "median": 100.0}
        if field in ("mu", "sigma"):
            kwargs |= {"mu": 1.0, "sigma": 0.1}
        kwargs[field] = math.inf
        with pytest.raises(ValueError, match="value must be finite"):
            LognormalParams(**kwargs)

    def test_min_greater_than_max_raises(self) -> None:
        with pytest.raises(ValueError, match="must be <= max"):
            LognormalParams(mean=100, median=100, min=10, max=5)

    def test_roundtrip_with_lognormal_from_mean_median(self) -> None:
        explicit = lognormal_from_mean_median(mean=4500, median=2100)
        auto = LognormalParams(mean=4500, median=2100)
        assert auto.mu == pytest.approx(explicit.mu, rel=1e-9)
        assert auto.sigma == pytest.approx(explicit.sigma, rel=1e-9)


class TestWeibullParams:
    def test_weibull_params_mean_median_derives_shape_scale(self) -> None:
        true_shape = 1.5
        median = 1_000.0
        mean = _weibull_mean_for(true_shape, median)
        params = WeibullParams(distribution="weibull", mean=mean, median=median)
        assert params.shape == pytest.approx(true_shape, rel=1e-6)
        expected_scale = median / math.log(2.0) ** (1.0 / true_shape)
        assert params.scale == pytest.approx(expected_scale, rel=1e-6)

    def test_weibull_params_derived_shape_scale_reproduce_mean_median(self) -> None:
        params = WeibullParams(distribution="weibull", mean=2_500, median=1_800)
        assert params.shape is not None and params.scale is not None
        derived_median = params.scale * math.log(2.0) ** (1.0 / params.shape)
        derived_mean = params.scale * math.gamma(1.0 + 1.0 / params.shape)
        assert derived_median == pytest.approx(1_800, rel=1e-6)
        assert derived_mean == pytest.approx(2_500, rel=1e-6)

    def test_weibull_params_explicit_shape_scale_preserved(self) -> None:
        # mean/median must match what shape=2.0, scale=500.0 implies
        params = WeibullParams(
            distribution="weibull",
            shape=2.0,
            scale=500.0,
            mean=500.0 * math.gamma(1.5),
            median=500.0 * math.log(2.0) ** 0.5,
        )
        assert params.shape == 2.0
        assert params.scale == 500.0

    @pytest.mark.parametrize(
        ("mean", "median", "expected"),
        [
            param(2_500, 416.28, "mean", id="mean_disagrees"),
            param(443.11, 1_800, "median", id="median_disagrees"),
        ],
    )  # fmt: skip
    def test_weibull_params_explicit_shape_scale_summary_mismatch_raises(
        self, mean: float, median: float, expected: str
    ) -> None:
        with pytest.raises(ValueError, match=f"{expected} .* disagrees with the"):
            WeibullParams(
                distribution="weibull",
                shape=2.0,
                scale=500.0,
                mean=mean,
                median=median,
            )

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            param(
                {"shape": 1.2, "scale": 42_000.0, "mean": 40_000, "median": 25_000},
                "shape/scale",
                id="untagged_weibull_fields",
            ),
            param(
                {"shape": 1.2, "mean": 40_000, "median": 25_000},
                "shape",
                id="untagged_lone_shape",
            ),
        ],
    )  # fmt: skip
    def test_untagged_config_with_weibull_fields_is_rejected(
        self, payload: dict[str, float], expected: str
    ) -> None:
        """Without the tag the union defaults to lognormal, which would silently
        ignore shape/scale and sample the wrong family."""
        with pytest.raises(ValueError, match=f"carry {expected}, which belong to"):
            LognormalParams.model_validate(payload)

    def test_tagged_weibull_with_lognormal_fields_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="carry mu/sigma, which belong to"):
            WeibullParams.model_validate(
                {
                    "distribution": "weibull",
                    "mu": 1.0,
                    "sigma": 1.0,
                    "mean": 40_000,
                    "median": 25_000,
                }
            )

    def test_weibull_params_derived_shape_scale_survive_roundtrip(self) -> None:
        """A config generated from mean/median reparses via the explicit path."""
        params = WeibullParams(distribution="weibull", mean=2_500, median=1_800)
        reparsed = WeibullParams.model_validate(params.model_dump())
        assert reparsed.shape == params.shape
        assert reparsed.scale == params.scale

    def test_weibull_params_mean_equal_median_raises(self) -> None:
        with pytest.raises(ValueError, match="must be > median"):
            WeibullParams(distribution="weibull", mean=1_000, median=1_000)

    def test_weibull_params_mean_less_than_median_raises(self) -> None:
        with pytest.raises(ValueError, match="must be > median"):
            WeibullParams(distribution="weibull", mean=500, median=1_000)

    @pytest.mark.parametrize(
        "kwargs",
        [
            param({"shape": 1.5}, id="lone_shape"),
            param({"scale": 1_000.0}, id="lone_scale"),
        ],
    )  # fmt: skip
    def test_weibull_params_partial_shape_scale_raises(
        self, kwargs: dict[str, float]
    ) -> None:
        with pytest.raises(ValueError, match="supplied as a pair"):
            WeibullParams(distribution="weibull", mean=2_500, median=1_800, **kwargs)

    @pytest.mark.parametrize(
        "field",
        ["shape", "scale", "mean", "median", "min", "max"],
    )
    def test_weibull_params_non_finite_field_raises(self, field: str) -> None:
        # As for LognormalParams: FiniteFloat rejects at the field validator.
        # Without it an infinite mean reached the brentq solve and failed with
        # "f(a) and f(b) must have different signs", which names nothing.
        kwargs: dict[str, float] = {"mean": 2_500.0, "median": 1_800.0}
        if field in ("shape", "scale"):
            kwargs |= {"shape": 1.5, "scale": 1.0}
        kwargs[field] = math.inf
        with pytest.raises(ValueError, match="value must be finite"):
            WeibullParams(distribution="weibull", **kwargs)

    def test_weibull_params_min_greater_than_max_raises(self) -> None:
        with pytest.raises(ValueError, match="must be <= max"):
            WeibullParams(
                distribution="weibull", mean=2_500, median=1_800, min=10, max=5
            )

    def test_weibull_params_missing_distribution_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="distribution"):
            WeibullParams(mean=2_500, median=1_800)

    @pytest.mark.parametrize(
        "ratio",
        [
            param(1.0001, id="near_one"),
            param(1.2, id="mild_skew"),
            param(2.0, id="moderate_skew"),
            param(5.0, id="heavy_skew"),
            param(20.0, id="extreme_skew"),
        ],
    )  # fmt: skip
    def test_weibull_params_solve_bracket_signs_opposite(self, ratio: float) -> None:
        """brentq bracket [0.05, 3.5] must straddle the root for all r > 1."""

        def f(k: float) -> float:
            return math.gamma(1.0 + 1.0 / k) - ratio * math.log(2.0) ** (1.0 / k)

        assert f(0.05) > 0
        assert f(3.5) < 0


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


class TestSampleWeibull:
    def test_sample_weibull_returns_correct_shape(self) -> None:
        params = WeibullParams(distribution="weibull", mean=2_500, median=1_800)
        rng = np.random.default_rng(42)
        samples = sample_weibull(params, rng, size=100)
        assert samples.shape == (100,)

    def test_sample_weibull_large_draw_matches_target_mean_median(self) -> None:
        params = WeibullParams(distribution="weibull", mean=2_500, median=1_800)
        rng = np.random.default_rng(42)
        samples = sample_weibull(params, rng, size=200_000)
        assert float(np.mean(samples)) == pytest.approx(2_500, rel=0.02)
        assert float(np.median(samples)) == pytest.approx(1_800, rel=0.02)

    def test_sample_weibull_min_max_bounds_respected(self) -> None:
        params = WeibullParams(
            distribution="weibull", mean=2_500, median=1_800, min=500, max=5_000
        )
        rng = np.random.default_rng(42)
        samples = sample_weibull(params, rng, size=1_000)
        assert samples.min() >= 500
        assert samples.max() <= 5_000

    def test_sample_weibull_clip_min_floor_applied(self) -> None:
        params = WeibullParams(distribution="weibull", mean=2_500, median=1_800)
        rng = np.random.default_rng(42)
        samples = sample_weibull(params, rng, size=1_000, clip_min=1_000)
        assert samples.min() >= 1_000

    def test_sample_weibull_fixed_seed_is_deterministic(self) -> None:
        params = WeibullParams(distribution="weibull", mean=2_500, median=1_800)
        samples1 = sample_weibull(params, np.random.default_rng(7), size=100)
        samples2 = sample_weibull(params, np.random.default_rng(7), size=100)
        np.testing.assert_array_equal(samples1, samples2)


class TestMixtureDelayConfigParsing:
    def test_mixture_delay_config_bare_dicts_parse_as_lognormal(self) -> None:
        """Regression: untagged component dicts (all existing configs) stay lognormal."""
        config = MixtureDelayConfig.model_validate(
            {
                "agentic_fraction": 0.7,
                "agentic_delay": {"mean": 3_000, "median": 2_000},
                "human_delay": {"mean": 45_000, "median": 30_000},
            }
        )
        assert isinstance(config.agentic_delay, LognormalParams)
        assert isinstance(config.human_delay, LognormalParams)
        assert config.agentic_delay.distribution == "lognormal"

    def test_mixture_delay_config_tagged_weibull_parses_as_weibull(self) -> None:
        config = MixtureDelayConfig.model_validate(
            {
                "agentic_delay": {
                    "distribution": "weibull",
                    "mean": 3_000,
                    "median": 2_000,
                },
                "human_delay": {
                    "distribution": "weibull",
                    "mean": 45_000,
                    "median": 30_000,
                },
            }
        )
        assert isinstance(config.agentic_delay, WeibullParams)
        assert isinstance(config.human_delay, WeibullParams)
        assert config.agentic_delay.shape is not None

    def test_mixture_delay_config_mixed_components_parse(self) -> None:
        config = MixtureDelayConfig.model_validate(
            {
                "agentic_delay": {"mean": 3_000, "median": 2_000},
                "human_delay": {
                    "distribution": "weibull",
                    "mean": 45_000,
                    "median": 30_000,
                },
            }
        )
        assert isinstance(config.agentic_delay, LognormalParams)
        assert isinstance(config.human_delay, WeibullParams)

    def test_mixture_delay_config_roundtrips_through_dump(self) -> None:
        config = MixtureDelayConfig.model_validate(
            {
                "agentic_delay": {"mean": 3_000, "median": 2_000},
                "human_delay": {
                    "distribution": "weibull",
                    "mean": 45_000,
                    "median": 30_000,
                },
            }
        )
        reloaded = MixtureDelayConfig.model_validate(config.model_dump())
        assert isinstance(reloaded.agentic_delay, LognormalParams)
        assert isinstance(reloaded.human_delay, WeibullParams)
        assert reloaded.human_delay.shape == pytest.approx(
            config.human_delay.shape, rel=1e-9
        )


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

    def test_sample_mixture_delay_agentic_fraction_one_pins_weibull_component(
        self,
    ) -> None:
        config = MixtureDelayConfig(
            agentic_fraction=1.0,
            agentic_delay=WeibullParams(
                distribution="weibull", mean=3_000, median=2_000
            ),
            human_delay=lognormal_from_mean_median(45_000, 30_000),
        )
        rng = np.random.default_rng(42)
        samples = sample_mixture_delay(config, rng, size=2_000)
        assert float(np.median(samples)) == pytest.approx(2_000, rel=0.1)
        assert float(np.mean(samples)) == pytest.approx(3_000, rel=0.1)

    def test_sample_mixture_delay_agentic_fraction_zero_pins_weibull_component(
        self,
    ) -> None:
        config = MixtureDelayConfig(
            agentic_fraction=0.0,
            agentic_delay=lognormal_from_mean_median(3_000, 2_000),
            human_delay=WeibullParams(
                distribution="weibull", mean=45_000, median=30_000
            ),
        )
        rng = np.random.default_rng(42)
        samples = sample_mixture_delay(config, rng, size=2_000)
        assert float(np.median(samples)) == pytest.approx(30_000, rel=0.1)
        assert float(np.mean(samples)) == pytest.approx(45_000, rel=0.1)

    def test_sample_mixture_delay_weibull_components_honor_max_cap(self) -> None:
        config = MixtureDelayConfig(
            agentic_fraction=0.5,
            agentic_delay=WeibullParams(
                distribution="weibull", mean=3_000, median=2_000, max=4_000
            ),
            human_delay=WeibullParams(
                distribution="weibull", mean=45_000, median=30_000
            ),
            max=10_000,
        )
        rng = np.random.default_rng(42)
        samples = sample_mixture_delay(config, rng, size=1_000)
        assert samples.max() <= 10_000
