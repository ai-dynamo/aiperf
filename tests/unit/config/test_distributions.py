# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for aiperf.config.distributions.

Covers:
- Discriminator routing from raw YAML-like dicts/scalars to correct types
- Scalar coercion to FixedDistribution
- Sampling correctness for every distribution type
- Validation errors (constraint violations, extra fields, invalid types)
- Removed types are rejected
- SequenceDistributionEntry integration with distribution fields
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError
from pytest import param

from aiperf.common import random_generator as rng
from aiperf.config.distributions import (
    Distribution,
    EmpiricalDistribution,
    EmpiricalPoint,
    FixedDistribution,
    LogNormalDistribution,
    MultimodalDistribution,
    NormalDistribution,
    PeakEntry,
    SamplingDistribution,
)
from aiperf.config.types import SequenceDistributionEntry

# TypeAdapter for the discriminated union
_TA = TypeAdapter(SamplingDistribution)

# Sample size for statistical tests
_N = 10_000

# Relative tolerance for statistical assertions (mean within 15%)
_RTOL = 0.15


# ============================================================
# Helpers
# ============================================================


def _get_rng() -> rng.RandomGenerator:
    return rng.derive("test.distributions")


def _sample_n(dist: Distribution, n: int = _N) -> list[float]:
    gen = _get_rng()
    return [dist.sample(gen) for _ in range(n)]


def _sample_int_n(dist: Distribution, n: int = _N) -> list[int]:
    gen = _get_rng()
    return [dist.sample_int(gen) for _ in range(n)]


# ============================================================
# 1. Discriminator Routing
# ============================================================


class TestDiscriminatorRouting:
    def test_int_scalar_routes_to_fixed(self) -> None:
        assert isinstance(_TA.validate_python(512), FixedDistribution)

    def test_float_scalar_routes_to_fixed(self) -> None:
        assert isinstance(_TA.validate_python(512.5), FixedDistribution)

    def test_stddev_routes_to_normal(self) -> None:
        assert isinstance(
            _TA.validate_python({"mean": 512, "stddev": 50}), NormalDistribution
        )

    def test_stddev_zero_routes_to_normal(self) -> None:
        assert isinstance(
            _TA.validate_python({"mean": 512, "stddev": 0}), NormalDistribution
        )

    def test_median_routes_to_lognormal(self) -> None:
        assert isinstance(
            _TA.validate_python({"mean": 512, "median": 400}), LogNormalDistribution
        )

    def test_peaks_routes_to_multimodal(self) -> None:
        d = _TA.validate_python(
            {
                "peaks": [
                    {"mean": 128, "stddev": 20, "weight": 60},
                    {"mean": 2048, "median": 1800, "weight": 40},
                ],
            }
        )
        assert isinstance(d, MultimodalDistribution)

    def test_peaks_without_weight_routes_to_multimodal(self) -> None:
        d = _TA.validate_python(
            {
                "peaks": [{"mean": 100, "stddev": 0}, {"mean": 900, "stddev": 0}],
            }
        )
        assert isinstance(d, MultimodalDistribution)

    def test_points_routes_to_empirical(self) -> None:
        d = _TA.validate_python({"points": [{"value": 128}, {"value": 512}]})
        assert isinstance(d, EmpiricalDistribution)

    def test_extra_type_field_rejected(self) -> None:
        with pytest.raises((ValidationError, ValueError)):
            _TA.validate_python({"mean": 512, "stddev": 50, "type": "normal"})

    def test_mean_alone_routes_to_normal_with_stddev_zero(self) -> None:
        d = _TA.validate_python({"mean": 512})
        assert isinstance(d, NormalDistribution)
        assert d.mean == 512
        assert d.stddev == 0.0

    def test_unknown_keys_rejected(self) -> None:
        with pytest.raises((ValidationError, ValueError)):
            _TA.validate_python({"alpha": 1.5})

    def test_already_constructed_fixed_passes_through(self) -> None:
        d = FixedDistribution(value=512)
        assert _TA.validate_python(d) is d

    def test_already_constructed_normal_passes_through(self) -> None:
        d = NormalDistribution(mean=512, stddev=50)
        assert _TA.validate_python(d) is d

    def test_already_constructed_lognormal_passes_through(self) -> None:
        d = LogNormalDistribution(mean=512, median=400)
        assert _TA.validate_python(d) is d

    def test_already_constructed_multimodal_passes_through(self) -> None:
        d = MultimodalDistribution(
            peaks=[
                PeakEntry(
                    distribution=NormalDistribution(mean=100, stddev=0), weight=60
                ),
                PeakEntry(
                    distribution=NormalDistribution(mean=900, stddev=0), weight=40
                ),
            ]
        )
        assert isinstance(_TA.validate_python(d), MultimodalDistribution)

    def test_already_constructed_empirical_passes_through(self) -> None:
        d = EmpiricalDistribution(
            points=[EmpiricalPoint(value=128), EmpiricalPoint(value=512)]
        )
        assert _TA.validate_python(d) is d


# ============================================================
# 2. Removed types are rejected
# ============================================================


class TestRemovedTypes:
    @pytest.mark.parametrize(
        "data",
        [
            param({"type": "uniform", "min": 256, "max": 1024}, id="uniform"),
            param({"type": "exponential", "mean": 500}, id="exponential"),
            param({"type": "zipf", "alpha": 1.5}, id="zipf"),
            param(
                {"type": "clamped", "distribution": {"mean": 512, "stddev": 50}, "max": 2048},
                id="clamped",
            ),
            param(
                {
                    "type": "mixture",
                    "components": [
                        {"distribution": {"mean": 128, "stddev": 20}, "weight": 60},
                        {"distribution": {"mean": 2048, "stddev": 200}, "weight": 40},
                    ],
                },
                id="mixture",
            ),
        ],
    )  # fmt: skip
    def test_removed_type_is_rejected(self, data: dict) -> None:
        with pytest.raises((ValidationError, ValueError)):
            _TA.validate_python(data)


# ============================================================
# 3. FixedDistribution
# ============================================================


class TestFixedDistribution:
    def test_sample_returns_exact_value(self) -> None:
        d = FixedDistribution(value=512.0)
        gen = _get_rng()
        assert d.sample(gen) == 512.0

    def test_expected_value_equals_value(self) -> None:
        d = FixedDistribution(value=256.0)
        assert d.expected_value == 256.0

    def test_mean_alias(self) -> None:
        d = FixedDistribution(value=128.0)
        assert d.mean == 128.0

    def test_sample_int_returns_int(self) -> None:
        d = FixedDistribution(value=512.0)
        gen = _get_rng()
        assert isinstance(d.sample_int(gen), int)

    def test_all_samples_identical(self) -> None:
        d = FixedDistribution(value=42.0)
        samples = _sample_n(d, 100)
        assert all(s == 42.0 for s in samples)

    @pytest.mark.parametrize(
        "value, expected",
        [
            param(512, 512.0, id="int"),
            param(512.5, 512.5, id="float"),
            param(0, 0.0, id="zero"),
            param(-10, -10.0, id="negative"),
        ],
    )  # fmt: skip
    def test_scalar_coerces_to_fixed(self, value: int | float, expected: float) -> None:
        d = _TA.validate_python(value)
        assert isinstance(d, FixedDistribution)
        assert d.value == expected

    def test_infinite_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FixedDistribution(value=float("inf"))

    def test_repr(self) -> None:
        assert repr(FixedDistribution(value=512.0)) == "fixed(512)"


# ============================================================
# 4. NormalDistribution
# ============================================================


class TestNormalDistribution:
    def test_sampled_mean_close_to_expected(self) -> None:
        d = NormalDistribution(mean=512, stddev=50)
        samples = _sample_n(d)
        assert abs(sum(samples) / len(samples) - 512) < 512 * _RTOL

    def test_expected_value_is_mean(self) -> None:
        d = NormalDistribution(mean=256, stddev=30)
        assert d.expected_value == 256

    def test_samples_non_negative(self) -> None:
        d = NormalDistribution(mean=10, stddev=20)
        samples = _sample_n(d)
        assert all(s >= 0 for s in samples)

    def test_deterministic_when_stddev_zero(self) -> None:
        d = NormalDistribution(mean=512, stddev=0)
        gen = _get_rng()
        assert all(d.sample(gen) == 512 for _ in range(100))

    def test_sample_int_always_ge_one(self) -> None:
        d = NormalDistribution(mean=1, stddev=5)
        samples = _sample_int_n(d)
        assert all(s >= 1 for s in samples)

    def test_negative_stddev_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NormalDistribution(mean=512, stddev=-1)

    def test_repr_with_stddev(self) -> None:
        assert "normal" in repr(NormalDistribution(mean=512, stddev=50))

    def test_repr_without_stddev(self) -> None:
        assert repr(NormalDistribution(mean=512, stddev=0)) == "normal(512)"


# ============================================================
# 5. LogNormalDistribution
# ============================================================


class TestLogNormalDistribution:
    def test_sampled_mean_close_to_expected(self) -> None:
        d = LogNormalDistribution(mean=512, median=400)
        samples = _sample_n(d)
        assert abs(sum(samples) / len(samples) - 512) < 512 * _RTOL

    def test_expected_value_is_mean(self) -> None:
        d = LogNormalDistribution(mean=512, median=400)
        assert d.expected_value == 512

    def test_all_samples_positive(self) -> None:
        d = LogNormalDistribution(mean=512, median=400)
        samples = _sample_n(d)
        assert all(s > 0 for s in samples)

    def test_deterministic_when_median_equals_mean(self) -> None:
        d = LogNormalDistribution(mean=512, median=512)
        gen = _get_rng()
        assert all(d.sample(gen) == 512 for _ in range(100))

    def test_sample_int_always_ge_one(self) -> None:
        d = LogNormalDistribution(mean=512, median=400)
        samples = _sample_int_n(d)
        assert all(s >= 1 for s in samples)

    def test_median_greater_than_mean_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LogNormalDistribution(mean=400, median=512)

    def test_zero_mean_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LogNormalDistribution(mean=0, median=0)

    def test_repr(self) -> None:
        assert "lognormal" in repr(LogNormalDistribution(mean=512, median=400))


# ============================================================
# 6. MultimodalDistribution
# ============================================================


class TestMultimodalDistribution:
    def _make_2(self, w0: float = 60.0) -> MultimodalDistribution:
        return _TA.validate_python(
            {
                "peaks": [
                    {"mean": 100, "stddev": 0, "weight": w0},
                    {"mean": 900, "stddev": 0, "weight": 100.0 - w0},
                ],
            }
        )

    def test_samples_only_first_peak_when_full_weight(self) -> None:
        d = self._make_2(w0=100.0)
        samples = _sample_int_n(d, 100)
        assert all(s == 100 for s in samples)

    def test_samples_only_second_peak_when_zero_weight(self) -> None:
        d = _TA.validate_python(
            {
                "peaks": [
                    {"mean": 100, "stddev": 0, "weight": 0.001},
                    {"mean": 900, "stddev": 0, "weight": 999},
                ]
            }
        )
        samples = _sample_int_n(d, 100)
        assert sum(1 for s in samples if s == 900) > 95

    def test_equal_weights_produce_roughly_equal_mix(self) -> None:
        d = self._make_2(w0=50.0)
        samples = _sample_int_n(d)
        count_low = sum(1 for s in samples if s < 500)
        assert 0.40 * _N < count_low < 0.60 * _N

    def test_expected_value_is_weighted_mean(self) -> None:
        d = self._make_2(w0=60.0)
        assert abs(d.expected_value - (0.6 * 100 + 0.4 * 900)) < 1.0

    def test_default_weight_is_one(self) -> None:
        d = _TA.validate_python(
            {"peaks": [{"mean": 100, "stddev": 0}, {"mean": 900, "stddev": 0}]}
        )
        assert d.peaks[0].weight == 1.0
        assert d.peaks[1].weight == 1.0

    def test_requires_at_least_two_peaks(self) -> None:
        with pytest.raises(ValidationError):
            _TA.validate_python({"peaks": [{"mean": 100, "stddev": 0}]})

    def test_three_peaks_accepted(self) -> None:
        d = _TA.validate_python(
            {
                "peaks": [
                    {"mean": 100, "stddev": 0, "weight": 50},
                    {"mean": 500, "stddev": 0, "weight": 30},
                    {"mean": 900, "stddev": 0, "weight": 20},
                ]
            }
        )
        assert isinstance(d, MultimodalDistribution)
        assert len(d.peaks) == 3
        assert abs(d.expected_value - (0.5 * 100 + 0.3 * 500 + 0.2 * 900)) < 1.0

    def test_peaks_can_be_mixed_types(self) -> None:
        d = _TA.validate_python(
            {
                "peaks": [
                    {"mean": 128, "stddev": 20, "weight": 70},
                    {"mean": 2048, "median": 1800, "weight": 30},
                ]
            }
        )
        assert isinstance(d.peaks[0].distribution, NormalDistribution)
        assert isinstance(d.peaks[1].distribution, LogNormalDistribution)

    def test_sample_int_always_ge_one(self) -> None:
        d = self._make_2(w0=50.0)
        samples = _sample_int_n(d)
        assert all(s >= 1 for s in samples)

    def test_repr_contains_multimodal(self) -> None:
        assert "multimodal" in repr(self._make_2())

    def test_repr_contains_percentages(self) -> None:
        assert "60%" in repr(self._make_2(w0=60.0))


# ============================================================
# 7. EmpiricalDistribution
# ============================================================


class TestEmpiricalDistribution:
    def _make(self) -> EmpiricalDistribution:
        return _TA.validate_python(
            {
                "points": [
                    {"value": 128, "weight": 40},
                    {"value": 512, "weight": 35},
                    {"value": 2048, "weight": 20},
                    {"value": 8192, "weight": 5},
                ]
            }
        )

    def test_sampled_mean_close_to_expected(self) -> None:
        d = self._make()
        samples = _sample_n(d)
        assert (
            abs(sum(samples) / len(samples) - d.expected_value)
            < d.expected_value * _RTOL
        )

    def test_expected_value_is_weighted_mean(self) -> None:
        d = self._make()
        total = 40 + 35 + 20 + 5
        expected = (128 * 40 + 512 * 35 + 2048 * 20 + 8192 * 5) / total
        assert abs(d.expected_value - expected) < 0.01

    def test_samples_are_exact_point_values(self) -> None:
        d = self._make()
        valid = {128.0, 512.0, 2048.0, 8192.0}
        samples = _sample_n(d)
        assert all(s in valid for s in samples)

    def test_sample_int_always_ge_one(self) -> None:
        d = self._make()
        samples = _sample_int_n(d)
        assert all(s >= 1 for s in samples)

    def test_empty_points_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EmpiricalDistribution(points=[])

    def test_single_point_weight_defaults_to_one(self) -> None:
        d = _TA.validate_python({"points": [{"value": 256}]})
        assert isinstance(d, EmpiricalDistribution)
        assert d.points[0].weight == 1.0

    def test_repr(self) -> None:
        assert "empirical" in repr(self._make())


# ============================================================
# 8. SequenceDistributionEntry integration
# ============================================================


class TestSequenceDistributionEntry:
    def test_isl_osl_as_scalars(self) -> None:
        entry = SequenceDistributionEntry(isl=512, osl=256, probability=100)
        assert isinstance(entry.isl, FixedDistribution)
        assert isinstance(entry.osl, FixedDistribution)

    def test_isl_as_normal_dict(self) -> None:
        entry = SequenceDistributionEntry(
            isl={"mean": 512, "stddev": 50}, osl=256, probability=100
        )
        assert isinstance(entry.isl, NormalDistribution)

    def test_isl_stddev_shorthand_creates_normal(self) -> None:
        entry = SequenceDistributionEntry(
            isl=512, isl_stddev=50, osl=256, probability=100
        )
        assert isinstance(entry.isl, NormalDistribution)
        assert entry.isl.mean == 512.0
        assert entry.isl.stddev == 50.0

    def test_osl_stddev_shorthand_creates_normal(self) -> None:
        entry = SequenceDistributionEntry(
            isl=512, osl=256, osl_stddev=25, probability=100
        )
        assert isinstance(entry.osl, NormalDistribution)
        assert entry.osl.mean == 256.0
        assert entry.osl.stddev == 25.0
