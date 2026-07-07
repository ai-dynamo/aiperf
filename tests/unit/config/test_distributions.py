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

from typing import Any

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


class TestDistributionBaseContract:
    def test_sample_raw_error_names_operation_and_subclass(self) -> None:
        dist = Distribution()
        with pytest.raises(NotImplementedError) as exc_info:
            dist._sample_raw(_get_rng())

        message = str(exc_info.value)
        assert "Distribution" in message
        assert "_sample_raw(rng)" in message

    def test_expected_value_error_names_operation_and_subclass(self) -> None:
        dist = Distribution()
        with pytest.raises(NotImplementedError) as exc_info:
            _ = dist.expected_value

        message = str(exc_info.value)
        assert "Distribution" in message
        assert "expected_value" in message

    def test_repr_error_names_operation_and_subclass(self) -> None:
        dist = Distribution()
        with pytest.raises(NotImplementedError) as exc_info:
            repr(dist)

        message = str(exc_info.value)
        assert "Distribution" in message
        assert "__repr__" in message


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

    def test_extra_type_field_accepted_when_matches(self) -> None:
        # `type:` is now optional; when present and matching, it's accepted and
        # stripped before subclass validation. See test_distribution_explicit_type.py
        # for the full strict-but-tolerant matrix.
        d = _TA.validate_python({"mean": 512, "stddev": 50, "type": "normal"})
        assert isinstance(d, NormalDistribution)
        assert d.mean == 512
        assert d.stddev == 50

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

    def test_all_zero_weight_peaks_rejected(self) -> None:
        # An all-zero-weight mixture has no normalizing constant, so
        # expected_value/mean/repr/sample would divide by sum-of-weights=0
        # and raise ZeroDivisionError. Reject it at validation time instead
        # of accepting a distribution that crashes on first use.
        with pytest.raises(ValidationError, match="positive total weight"):
            _TA.validate_python(
                {
                    "peaks": [
                        {"mean": 100, "stddev": 0, "weight": 0},
                        {"mean": 900, "stddev": 0, "weight": 0},
                    ]
                }
            )

    def test_all_zero_weight_peaks_rejected_direct_construction(self) -> None:
        # Direct model construction (not via the discriminated-union adapter)
        # is the path that reproduced the crash: validation accepted it, then
        # `.expected_value`/`.mean`/`repr(...)` raised ZeroDivisionError.
        with pytest.raises(ValidationError, match="positive total weight"):
            MultimodalDistribution(
                peaks=[
                    PeakEntry(
                        distribution=NormalDistribution(mean=100, stddev=0), weight=0.0
                    ),
                    PeakEntry(
                        distribution=NormalDistribution(mean=900, stddev=0), weight=0.0
                    ),
                ]
            )

    def test_single_zero_weight_peak_still_allowed(self) -> None:
        # A single 0-weight peak just drops out of the mixture as long as the
        # total is positive; expected_value collapses to the surviving peak.
        d = _TA.validate_python(
            {
                "peaks": [
                    {"mean": 100, "stddev": 0, "weight": 0},
                    {"mean": 900, "stddev": 0, "weight": 40},
                ]
            }
        )
        assert isinstance(d, MultimodalDistribution)
        assert d.expected_value == 900.0
        assert d.mean == 900.0
        assert "multimodal" in repr(d)

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


# ============================================================
# 9. Malformed inputs surface as pydantic.ValidationError
# ============================================================
#
# Regression for the sentinel-tag discriminator: `_distribution_discriminator`
# returns an unregistered tag (None) for unrecognized inputs instead of raising
# builtins.ValueError. A raw ValueError from a callable Discriminator escapes
# pydantic's wrapping — leaking the wrong exception type and dropping the field
# location — and makes the union's `custom_error_message` dead code. Returning
# None routes to no union member, so pydantic emits `invalid_distribution_type`
# with the offending field's loc. See distributions.py `_distribution_discriminator`.


def _isl_config(isl: Any) -> dict:
    """Minimal AIPerfConfig payload with the synthetic dataset's `isl` set to `isl`."""
    return {
        "benchmark": {
            "models": ["x"],
            "endpoint": {"urls": ["http://x/v1"]},
            "datasets": [
                {"name": "main", "type": "synthetic", "entries": 10, "isl": isl}
            ],
            "phases": [
                {"name": "p", "type": "concurrency", "duration": 10, "concurrency": 1}
            ],
        }
    }


class TestMalformedDistributionRaisesValidationError:
    @pytest.mark.parametrize(
        "bad",
        [
            param("512", id="bare-string"),
            param([1, 2, 3], id="list"),
            param({"foo": 1}, id="dict-unknown-keys"),
            param({"type": "gaussian", "mean": 512}, id="dict-typo-type"),
            param({"min": 0, "max": 100}, id="bounds-only-no-distribution-key"),
        ],
    )  # fmt: skip
    def test_typeadapter_raises_validationerror_not_value_error(self, bad: Any) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _TA.validate_python(bad)
        errors = exc_info.value.errors()
        assert any(e["type"] == "invalid_distribution_type" for e in errors), errors
        assert "Invalid distribution" in str(exc_info.value)

    @pytest.mark.parametrize(
        "bad",
        [
            param("512", id="bare-string"),
            param([1, 2, 3], id="list"),
            param({"foo": 1}, id="dict-unknown-keys"),
            param({"type": "gaussian", "mean": 512}, id="dict-typo-type"),
        ],
    )  # fmt: skip
    def test_public_api_raises_validationerror_naming_field(self, bad: Any) -> None:
        # Guards the AIPerfConfig.model_validate contract: malformed distributions
        # must raise pydantic.ValidationError with the field loc, never a bare
        # builtins.ValueError leaking from the discriminator.
        from aiperf.config.config import AIPerfConfig

        with pytest.raises(ValidationError) as exc_info:
            AIPerfConfig.model_validate(_isl_config(bad))
        assert any(
            e["type"] == "invalid_distribution_type" and "isl" in e["loc"]
            for e in exc_info.value.errors()
        ), exc_info.value.errors()


class TestBoolRejectedAsDistribution:
    """`bool` is an `int` subclass; without the explicit guard `isl: true` would
    silently coerce to FixedDistribution(value=1.0). It must be rejected."""

    @pytest.mark.parametrize(
        "value", [param(True, id="true"), param(False, id="false")]
    )
    def test_bool_scalar_rejected_by_typeadapter(self, value: bool) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _TA.validate_python(value)
        assert any(
            e["type"] == "invalid_distribution_type" for e in exc_info.value.errors()
        )

    @pytest.mark.parametrize(
        "value", [param(True, id="true"), param(False, id="false")]
    )
    def test_bool_rejected_via_public_api(self, value: bool) -> None:
        from aiperf.config.config import AIPerfConfig

        with pytest.raises(ValidationError):
            AIPerfConfig.model_validate(_isl_config(value))


class TestValidDistributionFormsStillParse:
    """Pin every distribution form the code actually supports so the malformed-input
    fix cannot regress the happy path. Note: string shorthands and non-finite (`inf`)
    values are NOT supported forms — the discriminator handles only scalars, dicts,
    and already-built instances, and finite bounds/values are enforced downstream."""

    @pytest.mark.parametrize(
        "payload, expected_cls",
        [
            param(512, FixedDistribution, id="int-scalar"),
            param(512.5, FixedDistribution, id="float-scalar"),
            param(0, FixedDistribution, id="zero-scalar"),
            param(-10, FixedDistribution, id="negative-scalar"),
            param({"value": 256}, FixedDistribution, id="value-key"),
            param({"mean": 512}, NormalDistribution, id="mean-alone-stddev-defaults-0"),
            param({"mean": 512, "stddev": 50}, NormalDistribution, id="mean-stddev"),
            param({"mean": 512, "median": 400}, LogNormalDistribution, id="mean-median"),
            param(
                {"peaks": [{"mean": 128, "stddev": 20}, {"mean": 2048, "median": 1800}]},
                MultimodalDistribution,
                id="peaks",
            ),
            param(
                {"points": [{"value": 128, "weight": 40}, {"value": 512, "weight": 60}]},
                EmpiricalDistribution,
                id="points",
            ),
            param({"type": "fixed", "value": 256}, FixedDistribution, id="explicit-fixed"),
            param(
                {"type": "normal", "mean": 512, "stddev": 50},
                NormalDistribution,
                id="explicit-normal",
            ),
            param(
                {"mean": 512, "stddev": 50, "min": 100, "max": 900},
                NormalDistribution,
                id="normal-with-bounds",
            ),
            param(
                {"value": 512, "min": 100, "max": 900},
                FixedDistribution,
                id="fixed-with-bounds",
            ),
        ],
    )  # fmt: skip
    def test_valid_form_routes_to_expected_class(
        self, payload: Any, expected_cls: type
    ) -> None:
        result = _TA.validate_python(payload)
        assert isinstance(result, expected_cls)
