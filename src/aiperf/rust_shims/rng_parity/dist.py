# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuous samplers and configured sampling distributions for the parity RNG.

Two layers, both byte-exact:

1. **Continuous primitives** — ``rand_distr`` 0.5.1's ziggurat samplers for the
   exponential and normal distributions and the Marsaglia-Tsang gamma sampler. Ported
   from ``exponential.rs``, ``normal.rs``, ``gamma.rs``, ``utils.rs`` (``ziggurat``) and
   ``rand-0.9.4/src/distr/float.rs`` (``into_float_with_exponent`` / ``Open01``). These
   operate on any generator exposing ``next_u64()`` and ``random()``.

2. **Configured distributions** — the ``SamplingDistribution`` family and
   ``SequenceLengthDistribution`` ported from ``rust/aiperf/src/rng/dist.rs``.

The ziggurat tables live in the generated :mod:`aiperf.common.rng_parity.ziggurat_tables`.
"""

from __future__ import annotations

import math

from aiperf.common.rng_parity import ziggurat_tables as zt
from aiperf.common.rng_parity.errors import RngError

__all__ = [
    "EmpiricalDistribution",
    "EmpiricalPoint",
    "FixedDistribution",
    "LogNormalDistribution",
    "MultimodalDistribution",
    "NormalDistribution",
    "PeakEntry",
    "SamplingDistribution",
    "SequenceLengthDistribution",
    "SequenceLengthPair",
    "positive_integer_from_f64",
    "sample_exp1",
    "sample_gamma",
    "sample_standard_normal",
]

_M64 = (1 << 64) - 1
_EPS = 2.0**-52  # f64::EPSILON
_I64_UPPER_EXCLUSIVE_AS_F64 = 9_223_372_036_854_775_808.0
_PROBABILITY_SUM_REL_TOLERANCE = 1.0e-6
_PROBABILITY_SUM_ABS_TOLERANCE = 1.0e-6


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


# ------------------------------------------------------------------ float helpers
def _into_float_with_exponent(bits: int, exponent: int) -> float:
    """Combine a 52-bit fraction with a biased exponent (``float.rs`` IntoFloat).

    ``bits`` supplies the low 52 fraction bits; the result lies in
    ``[2**exponent, 2**(exponent+1))``.
    """
    exponent_bits = (1023 + exponent) << 52
    combined = (bits & ((1 << 52) - 1)) | exponent_bits
    return _f64_from_bits(combined)


def _f64_from_bits(bits: int) -> float:
    import struct

    return struct.unpack("<d", struct.pack("<Q", bits & 0xFFFFFFFFFFFFFFFF))[0]


def _open01(gen) -> float:
    """Sample ``(0, 1)`` via the transmute method (``float.rs`` Open01 for f64)."""
    value = gen.random_u64()
    fraction = value >> (64 - 52)
    return _into_float_with_exponent(fraction, 0) - (1.0 - _EPS / 2.0)


# ------------------------------------------------------------------ ziggurat
def _ziggurat(gen, symmetric: bool, x_tab, f_tab, pdf, zero_case) -> float:
    """ZIGNOR ziggurat sampler (``rand_distr-0.5.1/src/utils.rs:30``)."""
    while True:
        bits = gen.random_u64()
        i = bits & 0xFF
        if symmetric:
            # value in [2,4) minus 3 -> [-1,1)
            u = _into_float_with_exponent(bits >> 12, 1) - 3.0
        else:
            u = _into_float_with_exponent(bits >> 12, 0) - (1.0 - _EPS / 2.0)
        x = u * x_tab[i]
        test_x = abs(x) if symmetric else x
        if test_x < x_tab[i + 1]:
            return x
        if i == 0:
            return zero_case(gen, u)
        if f_tab[i + 1] + (f_tab[i] - f_tab[i + 1]) * gen.random() < pdf(x):
            return x


def sample_exp1(gen) -> float:
    """Standard exponential ``Exp(1)`` via ziggurat (``exponential.rs`` ``Exp1``)."""

    def pdf(x: float) -> float:
        return math.exp(-x)

    def zero_case(g, _u: float) -> float:
        return zt.ZIG_EXP_R - math.log(g.random())

    return _ziggurat(gen, False, zt.ZIG_EXP_X, zt.ZIG_EXP_F, pdf, zero_case)


def sample_standard_normal(gen) -> float:
    """Standard normal ``N(0,1)`` via ziggurat (``normal.rs`` ``StandardNormal``)."""

    def pdf(x: float) -> float:
        return math.exp(-x * x / 2.0)

    def zero_case(g, u: float) -> float:
        x = 1.0
        y = 0.0
        while -2.0 * y < x * x:
            x_ = _open01(g)
            y_ = _open01(g)
            x = math.log(x_) / zt.ZIG_NORM_R
            y = math.log(y_)
        return (x - zt.ZIG_NORM_R) if u < 0.0 else (zt.ZIG_NORM_R - x)

    return _ziggurat(gen, True, zt.ZIG_NORM_X, zt.ZIG_NORM_F, pdf, zero_case)


def _sample_gamma_large(gen, shape: float, scale: float) -> float:
    """Marsaglia-Tsang gamma for ``shape > 1`` (``gamma.rs`` ``GammaLargeShape``)."""
    d = shape - (1.0 / 3.0)
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = sample_standard_normal(gen)
        v_cbrt = 1.0 + c * x
        if v_cbrt <= 0.0:
            continue
        v = v_cbrt * v_cbrt * v_cbrt
        u = _open01(gen)
        x_sqr = x * x
        if u < 1.0 - 0.0331 * x_sqr * x_sqr or math.log(u) < 0.5 * x_sqr + d * (
            1.0 - v + math.log(v)
        ):
            return d * v * scale


def sample_gamma(gen, shape: float, scale: float) -> float:
    """Gamma sampler dispatching on ``shape`` (``gamma.rs`` ``Gamma``).

    ``shape == 1`` -> ``Exp1 * scale``; ``shape < 1`` -> boosted large-shape sampler;
    ``shape > 1`` -> Marsaglia-Tsang.
    """
    if shape == 1.0:
        # Gamma(1, scale) == Exp(1/scale); Exp::sample == Exp1 * (1/lambda) == Exp1*scale.
        return sample_exp1(gen) * scale
    if shape < 1.0:
        u = _open01(gen)
        return _sample_gamma_large(gen, shape + 1.0, scale) * (u ** (1.0 / shape))
    return _sample_gamma_large(gen, shape, scale)


def positive_integer_from_f64(value: float, what: str) -> int:
    """``max(1, value as i64)`` with range/finite guards (``generator.rs``)."""
    if not _is_finite(value) or value >= _I64_UPPER_EXCLUSIVE_AS_F64:
        raise RngError.invalid_parameter(what, value)
    return max(1, math.trunc(value))


# ------------------------------------------------------------------ configured dists
def _clamp(value: float, min_v, max_v) -> float:
    out = value
    if min_v is not None and out < min_v:
        out = min_v
    if max_v is not None and out > max_v:
        out = max_v
    return out


def _validate_finite(value: float, what: str) -> None:
    if not _is_finite(value):
        raise RngError.invalid_parameter(what, value)


def _validate_weight(weight: float, what: str) -> None:
    if not _is_finite(weight) or weight < 0.0:
        raise RngError.invalid_parameter(what, weight)


def _validate_bounds(min_v, max_v) -> None:
    if min_v is not None:
        _validate_finite(min_v, "min")
    if max_v is not None:
        _validate_finite(max_v, "max")
    if min_v is not None and max_v is not None and min_v > max_v:
        raise RngError.invalid_bounds(min_v, max_v)


def _cumulative_weights(weights: list[float]) -> tuple[list[float], float]:
    total = 0.0
    out = []
    for weight in weights:
        if not _is_finite(weight) or weight < 0.0:
            raise RngError.invalid_weights("weights must be finite and non-negative")
        total += weight
        if not _is_finite(total):
            raise RngError.invalid_weights("weights must have a finite sum")
        out.append(total)
    if total <= 0.0:
        raise RngError.invalid_weights("weights must sum to a positive value")
    return out, total


def _weighted_index_for_random(cumulative_weights, total: float, random: float) -> int:
    r = random * total
    idx = _partition_point(cumulative_weights, r)
    return min(idx, len(cumulative_weights) - 1)


def _partition_point(sorted_values, r: float) -> int:
    lo, hi = 0, len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] <= r:
            lo = mid + 1
        else:
            hi = mid
    return lo


class PeakEntry:
    """A weighted component in a :class:`MultimodalDistribution`."""

    __slots__ = ("distribution", "weight")

    def __init__(self, distribution: SamplingDistribution, weight: float) -> None:
        _validate_weight(weight, "peak weight")
        self.distribution = distribution
        self.weight = weight


class EmpiricalPoint:
    """A weighted value in an :class:`EmpiricalDistribution`."""

    __slots__ = ("value", "weight")

    def __init__(self, value: float, weight: float) -> None:
        _validate_finite(value, "empirical value")
        _validate_weight(weight, "empirical weight")
        if weight <= 0.0:
            raise RngError.invalid_weights("empirical weights must be positive")
        self.value = value
        self.weight = weight


class FixedDistribution:
    """A constant-valued distribution."""

    __slots__ = ("value", "min", "max")

    def __init__(self, value: float) -> None:
        _validate_finite(value, "fixed value")
        self.value = value
        self.min = None
        self.max = None

    def sample(self, gen) -> float:
        return _clamp(self.value, self.min, self.max)

    def expected_value(self) -> float:
        return self.value


class NormalDistribution:
    """Positive normal distribution parameterized by mean and stddev."""

    __slots__ = ("mean", "stddev", "min", "max")

    def __init__(self, mean: float, stddev: float) -> None:
        if mean < 0.0:
            raise RngError.invalid_parameter("normal mean", mean)
        _validate_finite(mean, "normal mean")
        if stddev < 0.0:
            raise RngError.invalid_parameter("normal stddev", stddev)
        _validate_finite(stddev, "normal stddev")
        self.mean = mean
        self.stddev = stddev
        self.min = None
        self.max = None

    def sample(self, gen) -> float:
        raw = (
            self.mean
            if self.stddev <= 0.0
            else gen.sample_positive_normal(self.mean, self.stddev)
        )
        return _clamp(raw, self.min, self.max)

    def expected_value(self) -> float:
        return self.mean


class LogNormalDistribution:
    """Log-normal distribution parameterized by real-space mean and median."""

    __slots__ = ("mean", "median", "min", "max")

    def __init__(self, mean: float, median: float) -> None:
        if mean <= 0.0:
            raise RngError.invalid_parameter("lognormal mean", mean)
        if median <= 0.0:
            raise RngError.invalid_parameter("lognormal median", median)
        _validate_finite(mean, "lognormal mean")
        _validate_finite(median, "lognormal median")
        if median > mean:
            raise RngError.invalid_parameter("lognormal median", median)
        self.mean = mean
        self.median = median
        self.min = None
        self.max = None

    def _sigma(self) -> float:
        if self.median >= self.mean:
            return 0.0
        return math.sqrt(2.0 * math.log(self.mean / self.median))

    def sample(self, gen) -> float:
        sigma = self._sigma()
        if sigma <= 0.0:
            raw = self.mean
        else:
            raw = math.exp(
                gen.sample_normal(
                    math.log(self.median), sigma, float("-inf"), float("inf")
                )
            )
        return _clamp(raw, self.min, self.max)

    def expected_value(self) -> float:
        return self.mean


class MultimodalDistribution:
    """Weighted mixture of two or more distributions."""

    __slots__ = ("peaks", "cumulative_weights", "total_weight", "min", "max")

    def __init__(self, peaks: list[PeakEntry]) -> None:
        if len(peaks) < 2:
            raise RngError.empty_sequence("peaks")
        cumulative, total = _cumulative_weights([peak.weight for peak in peaks])
        self.peaks = peaks
        self.cumulative_weights = cumulative
        self.total_weight = total
        self.min = None
        self.max = None

    def sample(self, gen) -> float:
        idx = _weighted_index_for_random(
            self.cumulative_weights, self.total_weight, gen.random()
        )
        raw = self.peaks[idx].distribution.sample(gen)
        return _clamp(raw, self.min, self.max)

    def expected_value(self) -> float:
        return sum(
            peak.weight / self.total_weight * peak.distribution.expected_value()
            for peak in self.peaks
        )


class EmpiricalDistribution:
    """Discrete weighted empirical distribution."""

    __slots__ = ("points", "cumulative_weights", "total_weight", "min", "max")

    def __init__(self, points: list[EmpiricalPoint]) -> None:
        if len(points) == 0:
            raise RngError.empty_sequence("points")
        cumulative, total = _cumulative_weights([point.weight for point in points])
        self.points = points
        self.cumulative_weights = cumulative
        self.total_weight = total
        self.min = None
        self.max = None

    def sample(self, gen) -> float:
        idx = _weighted_index_for_random(
            self.cumulative_weights, self.total_weight, gen.random()
        )
        return _clamp(self.points[idx].value, self.min, self.max)

    def expected_value(self) -> float:
        return sum(
            point.weight / self.total_weight * point.value for point in self.points
        )


class SamplingDistribution:
    """Five-way sampling distribution used by AIPerf configuration (``dist.rs``)."""

    __slots__ = ("_inner",)

    def __init__(self, inner) -> None:
        self._inner = inner

    @property
    def inner(self):
        return self._inner

    @classmethod
    def fixed(cls, value: float) -> SamplingDistribution:
        return cls(FixedDistribution(value))

    @classmethod
    def normal(cls, mean: float, stddev: float) -> SamplingDistribution:
        return cls(NormalDistribution(mean, stddev))

    @classmethod
    def lognormal(cls, mean: float, median: float) -> SamplingDistribution:
        return cls(LogNormalDistribution(mean, median))

    @classmethod
    def multimodal(cls, peaks: list[PeakEntry]) -> SamplingDistribution:
        return cls(MultimodalDistribution(peaks))

    @classmethod
    def empirical(cls, points: list[EmpiricalPoint]) -> SamplingDistribution:
        return cls(EmpiricalDistribution(points))

    def with_bounds(self, min_v, max_v) -> SamplingDistribution:
        _validate_bounds(min_v, max_v)
        self._inner.min = min_v
        self._inner.max = max_v
        return self

    def sample(self, gen) -> float:
        return self._inner.sample(gen)

    def sample_int(self, gen) -> int:
        return positive_integer_from_f64(
            math.ceil(self.sample(gen)) * 1.0, "distribution integer sample"
        )

    def expected_value(self) -> float:
        return self._inner.expected_value()


class SequenceLengthPair:
    """One ISL/OSL pair with probability and optional normal stddevs (``dist.rs``)."""

    __slots__ = (
        "input_seq_len",
        "output_seq_len",
        "probability",
        "input_seq_len_stddev",
        "output_seq_len_stddev",
    )

    def __init__(
        self,
        input_seq_len: int,
        output_seq_len: int,
        probability: float,
        input_seq_len_stddev: float = 0.0,
        output_seq_len_stddev: float = 0.0,
    ) -> None:
        if input_seq_len <= 0:
            raise RngError.invalid_parameter("input_seq_len", float(input_seq_len))
        if output_seq_len <= 0:
            raise RngError.invalid_parameter("output_seq_len", float(output_seq_len))
        if not (0.0 <= probability <= 100.0) or not _is_finite(probability):
            raise RngError.invalid_parameter("probability", probability)
        if input_seq_len_stddev < 0.0 or not _is_finite(input_seq_len_stddev):
            raise RngError.invalid_parameter(
                "input_seq_len_stddev", input_seq_len_stddev
            )
        if output_seq_len_stddev < 0.0 or not _is_finite(output_seq_len_stddev):
            raise RngError.invalid_parameter(
                "output_seq_len_stddev", output_seq_len_stddev
            )
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.probability = probability
        self.input_seq_len_stddev = input_seq_len_stddev
        self.output_seq_len_stddev = output_seq_len_stddev


class SequenceLengthDistribution:
    """Probability distribution over sequence-length pairs (``dist.rs``)."""

    __slots__ = ("pairs", "cumulative_probs")

    def __init__(self, pairs: list[SequenceLengthPair]) -> None:
        if len(pairs) == 0:
            raise RngError.empty_sequence("sequence pairs")
        total = sum(pair.probability for pair in pairs)
        if not _probability_sum_is_close(total):
            raise RngError.invalid_probability_sum(total)
        cumulative = []
        acc = 0.0
        for pair in pairs:
            acc += pair.probability / 100.0
            cumulative.append(acc)
        self.pairs = pairs
        self.cumulative_probs = cumulative

    def _index_for_random(self, r: float) -> int:
        idx = _partition_point(self.cumulative_probs, r)
        return min(idx, len(self.pairs) - 1)

    def _sample_pair_at(self, idx: int, gen) -> tuple[int, int]:
        pair = self.pairs[idx]
        isl = (
            gen.sample_positive_normal_integer(
                float(pair.input_seq_len), pair.input_seq_len_stddev
            )
            if pair.input_seq_len_stddev > 0.0
            else pair.input_seq_len
        )
        osl = (
            gen.sample_positive_normal_integer(
                float(pair.output_seq_len), pair.output_seq_len_stddev
            )
            if pair.output_seq_len_stddev > 0.0
            else pair.output_seq_len
        )
        return isl, osl

    def sample(self, gen) -> tuple[int, int]:
        return self._sample_pair_at(self._index_for_random(gen.random()), gen)

    def sample_batch(self, gen, batch_size: int) -> list[tuple[int, int]]:
        if batch_size == 0:
            raise RngError.invalid_parameter("batch_size", 0.0)
        indices = [self._index_for_random(r) for r in gen.random_batch(batch_size)]
        return [self._sample_pair_at(idx, gen) for idx in indices]


def _probability_sum_is_close(total: float) -> bool:
    return abs(total - 100.0) <= (
        _PROBABILITY_SUM_ABS_TOLERANCE + _PROBABILITY_SUM_REL_TOLERANCE * 100.0
    )
