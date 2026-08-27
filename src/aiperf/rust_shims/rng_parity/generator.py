# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ParityRandomGenerator`` — byte-exact port of ``rng/generator.rs``.

One :class:`~aiperf.rust_shims.rng_parity.pcg64.Pcg64` drives every draw; the wrapper
reproduces ``rand`` 0.9.4's uniform/int/float conversions and ``rand_distr`` 0.5.1's
continuous distributions bit-for-bit (see :mod:`aiperf.rust_shims.rng_parity.dist`).

Ported from ``rust/aiperf/src/rng/generator.rs``:
- float ``[0,1)`` — ``rand-0.9.4/src/distr/float.rs`` StandardUniform for ``f64``.
- integer ranges — ``rand-0.9.4/src/distr/uniform_int.rs`` ``sample_single`` /
  ``sample_single_inclusive`` (**Canon's biased method**; the ``unbiased`` feature is
  not enabled for ``aiperf``).

The public method surface matches ``aiperf.common.random_generator.RandomGenerator`` so
this is a drop-in backend; the *values* mirror Rust rather than legacy Python/NumPy.
"""

from __future__ import annotations

import math
import os

from aiperf.rust_shims.rng_parity.dist import (
    positive_integer_from_f64,
    sample_exp1,
    sample_gamma,
    sample_standard_normal,
)
from aiperf.rust_shims.rng_parity.errors import RngError
from aiperf.rust_shims.rng_parity.pcg64 import Pcg64

__all__ = ["ParityRandomGenerator"]

_M64 = (1 << 64) - 1
_M32 = (1 << 32) - 1
_U64_CARDINALITY = 1 << 64
# 2**63 as f64 — first i64-unrepresentable positive integer (generator.rs).
_I64_UPPER_EXCLUSIVE_AS_F64 = 9_223_372_036_854_775_808.0
_NORMAL_REJECTION_LIMIT = 10_000
_INV_2_53 = 1.0 / (1 << 53)


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _round_ties_even(value: float) -> float:
    """``f64::round_ties_even`` — round half to even, returning a float."""
    return float(round(value))


class ParityRandomGenerator:
    """One deterministic ``Pcg64`` plus AIPerf's sampling convenience methods."""

    __slots__ = ("_seed", "_rng")

    def __init__(self, seed: int | None, rng: Pcg64) -> None:
        self._seed = seed
        self._rng = rng

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_seed(cls, seed: int | None) -> ParityRandomGenerator:
        """Construct from a deterministic ``u64`` seed or OS entropy (``from_seed``)."""
        if seed is None:
            return cls(None, Pcg64.from_seed(os.urandom(32)))
        seed &= _M64
        return cls(seed, Pcg64.from_u64_seed(seed))

    @property
    def seed(self) -> int | None:
        """Return the deterministic seed if this generator has one."""
        return self._seed

    def reseed(self, seed: int) -> None:
        """Replace the generator state with ``seed`` (``reseed``)."""
        seed &= _M64
        self._seed = seed
        self._rng = Pcg64.from_u64_seed(seed)

    def clone(self) -> ParityRandomGenerator:
        """Return an independent copy with identical state."""
        return ParityRandomGenerator(self._seed, self._rng.clone())

    # ------------------------------------------------------------------ raw draws
    def random_u64(self) -> int:
        """Generate one uniformly distributed ``u64``."""
        return self._rng.next_u64()

    def fill_bytes(self, length: int) -> bytes:
        """Return ``length`` random bytes."""
        return self._rng.fill_bytes(length)

    def random(self) -> float:
        """Uniform float in ``[0, 1)`` (StandardUniform f64)."""
        return (self._rng.next_u64() >> 11) * _INV_2_53

    # ------------------------------------------------------------------ int ranges
    def _sample_single_inclusive_u64(self, low: int, high: int) -> int:
        """Canon's biased single-value sampler over ``[low, high]`` (u64 domain)."""
        low &= _M64
        high &= _M64
        rng = (high - low + 1) & _M64
        if rng == 0:  # range is 2**64 (unrepresentable): full-domain draw
            return self._rng.next_u64()
        product = self._rng.next_u64() * rng
        result = product >> 64
        lo_order = product & _M64
        if lo_order > ((-rng) & _M64):
            new_hi_order = (self._rng.next_u64() * rng) >> 64
            if lo_order + new_hi_order > _M64:
                result += 1
        return (low + result) & _M64

    def _sample_single_u64(self, low: int, high: int) -> int:
        """Half-open Canon sampler over ``[low, high)`` (``sample_single``)."""
        return self._sample_single_inclusive_u64(low, (high - 1) & _M64)

    def _sample_single_inclusive_u32(self, low: int, high: int) -> int:
        """Canon's biased single-value sampler over ``[low, high]`` (u32 domain).

        ``rand``'s ``u32`` sampling draws ``random::<u32>()`` = ``next_u64() as u32`` (the
        low 32 bits of one ``next_u64`` draw), then multiplies in the ``u32`` domain.
        """
        low &= _M32
        high &= _M32
        rng = (high - low + 1) & _M32
        if rng == 0:  # range is 2**32 (unrepresentable): full-domain u32 draw
            return self._rng.next_u64() & _M32
        product = (self._rng.next_u64() & _M32) * rng
        result = product >> 32
        lo_order = product & _M32
        if lo_order > ((-rng) & _M32):
            new_hi_order = ((self._rng.next_u64() & _M32) * rng) >> 32
            if lo_order + new_hi_order > _M32:
                result += 1
        return (low + result) & _M32

    def _sample_single_usize(self, low: int, high: int) -> int:
        """Half-open ``usize`` sampler (``UniformUsize::sample_single``, 64-bit target).

        ``usize`` ranges use 32-bit sampling when ``high <= u32::MAX`` for portability,
        else fall back to the ``u64`` Canon path. Drives ``choice`` / ``sample``.
        """
        if high > _M32:
            return self._sample_single_u64(low, high)
        return self._sample_single_inclusive_u32(low, high - 1)

    def _sample_single_inclusive_usize(self, low: int, high: int) -> int:
        """Inclusive ``usize`` sampler (``UniformUsize::sample_single_inclusive``).

        Drives ``shuffle`` (``random_range(0..=i)``).
        """
        if high > _M32:
            return self._sample_single_inclusive_u64(low, high)
        return self._sample_single_inclusive_u32(low, high)

    def _uniform_index(self, n: int) -> int:
        """Uniform index in ``[0, n)`` for ``1 <= n <= 2**64`` (``uniform_index``)."""
        if n == _U64_CARDINALITY:
            return self._rng.next_u64()
        return self._sample_single_u64(0, n)

    def randrange(self, *args: int) -> int:
        """Uniform integer from ``range(...)`` semantics (``randrange``).

        Accepts Python's ``random.randrange`` calling convention as a drop-in:
        ``randrange(stop)``, ``randrange(start, stop)``, or ``randrange(start, stop, step)``.
        The underlying draw matches the Rust ``generator.rs`` ``randrange(start, stop, step)``.
        """
        if len(args) == 1:
            start, stop, step = 0, args[0], 1
        elif len(args) == 2:
            start, stop, step = args[0], args[1], 1
        elif len(args) == 3:
            start, stop, step = args
        else:
            raise TypeError(f"randrange expected 1 to 3 arguments, got {len(args)}")
        if step == 0:
            raise RngError.empty_range("randrange step=0")
        width = stop - start
        if step > 0:
            n = 0 if width <= 0 else ((width - 1) // step) + 1
        elif width >= 0:
            n = 0
        else:
            # Match Rust's ``((width + 1) / step) + 1`` with truncating division.
            n = _trunc_div(width + 1, step) + 1
        if n <= 0:
            raise RngError.empty_range("randrange")
        idx = self._uniform_index(n)
        return start + idx * step

    def randbelow(self, stop: int) -> int:
        """Uniform integer from ``[0, stop)`` (``randbelow``)."""
        return self.randrange(0, stop, 1)

    def randrange_u64(self, lo: int, hi: int) -> int:
        """Uniform integer from ``[lo, hi)`` in the ``u64`` domain (``randrange_u64``)."""
        if lo >= hi:
            raise RngError.empty_range("randrange_u64")
        return self._sample_single_u64(lo, hi)

    def randint(self, a: int, b: int) -> int:
        """Uniform integer ``N`` with ``a <= N <= b`` (``randint``)."""
        if a > b:
            raise RngError.empty_range("randint")
        width = b - a + 1
        return a + self._uniform_index(width)

    def uniform(self, a: float, b: float) -> float:
        """Uniform float in ``[a, b)`` or ``[b, a)`` when ``b < a`` (``uniform``)."""
        return a + (b - a) * self.random()

    # ------------------------------------------------------------------ selection
    def choice(self, seq):
        """Select one element uniformly from a non-empty sequence (``choice``)."""
        if len(seq) == 0:
            raise RngError.empty_sequence("choice")
        idx = self._sample_single_usize(0, len(seq))
        return seq[idx]

    def choices(self, population, k: int) -> list:
        """Select ``k`` elements uniformly with replacement (``choices``)."""
        if len(population) == 0 and k > 0:
            raise RngError.empty_sequence("choices")
        return [self.choice(population) for _ in range(k)]

    def sample(self, population, k: int) -> list:
        """Select ``k`` unique elements uniformly without replacement (``sample``)."""
        if k > len(population):
            raise RngError.sample_too_large(k, len(population))
        values = list(population)
        self.shuffle(values)
        return values[:k]

    def shuffle(self, values: list) -> None:
        """Fisher-Yates shuffle in place (``shuffle``)."""
        for i in range(len(values) - 1, 0, -1):
            j = self._sample_single_inclusive_usize(0, i)
            values[i], values[j] = values[j], values[i]

    def weighted_choice(self, values, weights):
        """Select one element uniformly or by cumulative weights (``weighted_choice``)."""
        if weights is None:
            return self.choice(values)
        idx = self._weighted_index(len(values), weights)
        return values[idx]

    def numpy_choice(
        self, values, size=None, weights=None, replace: bool = True, p=None
    ):
        """NumPy-style ``choice`` over a sequence (``numpy_choice``).

        Accepts the legacy ``RandomGenerator.numpy_choice`` shape as a drop-in: ``values``
        may be an ``int`` (treated as ``range(values)``), ``p`` is an alias for
        ``weights``, and ``size=None`` returns a single element instead of a list.
        """
        if isinstance(values, int):
            values = list(range(values))
        if p is not None:
            weights = p
        if size is None:
            return self.numpy_choice(values, 1, weights, replace)[0]
        if len(values) == 0 and size > 0:
            raise RngError.empty_sequence("numpy_choice")
        if not replace and size > len(values):
            raise RngError.sample_too_large(size, len(values))
        if weights is not None:
            _validated_weight_total(len(values), weights)
            if not replace and sum(1 for w in weights if w > 0.0) < size:
                raise RngError.invalid_weights(
                    "fewer positive weights than requested samples"
                )

        if replace:
            if weights is None:
                return [self.weighted_choice(values, None) for _ in range(size)]
            cumulative = _cumulative_weights(weights)
            return [
                values[self._weighted_index_cached(cumulative)] for _ in range(size)
            ]
        if weights is None:
            return self.sample(values, size)

        pool = list(values)
        pool_weights = list(weights)
        out = []
        for _ in range(size):
            idx = self._weighted_index(len(pool), pool_weights)
            out.append(pool.pop(idx))
            pool_weights.pop(idx)
        return out

    def _weighted_index(self, value_len: int, weights) -> int:
        total = _validated_weight_total(value_len, weights)
        r = self.random() * total
        return _cumulative_weight_index(weights, r)

    def _weighted_index_cached(self, cumulative) -> int:
        total = cumulative[-1]
        r = self.random() * total
        # First index whose running total exceeds ``r`` (partition_point on ``<= r``).
        idx = _partition_point(cumulative, r)
        return min(idx, len(cumulative) - 1)

    # ------------------------------------------------------------------ continuous
    def expovariate(self, lambd: float) -> float:
        """Exponential distribution with rate ``lambd`` (``expovariate``)."""
        if lambd <= 0.0 or not _is_finite(lambd):
            raise RngError.invalid_parameter("lambda", lambd)
        return sample_exp1(self) * (1.0 / lambd)

    def gammavariate(self, alpha: float, beta: float) -> float:
        """Gamma distribution with shape ``alpha`` and scale ``beta`` (``gammavariate``)."""
        if alpha <= 0.0 or not _is_finite(alpha):
            raise RngError.invalid_parameter("alpha", alpha)
        if beta <= 0.0 or not _is_finite(beta):
            raise RngError.invalid_parameter("beta", beta)
        return sample_gamma(self, alpha, beta)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size=None):
        """Normal distribution with mean ``loc`` and stddev ``scale`` (``normal``).

        Legacy-compatible: ``size=None`` returns a scalar; a given ``size`` returns a list
        via :meth:`normal_batch` (each element Rust-parity).
        """
        if size is not None:
            return self.normal_batch(loc, scale, size)
        if scale < 0.0 or not _is_finite(scale):
            raise RngError.invalid_parameter("scale", scale)
        if not _is_finite(loc):
            raise RngError.invalid_parameter("loc", loc)
        if scale == 0.0:
            return loc
        return loc + scale * sample_standard_normal(self)

    def sample_normal(
        self, mean: float, stddev: float, lower: float, upper: float
    ) -> float:
        """Bounded normal via rejection with clamp fallback (``sample_normal``)."""
        if lower != lower:
            raise RngError.invalid_parameter("lower", lower)
        if upper != upper:
            raise RngError.invalid_parameter("upper", upper)
        if lower > upper:
            raise RngError.invalid_bounds(lower, upper)
        if stddev < 0.0 or not _is_finite(stddev):
            raise RngError.invalid_parameter("stddev", stddev)
        if not _is_finite(mean):
            raise RngError.invalid_parameter("mean", mean)
        for _ in range(_NORMAL_REJECTION_LIMIT):
            n = self.normal(mean, stddev)
            if lower <= n <= upper:
                return n
        return max(lower, min(upper, mean))

    def sample_positive_normal(self, mean: float, stddev: float) -> float:
        """Normal truncated at zero (``sample_positive_normal``)."""
        if mean < 0.0:
            raise RngError.invalid_parameter("mean", mean)
        return self.sample_normal(mean, stddev, 0.0, float("inf"))

    def sample_positive_normal_integer(self, mean: float, stddev: float) -> int:
        """Positive integer from a positive normal (``sample_positive_normal_integer``)."""
        if not _is_finite(mean):
            raise RngError.invalid_parameter("mean", mean)
        if not _is_finite(stddev):
            raise RngError.invalid_parameter("stddev", stddev)
        if stddev <= 0.0:
            return positive_integer_from_f64(_round_ties_even(mean), "rounded mean")
        return positive_integer_from_f64(
            float(math.ceil(self.sample_positive_normal(mean, stddev))),
            "normal integer sample",
        )

    # ------------------------------------------------------------------ batch
    def integers(
        self, low: int, high: int | None = None, size: int | None = None, dtype=None
    ):
        """Generate integers with NumPy's ``[low, high)`` convention (``integers``).

        Legacy-compatible: ``high=None`` means ``[0, low)``, ``size=None`` returns a single
        integer, and ``dtype`` is accepted and ignored (values are Python ``int``).
        """
        if high is not None:
            lo, hi = low, high
        else:
            lo, hi = 0, low
        if lo >= hi:
            raise RngError.empty_range("integers")
        if size is None:
            return self.randrange(lo, hi, 1)
        return [self.randrange(lo, hi, 1) for _ in range(size)]

    def normal_batch(self, loc: float, scale: float, size: int) -> list[float]:
        """Generate ``size`` normal samples (``normal_batch``)."""
        if scale < 0.0 or not _is_finite(scale):
            raise RngError.invalid_parameter("scale", scale)
        if not _is_finite(loc):
            raise RngError.invalid_parameter("loc", loc)
        if scale == 0.0:
            return [loc] * size
        return [loc + scale * sample_standard_normal(self) for _ in range(size)]

    def random_batch(self, size: int) -> list[float]:
        """Generate ``size`` uniform floats in ``[0, 1)`` (``random_batch``)."""
        return [self.random() for _ in range(size)]


def _trunc_div(a: int, b: int) -> int:
    """Integer division truncating toward zero (Rust ``/`` on integers)."""
    q = a // b
    if (a % b != 0) and ((a < 0) != (b < 0)):
        q += 1
    return q


def _cumulative_weight_index(weights, r: float) -> int:
    cumulative = 0.0
    for idx, weight in enumerate(weights):
        cumulative += weight
        if r < cumulative:
            return idx
    return len(weights) - 1


def _cumulative_weights(weights) -> list[float]:
    running = 0.0
    out = []
    for weight in weights:
        running += weight
        out.append(running)
    return out


def _partition_point(sorted_values, r: float) -> int:
    """Count of leading elements ``<= r`` (slice ``partition_point`` with ``c <= r``)."""
    lo, hi = 0, len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] <= r:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _validated_weight_total(value_len: int, weights) -> float:
    if len(weights) != value_len:
        raise RngError.invalid_weights("weights length must match values length")
    if len(weights) == 0:
        raise RngError.invalid_weights("weights cannot be empty")
    if any((not _is_finite(w)) or w < 0.0 for w in weights):
        raise RngError.invalid_weights("weights must be finite and non-negative")
    total = sum(weights)
    if not _is_finite(total):
        raise RngError.invalid_weights("weights must have a finite sum")
    if total <= 0.0:
        raise RngError.invalid_weights("weights must sum to a positive value")
    return total
