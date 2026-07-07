# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf Configuration - Sampling Distribution Types

6 distribution types, auto-detected from field structure (no ``type:`` key needed):

    isl: 512                                                    # FixedDistribution
    isl: {mean: 512, stddev: 50}                                # NormalDistribution
    isl: {mean: 512, median: 400}                               # LogNormalDistribution
    isl: {p50: 50000, p99: 400000, mean: 60000}                 # PercentileDistribution
    isl: {peaks: [{...}, {...}], split: 60}                     # MultimodalDistribution
    isl: {points: [{value: 128, weight: 40}, ...]}              # EmpiricalDistribution

Discriminator rules (checked in order):
    scalar int/float     -> FixedDistribution
    "peaks" in dict      -> MultimodalDistribution
    "points" in dict     -> EmpiricalDistribution
    "p50"/"p99" in dict  -> PercentileDistribution
    "median" in dict     -> LogNormalDistribution
    "stddev" in dict     -> NormalDistribution
    "value" in dict      -> FixedDistribution
    "mean" alone         -> NormalDistribution (stddev defaults to 0)
    anything else        -> ValueError
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Self

from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    PrivateAttr,
    Tag,
    model_validator,
)

from aiperf.config.base import BaseConfig

if TYPE_CHECKING:
    from aiperf.common.random_generator import RandomGenerator


# ==============================================================================
# Base class
# ==============================================================================


class Distribution(BaseConfig):
    """Base class for sampling distributions."""

    # x-kubernetes-preserve-unknown-fields lets the apiserver accept the int|float
    # scalar shorthand (FixedDistribution.coerce_scalar) and the no-`type`-key
    # discriminated union — neither expressible in a Kubernetes structural schema.
    # The marker is set at the base-class level so every concrete subclass
    # (Fixed/Normal/LogNormal/Multimodal/Empirical) inherits it.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
    )

    min: Annotated[
        float | None,
        Field(
            default=None,
            description=(
                "Inclusive lower bound; samples below are clamped up. Applies to "
                "every distribution type — composes with mean/stddev/median/peaks/"
                "points/value."
            ),
        ),
    ] = None

    max: Annotated[
        float | None,
        Field(
            default=None,
            description="Inclusive upper bound; samples above are clamped down.",
        ),
    ] = None

    @model_validator(mode="before")
    @classmethod
    def _strip_explicit_type(cls, data: Any) -> Any:
        """Drop the optional `type:` key after the discriminator has already used it.

        The discriminator at the union level dispatches by `type:` if present;
        once a concrete subclass is chosen, the `type:` key is redundant and
        would trigger extra_forbidden under each subclass's strict ConfigDict.
        """
        if isinstance(data, dict) and "type" in data:
            return {k: v for k, v in data.items() if k != "type"}
        return data

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        # Reject non-finite bounds explicitly: NaN/inf would silently disable
        # clamping (NaN comparisons are always false; inf can never be exceeded).
        for name, val in (("min", self.min), ("max", self.max)):
            if val is not None and not math.isfinite(val):
                raise ValueError(
                    f"Distribution bound `{name}` must be finite, got {val!r}"
                )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"Distribution bounds invalid: min={self.min} > max={self.max}; "
                f"swap them or remove one."
            )
        return self

    def __getattr__(self, name: str) -> Any:
        if name == "mean":
            return self.expected_value
        # PrivateAttr reads (e.g. PercentileDistribution._solution) must resolve
        # through Pydantic's machinery, which this override would otherwise shadow.
        if name in object.__getattribute__(self, "__private_attributes__"):
            return super().__getattr__(name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def sample(self, rng: RandomGenerator) -> float:
        """Draw one sample, clamping into [min, max] if bounds are set.

        Subclasses implement ``_sample_raw``; the base class applies bounds
        post-draw so every distribution type composes with ``min``/``max``
        without nesting.
        """
        v = self._sample_raw(rng)
        if self.min is not None and v < self.min:
            v = self.min
        if self.max is not None and v > self.max:
            v = self.max
        return v

    def _sample_raw(self, rng: RandomGenerator) -> float:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _sample_raw(rng) to generate one unclamped sample."
        )

    def sample_int(self, rng: RandomGenerator) -> int:
        return max(1, math.ceil(self.sample(rng)))

    @property
    def expected_value(self) -> float:
        # Note: returns the unclamped analytic mean. Approximate when
        # ``min``/``max`` bite — kept simple because callers use this for
        # config-time displays, not statistical inference.
        raise NotImplementedError(
            f"{type(self).__name__} must implement expected_value for config-time displays."
        )

    def __repr__(self) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement __repr__ with its distribution parameters."
        )


# ==============================================================================
# Distributions
# ==============================================================================


class FixedDistribution(Distribution):
    """Returns a constant value on every sample. Scalars coerce to this."""

    value: Annotated[
        float, Field(description="The constant value returned on every sample.")
    ]

    @model_validator(mode="before")
    @classmethod
    def coerce_scalar(cls, data: Any) -> Any:
        if isinstance(data, (int, float)):
            return {"value": float(data)}
        return data

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        if not math.isfinite(self.value):
            raise ValueError(
                f"Fixed distribution value must be finite, got {self.value}"
            )
        return self

    def _sample_raw(self, rng: RandomGenerator) -> float:
        return self.value

    @property
    def expected_value(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"fixed({self.value:g})"


class NormalDistribution(Distribution):
    """Gaussian (truncated at 0) parameterized by mean and stddev."""

    mean: Annotated[
        float,
        Field(
            ge=0.0,
            description=(
                "Mean value. Must be >= 0; samples below 0 are truncated, so "
                "a negative mean would yield a degenerate distribution. "
                "Zero is allowed (e.g. OSL=0 disables output, turn_delay mean=0 "
                "disables inter-turn delay)."
            ),
        ),
    ]

    stddev: Annotated[
        float,
        Field(
            ge=0.0, default=0.0, description="Standard deviation. 0 = deterministic."
        ),
    ]

    @model_validator(mode="after")
    def validate_finite(self) -> Self:
        if not math.isfinite(self.mean):
            raise ValueError(
                f"Normal distribution mean must be finite, got {self.mean}"
            )
        if not math.isfinite(self.stddev):
            raise ValueError(
                f"Normal distribution stddev must be finite, got {self.stddev}"
            )
        return self

    def _sample_raw(self, rng: RandomGenerator) -> float:
        if self.stddev <= 0:
            return self.mean
        return rng.sample_positive_normal(self.mean, self.stddev)

    @property
    def expected_value(self) -> float:
        return self.mean

    def __repr__(self) -> str:
        if self.stddev <= 0:
            return f"normal({self.mean:g})"
        return f"normal(mean={self.mean:g}, stddev={self.stddev:g})"


class LogNormalDistribution(Distribution):
    """Log-normal parameterized by mean and median (right-skewed, always positive).

    Skew is controlled by the mean/median ratio: larger ratio = heavier right tail.
    When mean == median the distribution is deterministic.

    Internally: sigma = sqrt(2 * log(mean / median)), mu = log(median).
    """

    mean: Annotated[
        float, Field(gt=0.0, description="Desired mean of the output distribution.")
    ]

    median: Annotated[
        float,
        Field(
            gt=0.0,
            description="Desired median. Must be <= mean. Lower median = more right skew.",
        ),
    ]

    @model_validator(mode="after")
    def validate_median_le_mean(self) -> Self:
        if self.median > self.mean:
            raise ValueError(
                f"Log-normal median ({self.median}) must be <= mean ({self.mean})."
            )
        return self

    @property
    def _sigma(self) -> float:
        if self.median >= self.mean:
            return 0.0
        return math.sqrt(2.0 * math.log(self.mean / self.median))

    def _sample_raw(self, rng: RandomGenerator) -> float:
        sigma = self._sigma
        if sigma <= 0:
            return self.mean
        return math.exp(rng.sample_normal(math.log(self.median), sigma))

    @property
    def expected_value(self) -> float:
        return self.mean

    def __repr__(self) -> str:
        if self.median >= self.mean:
            return f"lognormal({self.mean:g})"
        return f"lognormal(mean={self.mean:g}, median={self.median:g})"


_Z99 = 2.3263478740408408
"""Phi^-1(0.99): the z-score of the 99th percentile of a standard normal."""

_SQRT2 = math.sqrt(2.0)

_LOG_FLOAT_MAX = math.log(sys.float_info.max)
"""Largest argument for which ``math.exp`` returns a finite value (~709.78).

Used to reject PercentileDistribution targets whose p99/p50 ratio makes the
fitted lognormal's mean or variance factor overflow to a non-finite value.
"""


def _phi(x: float) -> float:
    """Standard normal CDF via math.erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _phi_inv(p: float) -> float:
    """Standard normal inverse CDF via bisection.

    Deterministic and dependency-free; 80 bisection rounds over [-10, 10]
    give far more precision than the solver's 1e-3 fit tolerances need.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"phi_inv requires 0 < p < 1, got {p}")
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass(frozen=True)
class _PercentileSolution:
    """Solved sampling parameters for a PercentileDistribution.

    kind == "lognormal": sample exp(N(mu, sigma)).
    kind == "mixture": with probability tail_weight sample
    positive-normal(tail_mean, tail_stddev), else
    positive-normal(body_mean, body_stddev).
    """

    kind: str
    mu: float = 0.0
    sigma: float = 0.0
    body_mean: float = 0.0
    body_stddev: float = 0.0
    tail_mean: float = 0.0
    tail_stddev: float = 0.0
    tail_weight: float = 0.0


def _try_mixture_weight(
    p50: float, p99: float, mean: float, body_stddev: float, w: float
) -> _PercentileSolution | None:
    """Attempt an exact mixture fit at tail weight ``w``.

    Fixed-point iteration on the two cross-CDF terms; each round solves
    body_mean from the p50 equation, tail_mean from the mean equation, and
    tail_stddev from the p99 equation. Returns None when this ``w`` cannot
    satisfy the targets (caller tries the next weight).
    """
    f_tail_at_p50 = 0.0
    f_body_at_p99 = 1.0
    body_mean = tail_mean = tail_stddev = 0.0
    for _ in range(25):
        body_cdf_target = (0.5 - w * f_tail_at_p50) / (1.0 - w)
        if not 0.0 < body_cdf_target < 1.0:
            return None
        body_mean = p50 - _phi_inv(body_cdf_target) * body_stddev
        tail_mean = (mean - (1.0 - w) * body_mean) / w
        tail_cdf_target = (0.99 - (1.0 - w) * f_body_at_p99) / w
        if not 0.0 < tail_cdf_target < 1.0:
            return None
        z = _phi_inv(tail_cdf_target)
        if abs(z) < 1e-9:
            return None
        tail_stddev = (p99 - tail_mean) / z
        if tail_stddev <= 0.0:
            return None
        f_tail_at_p50 = _phi((p50 - tail_mean) / tail_stddev)
        f_body_at_p99 = _phi((p99 - body_mean) / body_stddev)

    # Truncation at 0 must be negligible or the sampled stats drift off target.
    if body_mean <= 2.0 * body_stddev or tail_mean <= 2.0 * tail_stddev:
        return None
    achieved_p50 = (1.0 - w) * _phi((p50 - body_mean) / body_stddev) + w * _phi(
        (p50 - tail_mean) / tail_stddev
    )
    achieved_p99 = (1.0 - w) * _phi((p99 - body_mean) / body_stddev) + w * _phi(
        (p99 - tail_mean) / tail_stddev
    )
    achieved_mean = (1.0 - w) * body_mean + w * tail_mean
    if abs(achieved_p50 - 0.5) > 0.005:
        return None
    if abs(achieved_p99 - 0.99) > 0.002:
        return None
    if abs(achieved_mean - mean) > 0.005 * mean:
        return None
    return _PercentileSolution(
        kind="mixture",
        body_mean=body_mean,
        body_stddev=body_stddev,
        tail_mean=tail_mean,
        tail_stddev=tail_stddev,
        tail_weight=w,
    )


def _solve_percentile(
    p50: float, p99: float, mean: float | None
) -> _PercentileSolution:
    """Solve sampling parameters hitting the percentile targets exactly.

    Without ``mean``: log-normal (2 params, 2 targets, closed form).
    With ``mean``: log-normal if the mean already matches the implied one,
    otherwise a two-component mixture searched over tail weights and body
    spreads. Raises ValueError when no searched shape fits.
    """
    mu = math.log(p50)
    try:
        sigma = math.log(p99 / p50) / _Z99
        implied_mean = p50 * math.exp(sigma**2 / 2.0)
    except OverflowError as exc:
        raise ValueError(
            f"Percentile targets too extreme: p99/p50 ratio {p99 / p50:g} produces a "
            f"non-finite mean or variance. Reduce the spread between p50 ({p50:g}) "
            f"and p99 ({p99:g})."
        ) from exc
    # exp(sigma**2) is the lognormal's variance factor; when sigma**2 exceeds the
    # largest finite math.exp argument the distribution's second moment is
    # non-finite even though the mean may still be representable. Reject so
    # downstream `expected_value > 0` gates never see a non-finite value.
    if not math.isfinite(implied_mean) or sigma**2 > _LOG_FLOAT_MAX:
        raise ValueError(
            f"Percentile targets too extreme: p99/p50 ratio {p99 / p50:g} produces a "
            f"non-finite mean or variance. Reduce the spread between p50 ({p50:g}) "
            f"and p99 ({p99:g})."
        )
    if mean is None or abs(mean - implied_mean) <= 0.01 * implied_mean:
        return _PercentileSolution(kind="lognormal", mu=mu, sigma=sigma)

    for body_cv in (0.2, 0.1, 0.3):
        body_stddev = body_cv * p50
        for w in (0.03, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45):
            solution = _try_mixture_weight(p50, p99, mean, body_stddev, w)
            if solution is not None:
                return solution
    raise ValueError(
        f"Percentile targets infeasible: no mixture found for p50={p50:g}, "
        f"p99={p99:g}, mean={mean:g}. The mean must lie above p50 and not "
        f"approach p99 (a log-normal with these percentiles implies mean~"
        f"{implied_mean:g}; omit `mean` to use it). Adjust the targets or "
        f"express the shape manually with a multimodal distribution."
    )


class PercentileDistribution(Distribution):
    """Right-skewed distribution parameterized directly by percentile targets.

    YAML:
        isl: {p50: 50000, p99: 400000}                 # log-normal, exact p50/p99
        isl: {p50: 50000, p99: 400000, mean: 60000}    # mixture, also pins the mean

    With only p50 and p99, a log-normal fits both exactly. Adding ``mean``
    covers shapes no 2-parameter distribution can express (e.g. p50=50k,
    p99=400k, mean=60k): a body component carries the median while a small
    heavy tail carries the p99, solved deterministically at config
    validation time so infeasible targets fail fast in `aiperf config
    validate` rather than mid-benchmark.
    """

    p50: Annotated[
        float,
        Field(gt=0.0, description="Target median (50th percentile) of the samples."),
    ]

    p99: Annotated[
        float,
        Field(
            gt=0.0,
            description="Target 99th percentile. Must be greater than p50.",
        ),
    ]

    mean: Annotated[
        float | None,
        Field(
            default=None,
            gt=0.0,
            description=(
                "Optional target mean. Must be greater than p50. When omitted, "
                "the mean implied by the p50/p99 log-normal fit applies."
            ),
        ),
    ] = None

    _solution: _PercentileSolution | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_and_solve(self) -> Self:
        for name, val in (("p50", self.p50), ("p99", self.p99), ("mean", self.mean)):
            if val is not None and not math.isfinite(val):
                raise ValueError(
                    f"Percentile target `{name}` must be finite, got {val!r}"
                )
        if self.p99 <= self.p50:
            raise ValueError(f"p99 ({self.p99}) must be greater than p50 ({self.p50}).")
        if self.mean is not None and self.mean <= self.p50:
            raise ValueError(
                f"mean ({self.mean}) must be greater than p50 ({self.p50}); "
                f"percentile distributions model right-skewed shapes. For "
                f"left-heavy shapes use a multimodal or empirical distribution."
            )
        self._solution = _solve_percentile(self.p50, self.p99, self.mean)
        return self

    def _solved(self) -> _PercentileSolution:
        if self._solution is None:
            self._solution = _solve_percentile(self.p50, self.p99, self.mean)
        return self._solution

    def _sample_raw(self, rng: RandomGenerator) -> float:
        s = self._solved()
        if s.kind == "lognormal":
            if s.sigma <= 0:
                return self.p50
            return math.exp(rng.sample_normal(s.mu, s.sigma))
        if rng.random() < s.tail_weight:
            return rng.sample_positive_normal(s.tail_mean, s.tail_stddev)
        return rng.sample_positive_normal(s.body_mean, s.body_stddev)

    @property
    def expected_value(self) -> float:
        if self.mean is not None:
            return self.mean
        sigma = math.log(self.p99 / self.p50) / _Z99
        return self.p50 * math.exp(sigma**2 / 2.0)

    def __repr__(self) -> str:
        if self.mean is not None:
            return f"percentile(p50={self.p50:g}, p99={self.p99:g}, mean={self.mean:g})"
        return f"percentile(p50={self.p50:g}, p99={self.p99:g})"


class PeakEntry(BaseConfig):
    """A weighted component in a multimodal distribution.

    The weight and distribution fields are written inline in YAML:
        {mean: 128, stddev: 20, weight: 60}

    The ``weight`` key is extracted before the remaining fields are parsed
    as a SamplingDistribution. Defaults to 1.0 (equal split when omitted).
    """

    model_config = ConfigDict(extra="forbid")

    distribution: Annotated[
        SamplingDistribution,
        Field(description="The sub-distribution for this peak."),
    ]
    weight: Annotated[
        float,
        Field(
            ge=0.0, default=1.0, description="Relative weight (normalised internally)."
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def inline_weight(cls, data: Any) -> Any:
        # Note: this validator is an internal canonicalization between
        # `{distribution: ..., weight: N}` (canonical) and inline form
        # `{mean: ..., stddev: ..., weight: N}` (user-facing). The polymorphism
        # lives entirely on the inner `distribution: SamplingDistribution` field
        # — Distribution's class-level x-kubernetes-preserve-unknown-fields
        # already covers that subtree, so no marker is needed here.
        if isinstance(data, dict):
            data = dict(data)
            weight = data.pop("weight", 1.0)
            if "distribution" in data:
                # Already in canonical form {distribution: {...}, weight: N}
                return {"distribution": data["distribution"], "weight": weight}
            # Inline form: remaining keys are the distribution fields
            return {"distribution": data, "weight": weight}
        return data


class MultimodalDistribution(Distribution):
    """Weighted mixture of N peaks (N >= 2).

    YAML:
        isl:
          peaks:
            - {mean: 128, stddev: 20, weight: 60}
            - {mean: 2048, median: 1800, weight: 40}
        # Equal split — omit weight:
        isl:
          peaks:
            - {mean: 128, stddev: 20}
            - {mean: 2048, median: 1800}
            - {mean: 8192, median: 4096}
    """

    peaks: Annotated[
        list[PeakEntry],
        Field(min_length=2, description="Two or more weighted sub-distributions."),
    ]

    @model_validator(mode="after")
    def validate_peaks(self) -> Self:
        if len(self.peaks) < 2:
            raise ValueError("peaks requires at least 2 entries")
        return self

    def _sample_raw(self, rng: RandomGenerator) -> float:
        total = sum(p.weight for p in self.peaks)
        r = rng.random() * total
        cumulative = 0.0
        for peak in self.peaks:
            cumulative += peak.weight
            if r < cumulative:
                return peak.distribution.sample(rng)
        return self.peaks[-1].distribution.sample(rng)

    @property
    def expected_value(self) -> float:
        total = sum(p.weight for p in self.peaks)
        return sum(p.weight / total * p.distribution.expected_value for p in self.peaks)

    def __repr__(self) -> str:
        total = sum(p.weight for p in self.peaks)
        parts = [
            f"{repr(p.distribution)} @ {p.weight / total * 100:.0f}%"
            for p in self.peaks
        ]
        return f"multimodal({', '.join(parts)})"


class EmpiricalPoint(BaseConfig):
    """A weighted value in an empirical distribution."""

    model_config = ConfigDict(extra="forbid")

    value: Annotated[float, Field(description="The discrete value.")]
    weight: Annotated[
        float,
        Field(
            gt=0.0, default=1.0, description="Relative weight (normalized internally)."
        ),
    ]


class EmpiricalDistribution(Distribution):
    """Discrete distribution sampled from weighted values.

    YAML:
        isl:
          points:
            - {value: 128, weight: 40}
            - {value: 512, weight: 35}
            - {value: 2048, weight: 20}
            - {value: 8192, weight: 5}
    """

    points: Annotated[
        list[EmpiricalPoint],
        Field(description="Weighted discrete values. Weights are relative."),
    ]

    @model_validator(mode="after")
    def validate_points(self) -> Self:
        if not self.points:
            raise ValueError("Empirical distribution requires at least 1 point")
        return self

    def _sample_raw(self, rng: RandomGenerator) -> float:
        total = sum(p.weight for p in self.points)
        r = rng.random() * total
        cumulative = 0.0
        for point in self.points:
            cumulative += point.weight
            if r < cumulative:
                return point.value
        return self.points[-1].value

    @property
    def expected_value(self) -> float:
        total = sum(p.weight for p in self.points)
        return sum(p.weight / total * p.value for p in self.points)

    def __repr__(self) -> str:
        total = sum(p.weight for p in self.points)
        parts = [f"{p.value:g} @ {p.weight / total * 100:.0f}%" for p in self.points]
        return f"empirical({', '.join(parts)})"


# ==============================================================================
# Discriminated union
# ==============================================================================

_TAG_MAP = {
    "FixedDistribution": "fixed",
    "NormalDistribution": "normal",
    "LogNormalDistribution": "lognormal",
    "PercentileDistribution": "percentile",
    "MultimodalDistribution": "multimodal",
    "EmpiricalDistribution": "empirical",
}

_CANONICAL_TYPES = (
    "fixed",
    "normal",
    "lognormal",
    "percentile",
    "multimodal",
    "empirical",
)


def _distribution_discriminator(v: Any) -> str:
    """Detect distribution type from `type:` key OR field structure.

    Order:
        scalar              -> "fixed"
        explicit "type:"    -> use it (must be one of _CANONICAL_TYPES)
        "peaks" in dict     -> "multimodal"
        "points" in dict    -> "empirical"
        "p50"/"p99" in dict -> "percentile"
        "median" in dict    -> "lognormal"
        "stddev" in dict    -> "normal"
        "value" in dict     -> "fixed"
        "mean" in dict      -> "normal"
        already-built       -> pass through via _TAG_MAP
        unknown             -> ValueError
    """
    if isinstance(v, (int, float)):
        return "fixed"
    if isinstance(v, dict):
        explicit = v.get("type")
        if isinstance(explicit, str):
            if explicit in _CANONICAL_TYPES:
                return explicit
            raise ValueError(
                f"Unknown distribution type {explicit!r}. "
                f"Expected one of {_CANONICAL_TYPES} or omit `type:` and rely on "
                f"structural inference (e.g. {{mean: 512, stddev: 100}})."
            )
        if "peaks" in v:
            return "multimodal"
        if "points" in v:
            return "empirical"
        if "p50" in v or "p99" in v:
            return "percentile"
        if "median" in v:
            return "lognormal"
        if "stddev" in v:
            return "normal"
        if "value" in v:
            return "fixed"
        if "mean" in v:
            return "normal"
        raise ValueError(
            "Cannot determine distribution type from keys. "
            "Expected: scalar, {mean+stddev}, {mean+median}, {p50, p99[, mean]}, "
            "{peaks:[distA, distB]}, or {points:[{value, weight}, ...]}."
        )
    tag = _TAG_MAP.get(type(v).__name__)
    if tag:
        return tag
    raise ValueError(f"Cannot parse {type(v).__name__!r} as a distribution.")


SamplingDistribution = Annotated[
    Annotated[FixedDistribution, Tag("fixed")]
    | Annotated[NormalDistribution, Tag("normal")]
    | Annotated[LogNormalDistribution, Tag("lognormal")]
    | Annotated[PercentileDistribution, Tag("percentile")]
    | Annotated[MultimodalDistribution, Tag("multimodal")]
    | Annotated[EmpiricalDistribution, Tag("empirical")],
    Discriminator(
        _distribution_discriminator,
        custom_error_type="invalid_distribution_type",
        custom_error_message=(
            "Invalid distribution. Expected: scalar, {mean+stddev}, {mean+median}, "
            "{p50, p99[, mean]}, {peaks:[{...weight:N}, ...]}, or "
            "{points:[{value, weight}, ...]}."
        ),
    ),
]
"""Discriminated union for all sampling distributions.

Accepts (no 'type' key required):
    512                                              -> FixedDistribution
    {mean: 512, stddev: 50}                          -> NormalDistribution
    {mean: 512, median: 400}                         -> LogNormalDistribution
    {p50: 50000, p99: 400000, mean: 60000}           -> PercentileDistribution
    {peaks: [{mean:128, stddev:20, weight:60},
             {mean:2048, median:1800, weight:40}]}   -> MultimodalDistribution
    {points: [{value: 128, weight: 40}, ...]}        -> EmpiricalDistribution
"""

# PeakEntry holds SamplingDistribution — resolve the forward reference.
# No other model references SamplingDistribution, so no other rebuild is needed.
PeakEntry.model_rebuild()
