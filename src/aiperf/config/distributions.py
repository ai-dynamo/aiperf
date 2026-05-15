# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf Configuration - Sampling Distribution Types

5 distribution types, auto-detected from field structure (no ``type:`` key needed):

    isl: 512                                                    # FixedDistribution
    isl: {mean: 512, stddev: 50}                                # NormalDistribution
    isl: {mean: 512, median: 400}                               # LogNormalDistribution
    isl: {peaks: [{...}, {...}], split: 60}                     # MultimodalDistribution
    isl: {points: [{value: 128, weight: 40}, ...]}              # EmpiricalDistribution

Discriminator rules (checked in order):
    scalar int/float   -> FixedDistribution
    "peaks" in dict    -> MultimodalDistribution
    "points" in dict   -> EmpiricalDistribution
    "median" in dict   -> LogNormalDistribution
    "stddev" in dict   -> NormalDistribution
    "value" in dict    -> FixedDistribution
    "mean" alone       -> NormalDistribution (stddev defaults to 0)
    anything else      -> ValueError
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import ConfigDict, Discriminator, Field, Tag, model_validator
from typing_extensions import Self

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
    "MultimodalDistribution": "multimodal",
    "EmpiricalDistribution": "empirical",
}

_CANONICAL_TYPES = ("fixed", "normal", "lognormal", "multimodal", "empirical")


def _distribution_discriminator(v: Any) -> str:
    """Detect distribution type from `type:` key OR field structure.

    Order:
        scalar              -> "fixed"
        explicit "type:"    -> use it (must be one of _CANONICAL_TYPES)
        "peaks" in dict     -> "multimodal"
        "points" in dict    -> "empirical"
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
            "Expected: scalar, {mean+stddev}, {mean+median}, "
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
    | Annotated[MultimodalDistribution, Tag("multimodal")]
    | Annotated[EmpiricalDistribution, Tag("empirical")],
    Discriminator(
        _distribution_discriminator,
        custom_error_type="invalid_distribution_type",
        custom_error_message=(
            "Invalid distribution. Expected: scalar, {mean+stddev}, {mean+median}, "
            "{peaks:[{...weight:N}, ...]}, or {points:[{value, weight}, ...]}."
        ),
    ),
]
"""Discriminated union for all sampling distributions.

Accepts (no 'type' key required):
    512                                              -> FixedDistribution
    {mean: 512, stddev: 50}                          -> NormalDistribution
    {mean: 512, median: 400}                         -> LogNormalDistribution
    {peaks: [{mean:128, stddev:20, weight:60},
             {mean:2048, median:1800, weight:40}]}   -> MultimodalDistribution
    {points: [{value: 128, weight: 40}, ...]}        -> EmpiricalDistribution
"""

# PeakEntry holds SamplingDistribution — resolve the forward reference.
# No other model references SamplingDistribution, so no other rebuild is needed.
PeakEntry.model_rebuild()
