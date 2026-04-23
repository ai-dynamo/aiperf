# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf Configuration - Sampling Distribution Types

5 distribution types, auto-detected from field structure (no ``type:`` key needed):

    isl: 512                                                    # FixedDistribution
    isl: {mean: 512, stddev: 50}                                # NormalDistribution
    isl: {mean: 512, median: 400}                               # LogNormalDistribution
    isl: {peaks: [{...}, {...}], split: 60}                     # BimodalDistribution
    isl: {points: [{value: 128, weight: 40}, ...]}              # EmpiricalDistribution

Discriminator rules (checked in order):
    scalar int/float   -> FixedDistribution
    "peaks" in dict    -> BimodalDistribution
    "points" in dict   -> EmpiricalDistribution
    "median" in dict   -> LogNormalDistribution
    "stddev" in dict   -> NormalDistribution
    "value" in dict    -> FixedDistribution
    "mean" alone       -> ValueError (ambiguous: add stddev or median)
    anything else      -> ValueError
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import ConfigDict, Discriminator, Field, Tag, model_validator
from typing_extensions import Self

from aiperf.config._base import BaseConfig

if TYPE_CHECKING:
    from aiperf.common.random_generator import RandomGenerator


# ==============================================================================
# Base class
# ==============================================================================


class Distribution(BaseConfig):
    """Base class for sampling distributions."""

    model_config = ConfigDict(extra="forbid")

    def __getattr__(self, name: str) -> Any:
        if name == "mean":
            return self.expected_value
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def sample(self, rng: RandomGenerator) -> float:
        raise NotImplementedError

    def sample_int(self, rng: RandomGenerator) -> int:
        return max(1, math.ceil(self.sample(rng)))

    @property
    def expected_value(self) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


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

    def sample(self, rng: RandomGenerator) -> float:
        return self.value

    @property
    def expected_value(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"fixed({self.value:g})"


class NormalDistribution(Distribution):
    """Gaussian (truncated at 0) parameterized by mean and stddev."""

    mean: Annotated[float, Field(description="Mean value.")]

    stddev: Annotated[
        float,
        Field(
            ge=0.0, default=0.0, description="Standard deviation. 0 = deterministic."
        ),
    ]

    def sample(self, rng: RandomGenerator) -> float:
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

    def sample(self, rng: RandomGenerator) -> float:
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

    def sample(self, rng: RandomGenerator) -> float:
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

    def sample(self, rng: RandomGenerator) -> float:
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


def _distribution_discriminator(v: Any) -> str:
    """Detect distribution type from field structure — no 'type' key needed.

    Order:
        scalar             -> "fixed"
        "peaks" in dict    -> "bimodal"
        "points" in dict   -> "empirical"
        "median" in dict   -> "lognormal"
        "stddev" in dict   -> "normal"
        "value" in dict    -> "fixed"
        already-built      -> pass through via _TAG_MAP
        "mean" alone       -> ValueError (ambiguous)
        unknown            -> ValueError
    """
    if isinstance(v, (int, float)):
        return "fixed"
    if isinstance(v, dict):
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
