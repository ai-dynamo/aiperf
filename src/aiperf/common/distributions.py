# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mean/median-parameterized lognormal and Weibull distributions.

Shared by the agentic-code dataset generator (inter-turn delay mixture model)
and the user-centric timing strategy (sampled per-user turn gaps). Both params
models accept real-space mean/median and solve the native parameters
(mu/sigma, shape/scale) so callers can pin the mean while controlling skew
via the median.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Annotated, Any, Literal

import numpy as np
from annotated_types import Ge, Gt
from numpy.random import Generator
from pydantic import AfterValidator, ConfigDict, Field, model_validator
from scipy.optimize import brentq

from aiperf.common.finite import FiniteFloat, is_finite_value
from aiperf.common.models import AIPerfBaseModel


def _reject_non_finite(value: float) -> float:
    """Reject NaN/inf, matching :data:`FiniteFloat`'s message."""
    if not is_finite_value(value):
        raise ValueError(f"value must be finite, got {value!r}")
    return value


# Bounded finite floats. The bound lives INSIDE the Annotated rather than in
# Field(gt=...) because the two do not compose on an optional field: pydantic
# emits `{"gt": 0.0}` for `FiniteFloat | None = Field(gt=0.0)` instead of
# `{"exclusiveMinimum": 0.0}`, and `gt` is not a JSON Schema keyword -- an
# editor validating against the published spec would silently stop enforcing
# the bound. That is the same schema-weaker-than-the-loader failure the
# untagged-Weibull guard below exists to prevent, so it is not cosmetic.
_PositiveFinite = Annotated[float, Gt(0.0), AfterValidator(_reject_non_finite)]
_NonNegativeFinite = Annotated[float, Ge(0.0), AfterValidator(_reject_non_finite)]

# Native parameters that identify one family to the exclusion of the other.
# AIPerfBaseModel allows extra keys and the union defaults to lognormal when
# untagged, so without this guard a config carrying shape/scale but no
# `distribution` tag parses as lognormal and silently samples the wrong family.
_LOGNORMAL_ONLY_FIELDS = ("mu", "sigma")
_WEIBULL_ONLY_FIELDS = ("shape", "scale")


def _reject_foreign_fields(
    extra: dict[str, Any] | None, foreign: tuple[str, ...], family: str, other: str
) -> None:
    stray = [name for name in foreign if name in (extra or {})]
    if stray:
        raise ValueError(
            f"{family} delay parameters carry {'/'.join(stray)}, which belong to "
            f'the {other} family. Set "distribution": "{other}" to use {other}, '
            "or drop the field(s)."
        )


class LognormalParams(AIPerfBaseModel):
    """Lognormal distribution parameters with real-space summary statistics.

    Can be constructed in two ways:
    1. Full: mu, sigma, mean, median all provided (e.g. from manifest.json or fit-stats)
    2. Simplified: just mean and median — mu/sigma auto-computed via model validator

    Carries no distribution tag: selecting a family is a delay-union concern, so
    the tag lives on the delay-only subclass (LognormalDelayParams) instead of
    leaking into every token-count and size config that reuses these params.
    """

    # Mirrors the _reject_foreign_fields validator so schema consumers (editors,
    # authoring-time validation) reject an untagged Weibull config too, instead
    # of silently matching this branch and dropping shape/scale. This guard
    # applies to every user of these params, tagged or not, because the
    # validator does: a schema weaker than the loader green-lights configs that
    # then fail to load.
    model_config = ConfigDict(
        json_schema_extra={
            "not": {"anyOf": [{"required": ["shape"]}, {"required": ["scale"]}]}
        }
    )

    # Every field is finite-checked: a gt/ge bound does not exclude inf, so a
    # plain float accepted mean=inf and derived sigma=inf from it. Log-space mean
    # is legitimately negative for sub-1 medians, so mu takes no bound at all and
    # can use FiniteFloat directly.
    mu: FiniteFloat | None = Field(default=None, description="Log-space mean")
    sigma: _NonNegativeFinite | None = Field(
        default=None, description="Log-space standard deviation"
    )
    mean: _PositiveFinite = Field(description="Real-space mean (derived)")
    median: _PositiveFinite = Field(description="Real-space median (derived)")
    min: _PositiveFinite | None = Field(
        default=None, description="Hard lower bound (rejection sampled)"
    )
    max: _PositiveFinite | None = Field(
        default=None, description="Hard upper bound (rejection sampled)"
    )

    @model_validator(mode="after")
    def compute_mu_sigma(self) -> LognormalParams:
        _reject_foreign_fields(
            self.model_extra, _WEIBULL_ONLY_FIELDS, "lognormal", "weibull"
        )
        if self.mean < self.median:
            raise ValueError(
                f"mean ({self.mean}) must be >= median ({self.median}) for lognormal"
            )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        if (self.mu is None) != (self.sigma is None):
            raise ValueError("mu and sigma must be supplied as a pair")
        if self.mu is None:
            self.mu = math.log(self.median)
            ratio = self.mean / self.median
            self.sigma = math.sqrt(2.0 * math.log(ratio)) if ratio > 1.0 else 0.0
        return self


# brentq bracket for the Weibull shape solve: f(0.05) > 0 for any sane
# mean/median ratio, and f(3.5) < 0 for all ratios > 1 because the Weibull
# mean/median ratio drops below 1 near shape ~3.44.
_WEIBULL_SHAPE_BRACKET = (0.05, 3.5)


def _weibull_shape_scale_from_mean_median(
    mean: float, median: float
) -> tuple[float, float]:
    """Solve Weibull shape/scale from real-space mean and median.

    median = scale * ln(2)^(1/shape), mean = scale * Gamma(1 + 1/shape), so the
    ratio mean/median = Gamma(1 + 1/shape) / ln(2)^(1/shape) pins the shape.
    """
    ratio = mean / median
    if ratio <= 1.0:
        raise ValueError(
            f"mean ({mean}) must be > median ({median}) to derive Weibull "
            "shape/scale; the mean/median parameterization supports "
            "right-skewed delays only. Supply shape and scale explicitly for "
            "other parameterizations."
        )

    def f(shape: float) -> float:
        return math.gamma(1.0 + 1.0 / shape) - ratio * math.log(2.0) ** (1.0 / shape)

    shape = float(brentq(f, *_WEIBULL_SHAPE_BRACKET))
    scale = median / math.log(2.0) ** (1.0 / shape)
    return shape, scale


def _weibull_mean_median(shape: float, scale: float) -> tuple[float, float]:
    """Real-space mean and median implied by a Weibull shape/scale pair."""
    return (
        scale * math.gamma(1.0 + 1.0 / shape),
        scale * math.log(2.0) ** (1.0 / shape),
    )


# Relative tolerance for cross-checking explicit shape/scale against the
# declared mean/median. The derived path reproduces both to ~1e-14, so this is
# loose enough to round-trip a generated config and tight enough to reject a
# hand-written pair that does not describe the same distribution.
_WEIBULL_SUMMARY_RTOL = 1e-3


def _weibull_summary_mismatch(name: str, declared: float, implied: float) -> str:
    return (
        f"{name} ({declared}) disagrees with the {name} implied by the supplied "
        f"shape/scale ({implied:.6g}); mean and median are derived summary "
        f"statistics, not independent inputs. Supply a {name} consistent with "
        "shape/scale, or omit shape/scale to solve them from mean/median."
    )


class WeibullParams(AIPerfBaseModel):
    """Weibull distribution parameters with real-space summary statistics.

    Can be constructed in two ways:
    1. Full: shape, scale, mean, median all provided. mean/median are derived
       summary statistics, so they are cross-checked against the distribution
       implied by shape/scale and a mismatch is rejected.
    2. Simplified: just mean and median (mean > median) — shape/scale solved
       numerically via a model validator
    """

    model_config = ConfigDict(
        json_schema_extra={
            "not": {"anyOf": [{"required": ["mu"]}, {"required": ["sigma"]}]}
        }
    )

    distribution: Literal["weibull"] = Field(
        description="Distribution family tag; required so untagged configs keep parsing as lognormal"
    )
    # FiniteFloat throughout, for the same reason as LognormalParams: gt=0.0
    # admits inf, and an infinite mean or bound propagates into the solve and
    # into rejection sampling rather than being caught at the edge.
    shape: _PositiveFinite | None = Field(
        default=None, description="Weibull shape parameter k"
    )
    scale: _PositiveFinite | None = Field(
        default=None, description="Weibull scale parameter lambda"
    )
    mean: _PositiveFinite = Field(description="Real-space mean (derived)")
    median: _PositiveFinite = Field(description="Real-space median (derived)")
    min: _PositiveFinite | None = Field(
        default=None, description="Hard lower bound (rejection sampled)"
    )
    max: _PositiveFinite | None = Field(
        default=None, description="Hard upper bound (rejection sampled)"
    )

    @model_validator(mode="after")
    def compute_shape_scale(self) -> WeibullParams:
        _reject_foreign_fields(
            self.model_extra, _LOGNORMAL_ONLY_FIELDS, "weibull", "lognormal"
        )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        if (self.shape is None) != (self.scale is None):
            raise ValueError("shape and scale must be supplied as a pair")
        if self.shape is None:
            self.shape, self.scale = _weibull_shape_scale_from_mean_median(
                self.mean, self.median
            )
            return self
        implied_mean, implied_median = _weibull_mean_median(self.shape, self.scale)
        if not math.isclose(self.mean, implied_mean, rel_tol=_WEIBULL_SUMMARY_RTOL):
            raise ValueError(_weibull_summary_mismatch("mean", self.mean, implied_mean))
        if not math.isclose(self.median, implied_median, rel_tol=_WEIBULL_SUMMARY_RTOL):
            raise ValueError(
                _weibull_summary_mismatch("median", self.median, implied_median)
            )
        return self


def _rejection_sample(
    draw: Callable[[int], np.ndarray],
    *,
    lo: float | None,
    hi: float | None,
    clip_min: float | None,
    max_attempts: int,
    size: int,
) -> np.ndarray:
    """Draw *size* samples from *draw*, resampling any that fall outside [lo, hi].

    Values still out of range after max_attempts are clamped, so the bounds hold
    even when the distribution puts little mass inside them. clip_min is a hard
    floor applied last.
    """
    samples = draw(size)
    if lo is not None or hi is not None:
        for _ in range(max_attempts):
            mask = np.zeros(len(samples), dtype=bool)
            if lo is not None:
                mask |= samples < lo
            if hi is not None:
                mask |= samples > hi
            if not mask.any():
                break
            samples[mask] = draw(int(mask.sum()))
        if lo is not None:
            samples = np.maximum(samples, lo)
        if hi is not None:
            samples = np.minimum(samples, hi)
    if clip_min is not None:
        samples = np.maximum(samples, clip_min)
    return samples


def sample_lognormal(
    params: LognormalParams,
    rng: Generator,
    *,
    size: int = 1,
    clip_min: float | None = None,
    max_attempts: int = 100,
) -> np.ndarray:
    """Draw samples from a lognormal distribution.

    Uses rejection sampling for params.min and params.max (resample out-of-range
    values). clip_min is a hard floor applied after rejection sampling.
    """
    return _rejection_sample(
        lambda n: rng.lognormal(mean=params.mu, sigma=params.sigma, size=n),
        lo=params.min,
        hi=params.max,
        clip_min=clip_min,
        max_attempts=max_attempts,
        size=size,
    )


def sample_weibull(
    params: WeibullParams,
    rng: Generator,
    *,
    size: int = 1,
    clip_min: float | None = None,
    max_attempts: int = 100,
) -> np.ndarray:
    """Draw samples from a Weibull distribution.

    Uses rejection sampling for params.min and params.max (resample out-of-range
    values). clip_min is a hard floor applied after rejection sampling.
    """
    return _rejection_sample(
        lambda n: rng.weibull(params.shape, size=n) * params.scale,
        lo=params.min,
        hi=params.max,
        clip_min=clip_min,
        max_attempts=max_attempts,
        size=size,
    )
