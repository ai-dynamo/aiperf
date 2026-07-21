# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampling-based sweep types (Sobol, Latin Hypercube).

Re-exported from `aiperf.config.sweep` so external imports stay stable.
"""

from __future__ import annotations

import math
import warnings
from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from aiperf.common.finite import is_finite_value
from aiperf.config.base import BaseConfig
from aiperf.config.loader.dotted_path import _validate_dotted_path
from aiperf.config.sweep.config import _GridSweepBase

__all__ = [
    "LatinHypercubeSweep",
    "SamplingDimension",
    "SobolSweep",
    "_SamplingSweepBase",
]


class SamplingDimension(BaseConfig):
    """One dimension of a QMC sampling space.

    Used by `SobolSweep` and `LatinHypercubeSweep`. Distinct from
    `SearchSpaceDimension` (BO) — same path/lo/hi grammar, but native
    `choices` support and explicit `scale` validation.

    Path is a dotted path of the form `phases.profiling.concurrency` —
    the same grammar accepted by `_set_nested_value`.
    """

    model_config = ConfigDict(extra="forbid")

    path: Annotated[str, Field(description="Dotted-path into BenchmarkConfig.")]
    lo: Annotated[
        float | None,
        Field(default=None, description="Inclusive lower bound (with hi)."),
    ]
    hi: Annotated[
        float | None,
        Field(default=None, description="Inclusive upper bound (with lo)."),
    ]
    scale: Annotated[
        Literal["linear", "log"],
        Field(default="linear", description="Mapping from unit cube to value."),
    ]
    kind: Annotated[
        Literal["int", "real"],
        Field(default="real", description="Output type after mapping."),
    ]
    choices: Annotated[
        list[Any] | None,
        Field(
            default=None,
            description="Discrete value list. Mutually exclusive with (lo, hi).",
        ),
    ]

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return _validate_dotted_path(v)

    @field_validator("lo", "hi")
    @classmethod
    def _validate_finite_bounds(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError(f"lo/hi must be finite, got {v!r}.")
        return v

    @model_validator(mode="after")
    def _check_range_or_choices(self) -> Self:
        has_range = self.lo is not None and self.hi is not None
        has_choices = bool(self.choices)
        if self.choices is not None and len(self.choices) < 1:
            raise ValueError(
                f"dim {self.path!r}: choices must have at least one entry."
            )
        if self.choices is not None:
            for entry in self.choices:
                try:
                    hash(entry)
                except TypeError as e:
                    raise ValueError(
                        f"dim {self.path!r}: choices entries must be "
                        f"hashable scalars; got {type(entry).__name__}."
                    ) from e
                # Why: NaN/inf in `choices` would slip past _validate_finite_bounds
                # (which only covers lo/hi) and propagate through _map_dim into
                # per-variant configs. The orchestrator's defensive guard would
                # then raise mid-flight rather than at config validation.
                # Numeric entries must be finite; non-numeric entries (str, dict,
                # ...) are intentionally untouched -- choices may legitimately be
                # categorical labels or nested dicts.
                if (
                    isinstance(entry, int | float)
                    and not isinstance(entry, bool)
                    and not is_finite_value(entry)
                ):
                    raise ValueError(
                        f"dim {self.path!r}: choices entries must be "
                        f"finite numeric values; got {entry!r}."
                    )
        if has_range == has_choices:
            raise ValueError(
                f"dim {self.path!r}: provide either (lo, hi) or choices, not both/neither."
            )
        if has_range and self.hi <= self.lo:
            raise ValueError(
                f"dim {self.path!r}: hi ({self.hi}) must be > lo ({self.lo})."
            )
        if has_range and self.scale == "log" and self.lo <= 0:
            raise ValueError(f"dim {self.path!r}: log-scale requires lo > 0.")
        return self

    @model_validator(mode="after")
    def _warn_on_narrow_int_range(self) -> Self:
        if self.kind != "int" or self.lo is None or self.hi is None:
            return self
        if self.scale == "log":
            # Distinct integer log-buckets: floor(log(hi)) - ceil(log(lo)).
            try:
                span = math.floor(math.log(self.hi)) - math.ceil(math.log(self.lo))
            except ValueError:
                return self
            too_narrow = span < 1
        else:
            too_narrow = (self.hi - self.lo) < 2
        if too_narrow:
            warnings.warn(
                f"dim {self.path!r}: integer {self.scale}-scale range "
                f"[{self.lo}, {self.hi}] is too narrow for sampling diversity; "
                f"many samples will collapse to duplicates.",
                stacklevel=2,
            )
        return self


class _SamplingSweepBase(_GridSweepBase):
    """Shared fields for QMC sampling sweeps (sobol, latin_hypercube).

    Inherits iteration_order, same_seed, cooldown_seconds, sla_filters,
    post_process from _GridSweepBase / _SweepBase.
    """

    samples: Annotated[
        int,
        Field(
            ge=2,
            le=2**20,
            description=(
                "Number of variations to draw. Capped at 2**20 (1,048,576) — "
                "scipy's Sobol engine fails at 2**30 anyway, and full "
                "materialization beyond ~1M variants risks orchestrator OOM."
            ),
        ),
    ]
    seed: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description=(
                "RNG seed for reproducibility. Must be non-negative; "
                "scipy's QMC engines reject negative seeds with an opaque "
                "low-level error."
            ),
        ),
    ]
    dimensions: Annotated[
        list[SamplingDimension],
        Field(min_length=1, description="Dimensions to sample over."),
    ]
    label_format: Annotated[
        Literal["index", "kv"],
        Field(
            default="index",
            description=(
                "Variation-label format. 'index' => '<type>_NNNN' "
                "(short, sortable). 'kv' => 'k1=v1, k2=v2' (grid-style)."
            ),
        ),
    ]

    @model_validator(mode="after")
    def _check_unique_dim_paths(self) -> Self:
        seen: set[str] = set()
        dups: list[str] = []
        for d in self.dimensions:
            if d.path in seen and d.path not in dups:
                dups.append(d.path)
            seen.add(d.path)
        if dups:
            raise ValueError(
                f"sampling sweep dimensions must have unique paths; "
                f"duplicates: {dups!r}."
            )
        return self


class SobolSweep(_SamplingSweepBase):
    """Sobol quasi-Monte-Carlo sweep — low-discrepancy joint coverage.

    Note on ``scramble=False``: scipy's raw Sobol sequence starts at the
    origin (the first row is exactly all-zeros), so the first variant
    will land on ``(lo, lo, ...)`` for every dimension. With small
    ``samples`` this produces degenerate corner-only coverage. Prefer
    the default ``scramble=True`` (Owen-scrambled) unless you have a
    specific reason to want raw Sobol points.
    """

    type: Annotated[
        Literal["sobol"],
        Field(default="sobol", description="Sweep type discriminator."),
    ]
    scramble: Annotated[
        bool,
        Field(
            default=True,
            description="Owen-scrambled Sobol (recommended). False = raw Sobol sequence.",
        ),
    ]


class LatinHypercubeSweep(_SamplingSweepBase):
    """Latin hypercube sampling — perfect marginal stratification per axis."""

    type: Annotated[
        Literal["latin_hypercube"],
        Field(default="latin_hypercube", description="Sweep type discriminator."),
    ]
    optimization: Annotated[
        Literal[None, "random-cd", "lloyd"],
        Field(
            default="random-cd",
            description=(
                "LHS post-processing to reduce centered discrepancy. "
                "None = vanilla LHS."
            ),
        ),
    ]
