# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared sub-types for adaptive search sweeps.

The full BO-specific config now lives on `AdaptiveSearchSweep` in
`aiperf.config.sweep`. This module retains the leaf types (`SLAFilter`,
`SearchSpaceDimension`) that other modules import directly.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from aiperf.config.base import BaseConfig
from aiperf.config.loader.dotted_path import _validate_dotted_path

__all__ = ["SLAFilter", "SLOTier", "SearchSpaceDimension"]


class SLAFilter(BaseConfig):
    """SLA constraint applied to BO scoring or grid filtering.

    A trial is considered feasible iff ``stat(metric_tag) op threshold`` holds.
    ``BayesianSearchPlanner`` consumes these for soft-penalty BO scoring and
    ``write_search_history`` consumes them for lexicographic
    feasibility-first best-result selection.

    Lives in ``aiperf.config`` rather than ``aiperf.search_recipes`` because
    ``AdaptiveSearchSweep`` carries a ``list[SLAFilter]`` field; the inverse
    location would force a circular import. Re-exported from
    ``aiperf.search_recipes._base`` for ergonomic recipe authoring.

    Example: enforce p95 TTFT under 200 ms via
    ``SLAFilter(metric_tag="time_to_first_token", stat="p95", op="lt", threshold=200.0)``.
    """

    model_config = ConfigDict(extra="forbid")

    metric_tag: str = Field(
        description=(
            "Metric tag to filter on, e.g. 'time_to_first_token'. Must match a key in "
            "RunResult.summary_metrics produced by the run."
        ),
    )
    stat: Literal[
        "avg", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "min", "max"
    ] = Field(
        default="p95",
        description="Statistic on the metric to compare against the threshold.",
    )
    op: Literal["lt", "le", "gt", "ge"] = Field(
        description="Comparison operator. Filter passes when stat(metric) op threshold is true.",
    )
    threshold: float = Field(
        description="Numeric threshold the metric statistic is compared against.",
    )

    @field_validator("metric_tag")
    @classmethod
    def _validate_metric_tag(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "SLAFilter.metric_tag must be a non-empty, non-whitespace "
                "string matching a key in RunResult.summary_metrics."
            )
        return v

    @field_validator("threshold")
    @classmethod
    def _validate_threshold_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(
                f"SLAFilter.threshold must be finite (NaN/inf comparisons "
                f"would silently fail every trial), got {v!r}."
            )
        return v


class SearchSpaceDimension(BaseConfig):
    """One dimension of the BO search space.

    `path` is a dotted path of the form `phases.profiling.concurrency` —
    the same grammar accepted by `aiperf.config.sweep._set_nested_value`.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description="Dotted-path into BenchmarkConfig (e.g. 'phases.profiling.concurrency')."
    )
    lo: float = Field(description="Inclusive lower bound.")
    hi: float = Field(description="Inclusive upper bound.")
    kind: Literal["int", "real"] = Field(
        default="real",
        description="Dimension type. 'int' rounds planner suggestions to integers; 'real' keeps floats.",
    )
    prior: Literal["uniform", "log-uniform"] = Field(
        default="uniform",
        description=(
            "Sampling prior for BO/Optuna planners. 'uniform' draws Sobol/random "
            "samples linearly across [lo, hi]; 'log-uniform' draws them log-spaced. "
            "Use 'log-uniform' when a parameter spans orders of magnitude (e.g. "
            "concurrency in [1, 1000]) so the initial-points phase covers low "
            "and high decades evenly. Requires lo > 0. Ignored by planners that "
            "do not draw initial points (monotonic, smooth_isotonic) where the "
            "search procedure is structurally adaptive rather than sampling-based."
        ),
    )

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        return _validate_dotted_path(v)

    @field_validator("lo", "hi")
    @classmethod
    def _validate_finite_bounds(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"lo/hi must be finite, got {v!r}.")
        return v

    @model_validator(mode="after")
    def _check_bounds(self) -> SearchSpaceDimension:
        if self.hi <= self.lo:
            raise ValueError(
                f"search-space dim {self.path!r}: hi ({self.hi}) must be > lo ({self.lo})."
            )
        # log-uniform requires lo > 0: log(0) is -inf and log(<0) is undefined,
        # so Optuna's suggest_*(log=True) raises at trial time. Validate here
        # so the error surfaces at config parse, not deep inside an ask() call.
        if self.prior == "log-uniform" and self.lo <= 0:
            raise ValueError(
                f"search-space dim {self.path!r}: prior='log-uniform' requires "
                f"lo > 0, got lo={self.lo}."
            )
        return self


class SLOTier(BaseConfig):
    """Named group of SLA filters representing one service-level objective.

    Defines a single tier for multi-tier SLO boundary search. Each tier
    contains one or more SLA filters that must ALL pass for the tier to be
    considered feasible at a given concurrency level.

    Lives in the config layer (alongside SLAFilter) to avoid circular imports
    between config and orchestrator packages. Re-exported from
    ``aiperf.orchestrator.search_planner.multi_tier_models`` for ergonomic
    orchestrator usage.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="Unique tier identifier for output artifacts")
    filters: list[SLAFilter] = Field(
        description="SLA filters that must ALL pass for this tier to be feasible",
        min_length=1,
    )
    ordering_rank: int | None = Field(
        default=None,
        ge=0,
        description="Rank in detected monotonic ordering (0=strictest); None if unordered",
    )
