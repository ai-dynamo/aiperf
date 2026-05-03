# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema for the BO / adaptive outer-loop configuration.

Lives in the config layer (not the orchestrator) because MultiRunConfig
holds an adaptive_search field — placing this in aiperf.orchestrator would
force a reverse import from aiperf.config.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from aiperf.common.enums import OptimizationDirection
from aiperf.config._base import BaseConfig
from aiperf.plugin.enums import SearchPlannerType

__all__ = ["AdaptiveSearchConfig", "SLAFilter", "SearchSpaceDimension"]


class SLAFilter(BaseConfig):
    """SLA constraint applied to BO scoring or grid filtering.

    A trial is considered feasible iff ``stat(metric_tag) op threshold`` holds.
    Phase 2 wires this into ``BayesianSearchPlanner`` for soft-penalty BO
    scoring and into ``write_search_history`` for lexicographic
    feasibility-first best-result selection.

    Lives in ``aiperf.config`` rather than ``aiperf.search_recipes`` because
    ``AdaptiveSearchConfig`` carries a ``list[SLAFilter]`` field; the inverse
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
    stat: Literal["avg", "p50", "p90", "p95", "p99"] = Field(
        default="p95",
        description="Statistic on the metric to compare against the threshold.",
    )
    op: Literal["lt", "le", "gt", "ge"] = Field(
        description="Comparison operator. Filter passes when stat(metric) op threshold is true.",
    )
    threshold: float = Field(
        description="Numeric threshold the metric statistic is compared against.",
    )


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
        description="Dimension type. 'int' rounds skopt suggestions to integers; 'real' keeps floats.",
    )

    @model_validator(mode="after")
    def _check_bounds(self) -> SearchSpaceDimension:
        if self.hi <= self.lo:
            raise ValueError(
                f"search-space dim {self.path!r}: hi ({self.hi}) must be > lo ({self.lo})."
            )
        return self


class AdaptiveSearchConfig(BaseConfig):
    """Configuration for an adaptive outer loop (e.g. Bayesian Optimization).

    Attached to MultiRunConfig.adaptive_search when --search-* flags are set; absent
    otherwise. Propagates to BenchmarkPlan.adaptive_search in build_benchmark_plan
    and is consumed by MultiRunOrchestrator.execute_adaptive_search.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["bayes"] = Field(
        default="bayes",
        description="Search algorithm. v1 only supports Bayesian Optimization (`bayes`).",
    )
    planner: SearchPlannerType = Field(
        default=SearchPlannerType.BAYESIAN,
        description=(
            "Outer-loop search planner plugin name. Defaults to `bayesian` "
            "(scikit-optimize-backed). Third-party planners registered under "
            "the `search_planner` plugin category are valid here. Selected "
            "via `--search-planner` on the CLI."
        ),
    )
    search_space: list[SearchSpaceDimension] = Field(
        description="Dimensions to optimize over. Must be non-empty.",
        min_length=1,
    )
    objective_metric: str = Field(
        description="Metric tag to optimize, e.g. 'output_token_throughput'. "
        "Must match a key in RunResult.summary_metrics produced by the run.",
    )
    objective_stat: Literal["avg", "p50", "p90", "p95", "p99"] = Field(
        default="avg",
        description="Statistic on the metric (matches JsonMetricResult fields).",
    )
    objective_direction: OptimizationDirection = Field(
        description="Whether higher (MAXIMIZE) or lower (MINIMIZE) is better.",
    )
    max_iterations: int = Field(
        ge=2,
        le=200,
        description="Maximum number of BO iterations. Each iteration runs `plan.trials` benchmarks.",
    )
    n_initial_points: int = Field(
        default=5,
        ge=1,
        description="Sobol-random points before skopt fits the GP. Must be < max_iterations.",
    )
    plateau_window: int = Field(
        default=5,
        ge=2,
        description="Number of recent iterations to inspect for plateau detection.",
    )
    plateau_threshold: float = Field(
        default=0.01,
        gt=0,
        description="Coefficient-of-variation threshold for plateau (relative; scale-free).",
    )
    improvement_patience: int = Field(
        default=10,
        ge=2,
        description=(
            "Stop after this many consecutive iterations with no improvement "
            "over the running best objective. Standard idiom from skopt's "
            "HollowIterationsStopper and Hyperopt's no_progress_loss; "
            "complements plateau_threshold (either signal can stop the loop)."
        ),
    )
    random_seed: int | None = Field(
        default=None,
        description="If set, passed as `random_state` to skopt.Optimizer for reproducibility.",
    )
    sla_filters: list[SLAFilter] = Field(
        default_factory=list,
        description=(
            "SLA constraints applied to BO scoring (soft penalty in skopt's loss "
            "space) and to best-result selection (lexicographic feasibility-first "
            "filtering). Empty list means unconstrained — pure single-objective BO."
        ),
    )
    recipe_name: str | None = Field(
        default=None,
        description=(
            "Name of the search recipe that produced this config, when expanded "
            "via --search-recipe. Surfaced in search_history.json for post-run "
            "audit. None when AdaptiveSearchConfig was built from explicit --search-* flags."
        ),
    )

    @model_validator(mode="after")
    def _check_initial_points_below_max_iterations(self) -> AdaptiveSearchConfig:
        if self.n_initial_points >= self.max_iterations:
            raise ValueError(
                f"n_initial_points ({self.n_initial_points}) must be < max_iterations ({self.max_iterations}); "
                f"otherwise the GP never fits."
            )
        return self
