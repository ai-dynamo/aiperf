# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep configuration models for parameter exploration.

Supports the following sweep strategies:
- Grid: Cartesian product of all parameter values
- Zip: Element-wise (lockstep) pairing of parameter values
- Scenarios: Hand-picked configurations deep-merged with base
"""

from __future__ import annotations

import math
import warnings
from typing import Annotated, Any, ClassVar, Literal

from pydantic import ConfigDict, Discriminator, Field, model_validator
from typing_extensions import Self

from aiperf.common.enums import OptimizationDirection, SweepMode
from aiperf.common.finite import FiniteFloat
from aiperf.config.base import BaseConfig
from aiperf.config.loader.dotted_path import _validate_dotted_path
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter, SLOTier
from aiperf.plugin.enums import SearchPlannerType
from aiperf.search_recipes._post_process import PostProcessSpec

__all__ = [
    "MAGIC_LIST_FIELDS",
    "AdaptiveObjective",
    "AdaptiveSearchSweep",
    "GridSweep",
    "LatinHypercubeSweep",
    "Objective",
    "OutcomeConstraint",
    "SamplingDimension",
    "ScenarioSweep",
    "SobolSweep",
    "SweepConfig",
    "SweepVariation",
    "ZipSweep",
    "_set_nested_value",
    "expand_sweep",
]


class _SweepBase(BaseConfig):
    """Shared fields across every sweep variant.

    Inherits ``model_config = extra="forbid"`` from BaseConfig. Subclasses
    that add fields don't accidentally accept stray keys.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    cooldown_seconds: Annotated[
        float,
        Field(
            ge=0,
            default=0.0,
            description="Cooldown between sweep variations (or BO iterations).",
        ),
    ]
    sla_filters: Annotated[
        list[SLAFilter],
        Field(
            default_factory=list,
            description="SLA constraints applied at trial scoring / aggregation time.",
        ),
    ]
    post_process: Annotated[
        PostProcessSpec | None,
        Field(
            default=None,
            description="Optional post-aggregation handler emitted by a search recipe.",
        ),
    ]
    recipe_name: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Source recipe name (audit trail when expanded via "
                "--search-recipe). Used by the live Pareto plot path to look "
                "up the recipe's pareto_axes after expansion has stripped the "
                "originating --search-recipe flag."
            ),
        ),
    ]


class _GridSweepBase(_SweepBase):
    """Shared fields for non-adaptive sweeps (grid, zip, scenarios, QMC sampling).

    ``iteration_order``/``same_seed`` are inapplicable to adaptive_search,
    which inherits ``_SweepBase`` directly.
    """

    iteration_order: Annotated[
        SweepMode,
        Field(
            default=SweepMode.REPEATED,
            description="Trial/variation iteration order. 'repeated': trials outer, "
            "variations inner. 'independent': variations outer, trials inner. "
            "Inapplicable to adaptive_search.",
        ),
    ]
    same_seed: Annotated[
        bool,
        Field(
            default=False,
            description="If True, every variation reuses the envelope's random_seed. "
            "If False, each variation derives `random_seed + variation.index`.",
        ),
    ]


class GridSweep(_GridSweepBase):
    """Grid sweep - all combinations of parameters (Cartesian product)."""

    type: Literal["grid"] = Field(
        default="grid", description="Sweep type discriminator."
    )
    parameters: dict[str, list[Any]] = Field(
        ...,
        description="Parameters to sweep: dot-notation path -> list of values.",
        min_length=1,
    )

    @model_validator(mode="after")
    def _validate_parameters(self) -> Self:
        _validate_grid_parameters(self.parameters, sweep_kind="grid")
        return self


class ZipSweep(_GridSweepBase):
    """Zip sweep — parameters paired element-wise (lockstep).

    Like ``GridSweep`` but uses ``zip(strict=True)`` instead of Cartesian
    product. All parameter lists must have identical length. Use when you
    want N runs each setting a coordinated tuple of fields, without the
    NxM blow-up of grid. Canonical use case: paired ISL/OSL.
    """

    type: Literal["zip"] = Field(default="zip", description="Sweep type discriminator.")
    parameters: dict[str, list[Any]] = Field(
        ...,
        description="Parameters to sweep in lockstep: dot-notation path -> list of values. All lists must have equal length.",
        min_length=1,
    )

    @model_validator(mode="after")
    def _validate_parameters(self) -> Self:
        _validate_grid_parameters(self.parameters, sweep_kind="zip")
        return self

    @model_validator(mode="after")
    def _check_equal_lengths(self) -> Self:
        lengths = {k: len(v) for k, v in self.parameters.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"zip sweep parameters must all have equal length; got {lengths!r}."
            )
        return self


def _validate_grid_parameters(
    parameters: dict[str, list[Any]], *, sweep_kind: str
) -> None:
    """Validate every grid/zip sweep parameter at config-load time.

    Enforces three contracts at validation time so a malformed sweep block
    surfaces from ``aiperf config validate`` rather than waiting for the
    runtime ``expand_sweep`` walker:

    * Dotted path is structurally valid (no empty/leading-dot/double-dot,
      no envelope prefix, no non-sweepable first segment).
    * Value list is non-empty (an empty list silently produces zero
      variations, hiding the typo).
    * Every numeric value is finite — NaN/Inf would survive into
      ``FiniteFloat``-bounded sweep targets and poison downstream metrics
      per the project's NaN/Inf discipline (see CLAUDE.md).
    """
    for path, values in parameters.items():
        try:
            _validate_dotted_path(path)
        except ValueError as e:
            raise ValueError(f"{sweep_kind} sweep parameter: {e}") from e
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                f"{sweep_kind} sweep parameter {path!r}: value list must be non-empty."
            )
        for v in values:
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(
                    f"{sweep_kind} sweep parameter {path!r}: value {v!r} is not "
                    f"finite. NaN and +/-inf are rejected by AIPerf's "
                    f"FiniteFloat contract (see CLAUDE.md)."
                )


class ScenarioSweep(_GridSweepBase):
    """Scenario sweep - hand-picked configurations deep-merged with base."""

    type: Literal["scenarios"] = Field(
        default="scenarios", description="Sweep type discriminator."
    )
    runs: list[dict[str, Any]] = Field(
        ...,
        description="List of scenario dicts to deep-merge with base config.",
        min_length=1,
    )

    @model_validator(mode="after")
    def _check_unique_run_names(self) -> Self:
        # Cell directories are derived from `variation.label`, which falls
        # back to the run's `name`. Duplicate names => colliding cell dirs;
        # the second run's artifacts clobber the first. Reject here so the
        # user sees the conflict instead of silent data loss.
        seen: set[str] = set()
        dups: list[str] = []
        for run in self.runs:
            name = run.get("name") if isinstance(run, dict) else None
            if not isinstance(name, str):
                continue
            if name in seen and name not in dups:
                dups.append(name)
            seen.add(name)
        if dups:
            raise ValueError(
                f"scenario sweep runs must have unique names; duplicates: {dups!r}."
            )
        return self


class Objective(BaseConfig):
    """Single optimization objective.

    Multiple ``Objective`` entries on an ``AdaptiveSearchSweep`` enable
    Pareto Bayesian optimization (e.g. qlognehvi / qnehvi / qehvi). A
    length-1 list is single-objective BO.
    """

    model_config = ConfigDict(extra="forbid")

    metric: Annotated[
        str,
        Field(description="Metric tag to optimize (e.g. 'output_token_throughput')."),
    ]
    stat: Annotated[
        Literal["avg", "p50", "p90", "p95", "p99"],
        Field(default="avg", description="Statistic on the metric."),
    ]
    direction: Annotated[
        OptimizationDirection,
        Field(description="MAXIMIZE or MINIMIZE."),
    ]
    threshold: Annotated[
        FiniteFloat | None,
        Field(
            default=None,
            description=(
                "Pareto reference point for hypervolume computation. Trials worse "
                "than this on this objective do not contribute to hypervolume. "
                "When None, auto-derived from the worst observed value among "
                "Sobol initial points. Ignored for single-objective runs."
            ),
        ),
    ] = None


# Backward-compat alias: callers (search_recipes, search_planner, schema,
# tests) import ``AdaptiveObjective``; ``Objective`` is the canonical name.
AdaptiveObjective = Objective


class OutcomeConstraint(BaseConfig):
    """Feasibility gate on a metric the optimizer is *not* optimizing.

    Distinct from ``Objective.threshold`` (Pareto reference point) and from
    ``SLAFilter`` (post-hoc filter on benchmark eligibility). An outcome
    constraint masks BoTorch acquisition: candidates predicted to violate
    it are downweighted.
    """

    model_config = ConfigDict(extra="forbid")

    metric: Annotated[str, Field(description="Metric tag to constrain.")]
    op: Annotated[
        Literal["<=", ">=", "=="],
        Field(description="Comparison operator."),
    ]
    bound: Annotated[FiniteFloat, Field(description="Threshold value.")]


class AdaptiveSearchSweep(_SweepBase):
    """Adaptive outer-loop search (Bayesian / monotonic / etc.).

    A third sweep variant: BO drives variation choice itself, so the
    grid-style ``iteration_order`` and ``same_seed`` knobs do not apply.
    Inherits cooldown/sla_filters/post_process from `_SweepBase` only.
    """

    type: Annotated[
        Literal["adaptive_search"],
        Field(default="adaptive_search", description="Sweep type discriminator."),
    ]

    planner: Annotated[
        SearchPlannerType,
        Field(
            default=SearchPlannerType.BAYESIAN,
            description="Outer-loop planner plugin name (bayesian | optuna | monotonic_sla | smooth_isotonic).",
        ),
    ]
    search_space: Annotated[
        list[SearchSpaceDimension],
        Field(
            min_length=1,
            description="Dimensions to optimize over.",
        ),
    ]
    objectives: Annotated[
        list[Objective],
        Field(
            min_length=1,
            description=(
                "What to optimize. Length-1 = single-objective search; length-N = "
                "multi-objective search. BoTorch-backed Pareto BO requires "
                "--optuna-sampler botorch with a multi-objective acquisition like "
                "qlognehvi."
            ),
        ),
    ]
    outcome_constraints: Annotated[
        list[OutcomeConstraint],
        Field(
            default_factory=list,
            description="Feasibility gates on non-optimized metrics. Empty = no constraints.",
        ),
    ]
    max_iterations: Annotated[
        int,
        Field(
            ge=2,
            le=200,
            description="Maximum BO iterations.",
        ),
    ]
    n_initial_points: Annotated[
        int,
        Field(
            ge=1,
            default=5,
            description="Sobol-random points before the GP fits. Must be < max_iterations for `bayesian`/`optuna` planners; ignored by 1-D SLA planners.",
        ),
    ]
    plateau_window: Annotated[
        int,
        Field(
            ge=2,
            default=8,
            description=(
                "Recent-iteration window for plateau detection. Default 8 "
                "keeps the window strictly larger than ``n_initial_points`` "
                "(default 5), so the plateau test always inspects at least a "
                "few GP-driven iterations rather than tripping on the "
                "Sobol-only prefix when the initial points happen to land in "
                "a flat region."
            ),
        ),
    ]
    plateau_threshold: Annotated[
        float,
        Field(
            gt=0,
            default=0.01,
            description="Coefficient-of-variation threshold for plateau stop.",
        ),
    ]
    improvement_patience: Annotated[
        int,
        Field(
            ge=2,
            default=10,
            description="Stop after this many no-improvement iterations.",
        ),
    ]
    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description="If set, passed as `random_state` to the planner backend.",
        ),
    ]
    optuna_sampler: Annotated[
        Literal["gp", "tpe", "botorch"],
        Field(
            default="botorch",
            description=(
                "Optuna sampler when planner=optuna. Default ``botorch`` is the "
                "preferred Gaussian-process qLogNEI/qLogNEHVI path and requires "
                "the optional ``botorch`` extra; when this default is implicit and "
                "the optional stack is unavailable, the planner warns and falls "
                "back to ``tpe``. ``tpe`` is the dep-light Tree-Parzen estimator "
                "that ships with Optuna core. ``gp`` is Optuna's pure-GP sampler "
                "and requires torch."
            ),
        ),
    ]
    optuna_acquisition: Annotated[
        Literal["logei", "qlogei", "qnei", "qlognei", "qehvi", "qnehvi", "qlognehvi"]
        | None,
        Field(
            default=None,
            description=(
                "Acquisition override for the BoTorch sampler. None lets Optuna "
                "pick (single-objective LogEI). ``qnei`` = Letham 2017 noisy-EI; "
                "``qlognei`` = Ament 2023 (BoTorch's strongly recommended modern "
                "default; requires botorch>=0.10). ``qehvi``/``qnehvi``/``qlognehvi`` "
                "are multi-objective hypervolume acquisitions for Pareto BO. Only "
                "meaningful when planner=optuna and optuna_sampler=botorch; ignored "
                "otherwise."
            ),
        ),
    ]
    optuna_terminator: Annotated[
        Literal["regret", "emmr", "none"],
        Field(
            default="none",
            description=(
                "Optional posterior-regret stopping rule (Makarova 2022 / "
                "Ishibashi 2023; same family as Wilson 2024). Layered on top "
                "of three-signal convergence. Only meaningful when "
                "planner=optuna; ignored otherwise."
            ),
        ),
    ]
    objective_pooling: Annotated[
        Literal["mean", "pooled"],
        Field(
            default="mean",
            description=(
                "Per-trial percentile aggregation. ``mean`` = arithmetic mean "
                "of per-trial percentiles across trials (current default). "
                "``pooled`` = walks each trial's profile_export.jsonl and "
                "computes np.percentile over the pooled raw-sample bag (the "
                "statistic that satisfies SLO claims; requires --export-level "
                "records). Only meaningful when objective.stat is a "
                "percentile (p50/p90/p95/p99); a no-op when stat=avg."
            ),
        ),
    ]
    monotonic_stability_trials: Annotated[
        int,
        Field(
            ge=1,
            default=2,
            description="MonotonicSLA planner stability-window size.",
        ),
    ]

    # smooth_isotonic SLA planner knobs
    sla_replicates: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description=(
                "Replicate-step trial count override for the smooth_isotonic SLA "
                "planner. 0 = auto (Hyperband-flavored formula based on "
                "sigma_margin/threshold). >0 = fixed override. Ignored by other "
                "planners."
            ),
        ),
    ]
    sla_precision: Annotated[
        Literal["tight", "normal", "coarse"],
        Field(
            default="normal",
            description=(
                "Per-probe sample-budget knob for the smooth_isotonic SLA "
                "planner. tight=10000, normal=1000, coarse=300 requests. "
                "Drives p99 CI width. Ignored by other planners."
            ),
        ),
    ]
    sla_warmup_seconds: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            description=(
                "Discard this many seconds at the start of each probe before "
                "computing margins, to skip CUDA-graph warmup and KV-cache "
                "shape transients. None = auto (max(30s, 3 * inter-batch-time)). "
                "Ignored by other planners."
            ),
        ),
    ]

    constraint_mode: Annotated[
        Literal["penalty", "eic"],
        Field(
            default="penalty",
            description="Constrained-BO acquisition mode (penalty | eic).",
        ),
    ]

    sla_tiers: Annotated[
        list[SLOTier],
        Field(
            default_factory=list,
            description=(
                "Multi-tier SLO definitions parsed from --search-sla-tier flags. "
                "Each tier is a named group of SLA filters sharing a common "
                "boundary search. When non-empty (2-10 tiers), the MultiTierPlanner "
                "is activated instead of single-tier planners."
            ),
        ),
    ]

    @model_validator(mode="after")
    def _check_initial_points_below_max_iterations(self) -> Self:
        # n_initial_points is a BO-specific knob (Sobol/random points before
        # the GP fits). Non-BO planners drive their own probe sequence and
        # ignore this field; enforcing the gate against them rejects valid
        # configs (e.g. monotonic_sla with max_iterations=3 + the schema
        # default n_initial_points=5).
        if str(self.planner) not in {
            str(SearchPlannerType.BAYESIAN),
            str(SearchPlannerType.OPTUNA),
        }:
            return self
        if self.n_initial_points >= self.max_iterations:
            raise ValueError(
                f"n_initial_points ({self.n_initial_points}) must be < "
                f"max_iterations ({self.max_iterations}); otherwise the GP never fits."
            )
        return self

    @model_validator(mode="after")
    def _check_unique_search_space_paths(self) -> Self:
        # Two dimensions writing to the same dotted path is silently broken:
        # the planner explores both axes independently against the same
        # backing field, the second write per trial clobbers the first, and
        # the optimization wastes iterations on a phantom degree of freedom.
        # Mirror the QMC sampling-sweep dedup so the user sees the conflict
        # at validation time.
        seen: set[str] = set()
        dups: list[str] = []
        for d in self.search_space:
            if d.path in seen and d.path not in dups:
                dups.append(d.path)
            seen.add(d.path)
        if dups:
            raise ValueError(
                f"adaptive_search search_space dimensions must have unique "
                f"paths; duplicates: {dups!r}."
            )
        return self

    @model_validator(mode="after")
    def _warn_three_signal_keys_with_1d_planner(self) -> Self:
        # improvement_patience / plateau_window / plateau_threshold are
        # consumed by the N-D planners (bayesian, optuna) via
        # `evaluate_three_signal_convergence`. The 1D SLA planners
        # (monotonic_sla, smooth_isotonic) terminate on
        # max_iterations + algorithm-specific signals
        # (*_precision_reached / *_no_failure_in_range / *_no_pass_in_range)
        # and silently ignore these knobs. Warn the user so a config like
        # `--search-recipe max-concurrency-under-sla --improvement-patience 3`
        # doesn't masquerade as an early-stop request.
        if str(self.planner) not in {
            str(SearchPlannerType.MONOTONIC_SLA),
            str(SearchPlannerType.SMOOTH_ISOTONIC),
        }:
            return self
        three_signal_fields = (
            "improvement_patience",
            "plateau_window",
            "plateau_threshold",
        )
        explicitly_set = [f for f in three_signal_fields if f in self.model_fields_set]
        if not explicitly_set:
            return self
        warnings.warn(
            f"Three-signal convergence keys ({', '.join(explicitly_set)}) only "
            f"apply to N-D planners (bayesian, optuna). Selected planner "
            f"'{self.planner}' uses algorithm-specific termination "
            f"(max_iterations + *_precision_reached / *_no_failure_in_range / "
            f"*_no_pass_in_range). The values you set for "
            f"{', '.join(explicitly_set)} will be ignored.",
            UserWarning,
            stacklevel=2,
        )
        return self

    _SINGLE_OBJ_ACQUISITIONS: ClassVar[frozenset[str]] = frozenset(
        {"logei", "qlogei", "qnei", "qlognei"}
    )
    _MULTI_OBJ_ACQUISITIONS: ClassVar[frozenset[str]] = frozenset(
        {"qehvi", "qnehvi", "qlognehvi"}
    )

    @model_validator(mode="after")
    def _check_acquisition_matches_objective_count(self) -> Self:
        acq = self.optuna_acquisition
        if acq is None:
            return self
        n = len(self.objectives)
        if acq in self._SINGLE_OBJ_ACQUISITIONS and n != 1:
            raise ValueError(
                f"--optuna-acquisition {acq!r} is single-objective; got "
                f"{n} objectives. Use a multi-objective acquisition "
                "(qlognehvi recommended) or reduce to a single objective."
            )
        if acq in self._MULTI_OBJ_ACQUISITIONS and n == 1:
            raise ValueError(
                f"--optuna-acquisition {acq!r} is multi-objective; got 1 "
                "objective. Use a single-objective acquisition (qlognei "
                "recommended) or add a second objective."
            )
        return self


# QMC sampling sweep types (``LatinHypercubeSweep``, ``SobolSweep``,
# ``SamplingDimension``) live in ``sampling.py`` to keep this file
# under the ergonomics file-size cap. They are first-class members of the
# ``SweepConfig`` discriminated union below; the import is mid-file because
# ``sampling.py`` imports ``_GridSweepBase`` from this module, so it
# must be defined first.
from aiperf.config.sweep.sampling import LatinHypercubeSweep  # noqa: E402, I001
from aiperf.config.sweep.sampling import SamplingDimension  # noqa: E402
from aiperf.config.sweep.sampling import SobolSweep  # noqa: E402


SweepConfig = Annotated[
    GridSweep
    | ZipSweep
    | ScenarioSweep
    | AdaptiveSearchSweep
    | SobolSweep
    | LatinHypercubeSweep,
    Discriminator("type"),
]


class SweepVariation(BaseConfig):
    """Metadata for a single sweep variation."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(description="Zero-based variation index.")
    label: str = Field(description="Human-readable label for this variation.")
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter values that differ from base config.",
    )

    @property
    def dir_name(self) -> str:
        """Filesystem-safe directory name for this variation's artifacts.

        Naming convention is
        ``{last_segment_of_dotted_key}_{value}``. Multi-dim variations
        join components with ``__``. Falls back to ``label`` when
        ``values`` is empty (e.g. ``"base"`` for non-sweep runs, or
        named scenarios like ``"scenario_a"``).

        Examples:

        >>> # values={"phases.profiling.concurrency": 10}    -> "concurrency_10"
        >>> # values={"phases.profiling.request_rate": 5.0}  -> "request_rate_5.0"
        >>> # values={"a.b": 1, "c.d": 2}                    -> "b_1__d_2"
        >>> # values={}                                      -> self.label
        """
        return _format_dir_name(self.values) or self.label


def _format_dir_name(values: dict[str, Any]) -> str:
    """Format a variation's ``values`` dict into a filesystem-safe dir name.

    Returns ``""`` when ``values`` is empty so callers can fall back to a
    user-provided label. Single-dim sweeps produce ``{last_seg}_{value}``
    (e.g. ``concurrency_10``); multi-dim joins components with ``__``.
    """
    if not values:
        return ""
    parts = []
    for dotted_key, value in values.items():
        last_seg = dotted_key.rsplit(".", 1)[-1]
        parts.append(f"{last_seg}_{value}")
    return _sanitize_dir_segment("__".join(parts))


def _sanitize_dir_segment(segment: str) -> str:
    """Strip filesystem-unsafe characters from a single path segment."""
    import re as _re

    sanitized = _re.sub(r"[/\\]|\.\.", "", segment)
    return _re.sub(r'[<>:"|?*]', "", sanitized)


# Resolve the forward ref on `SearchRecipeOutput.adaptive_search` once
# `AdaptiveSearchSweep` is concrete in this module. `_base.py` makes a
# best-effort rebuild during its own init, but skips when this module
# is mid-import; we re-run it here so the rebuild always lands.
import sys as _sys  # noqa: E402  -- import order glue

from aiperf.config.sweep.expand import (  # noqa: E402  re-export
    MAGIC_LIST_FIELDS,
    _set_nested_value,
    expand_sweep,
)

if "aiperf.search_recipes._base" in _sys.modules:
    _base = _sys.modules["aiperf.search_recipes._base"]
    if hasattr(_base, "SearchRecipeOutput"):
        _base.SearchRecipeOutput.model_rebuild(
            _types_namespace={"AdaptiveSearchSweep": AdaptiveSearchSweep}
        )
