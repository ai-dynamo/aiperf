# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SearchPlanner ABC and SearchIteration dataclass.

The BO config itself lives on ``AdaptiveSearchSweep`` in
``aiperf.config.sweep``; the leaf sub-types (``SearchSpaceDimension``,
``SLAFilter``) live in ``aiperf.config.sweep.adaptive``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.sweep import SweepVariation
    from aiperf.orchestrator.models import RunResult


__all__ = ["SearchIteration", "SearchPlanner"]


@dataclass
class SearchIteration:
    """One entry in the BO trajectory log.

    Written to search_history.json incrementally after each iteration. `results`
    is the per-trial RunResult list at this BO point (length == plan.trials
    for FixedTrialsStrategy).
    """

    iteration_idx: int
    variation_values: dict[str, Any]
    objective_value: float | None = None
    objective_values: list[float] | None = None
    """Vector form of the objective for multi-objective Pareto BO.

    When populated, ``objective_value`` mirrors ``objective_values[0]`` so
    single-objective consumers can read the scalar form. Multi-objective
    planners (Optuna with ``directions=[...]``) write the full vector here;
    the search-history exporter and Pareto-frontier code consume this
    field directly.
    """
    results: list[Any] = field(default_factory=list)
    feasible: bool = True
    """Whether the iteration's RunResults satisfied all configured SLA filters.

    Used by best-result selection to enforce feasibility-first ranking. Defaults
    to True so iterations from configs with no SLA filters degenerate to plain
    ranking unchanged. Computed by each subclass's ``tell`` (e.g.
    ``OptunaSearchPlanner.tell``, ``MonotonicSLASearchPlanner.tell``,
    ``SmoothIsotonicSLAPlanner.tell``): an iteration is feasible iff at least
    one trial's metrics passed every filter.
    """
    non_monotonic_warning: bool = False
    """True iff this iteration's verdict revealed a non-monotonic SLA boundary.

    Set by ``MonotonicSLASearchPlanner`` and ``SmoothIsotonicSLAPlanner``: a
    feasible verdict appears at a swept value at-or-above the latched
    ``infeasible_min``, or an infeasible verdict appears at-or-below the
    latched ``feasible_max``. Surfaced into ``search_history.json`` so the
    post-run pipeline (``boundary_summary``) can flag the trajectory
    without re-deriving it. Defaults to False so the BO planner and other
    consumers stay unchanged.
    """


class SearchPlanner(ABC):
    """Abstract base for adaptive outer-loop planners.

    Shipped implementations:
      - ``BayesianSearchPlanner`` (curated Optuna+BoTorch preset; auto-selects
        qLogNEI / qLogNEHVI based on objective count)
      - ``OptunaSearchPlanner`` (expert mode: Optuna TPE / GP / BoTorch backends)
      - ``MonotonicSLASearchPlanner`` (1D binary search on SLA boundary)
      - ``SmoothIsotonicSLAPlanner`` (1D PAVA + PCHIP isotonic regression)
    """

    @abstractmethod
    def ask(self) -> tuple[BenchmarkConfig, SweepVariation] | None:
        """Return (cfg, variation) for the next iteration, or None when done.

        The cfg is a deep-copied BenchmarkConfig with the proposed values
        substituted at their dotted paths. The SweepVariation has
        `index = iteration_idx`, `label = "search_iter_NNNN"`, and
        `values = {path: proposed_value, ...}` so downstream
        `aggregate_sweep_and_export` groups results naturally.
        """

    @abstractmethod
    def tell(self, variation: SweepVariation, results: list[RunResult]) -> None:
        """Absorb results for the most recent `ask()` variation.

        Implementations should reject unpaired calls, update their convergence state,
        and append exactly one SearchIteration for the reported variation.
        """

    @abstractmethod
    def is_converged(self) -> bool:
        """True when max_iterations exhausted or plateau detected."""

    @abstractmethod
    def history(self) -> list[SearchIteration]:
        """All iterations recorded so far, in submission order."""

    @property
    def iter_count(self) -> int:
        """Number of completed ask/tell cycles (== length of history()).

        Default reads ``self._iter`` which all shipped concrete planners
        maintain. Subclasses with a different counter representation MUST
        override this property; a missing ``_iter`` raises
        :class:`AttributeError` so the gap surfaces immediately rather
        than silently logging "0 iterations" for every cancel / converge
        / abort terminal state.
        """
        return int(self._iter)  # type: ignore[attr-defined]

    def convergence_reason(self) -> str | None:
        """Which signal caused the most recent True from is_converged().

        Shared base-level strings: ``"max_iterations"``,
        ``"improvement_patience"``, ``"plateau_cv"``, or ``None`` (not
        converged, or implementation does not track reasons). Subclasses
        extend with planner-specific reasons — see each subclass's
        ``convergence_reason`` for the full set (e.g.
        ``OptunaSearchPlanner`` adds ``"posterior_regret_bound"`` /
        ``"emmr"``; ``MonotonicSLASearchPlanner`` adds
        ``"monotonic_precision_reached"`` /
        ``"monotonic_no_pass_in_range"`` /
        ``"monotonic_no_failure_in_range"``;
        ``SmoothIsotonicSLAPlanner`` adds ``"smooth_isotonic_*"`` strings).
        Surfaced by the orchestrator on loop exit and recorded in
        ``search_history.json`` so users can audit why a run terminated
        where it did. Default returns None; concrete subclasses SHOULD
        override if they support multiple termination signals.
        """
        return None

    def boundary_summary(self) -> dict[str, Any] | None:
        """Optional SLA-feasibility boundary report for 1D planners.

        Returns a dict describing the discovered SLA-feasibility boundary on
        the swept axis, or ``None`` when the planner has no single-boundary
        concept (e.g. N-D Bayesian / Optuna search over multiple dimensions),
        when its history is empty, or when no verdict has latched yet.

        Returned dict shape (byte-shape-identical across planners so
        downstream consumers don't branch on planner type)::

            {
                "swept_dim_path": str,
                "feasible_max":   {"value": ..., "iteration_idx": int,
                                   "objective_value": float | None} | None,
                "infeasible_min": {"value": ..., "iteration_idx": int,
                                   "first_breach": {...}} | None,
                # Optional, set by 1D-SLA planners that track them:
                "non_monotonic_warning": bool,
                "convergence_reason":    str | None,
                # Smooth-isotonic-only extras (boundary_type, boundary_ci_low,
                # boundary_ci_high, binding_constraint, ...) when applicable.
            }

        Surfaced into ``search_history.json`` so post-run artifacts
        (``sla_breach.json``) avoid re-deriving the boundary from the raw
        trajectory. Default returns None; concrete 1D-SLA subclasses
        (``MonotonicSLASearchPlanner``, ``SmoothIsotonicSLAPlanner``,
        ``OptunaSearchPlanner`` for 1D spaces) override.
        """
        return None
