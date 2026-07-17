# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental writer for search_history.json (BO trajectory log).

Called after every BO iteration so a partial trajectory survives a crash.
Sits next to sweep_aggregate/ in the artifact dir, NOT inside it.

Output schema::

    {
      "config": {"objectives": [{metric, stat, direction, threshold}, ...],
                 "outcome_constraints": [{metric, op, bound}, ...],
                 ...rest of AdaptiveSearchSweep including sla_filters},
      "iterations": [
        {"iteration_idx": int, "variation_values": {...},
         "objective_values": list[float] | None, "feasible": bool,
         "non_monotonic_warning": bool}
      ],
      "best_trials": [
        {"iteration_idx": int, "objective_values": list[float],
         "variation_values": {...}, "feasible": bool,
         "feasible_count": int, "pareto_rank": int}
      ] | null,
      "boundary_summary": {"swept_dim_path": str,
                           "feasible_max": {...} | null,
                           "infeasible_min": {..., "first_breach": {...}} | null}
                          | null,
      "recipe": str | null,
      "convergence_reason": str | null
    }

``best_trials`` is a list because multi-objective runs surface the full
non-dominated (Pareto) front. For length-1 ``objectives`` the list is
length-1 (the single argmax/argmin under feasibility-first lexicographic
ranking). For length-N ``objectives`` every member of the front is emitted
with ``pareto_rank == 0``. Selection is feasibility-first: when at least
one iteration satisfied every configured SLA filter the front is computed
over the feasible subset; otherwise it falls back to the full pool with
``feasible_count == 0`` so the reader can tell the two cases apart.

``boundary_summary`` reports the literal SLA-feasibility boundary on
the swept axis: ``feasible_max`` is the highest swept-dim value seen among
feasible iterations; ``infeasible_min`` the lowest among infeasible. Distinct
from ``best_trials`` (the GP/objective winners). Only populated for 1D
search spaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.finite import scrub_non_finite
from aiperf.orchestrator.search_planner._sla_helpers import first_failing_filter

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.config.sweep import AdaptiveSearchSweep
    from aiperf.orchestrator.search_planner.base import SearchIteration

__all__ = ["write_search_history"]


def write_search_history(
    base_dir: Path,
    history: list[SearchIteration],
    cfg: AdaptiveSearchSweep,
    *,
    convergence_reason: str | None = None,
    planner: Any = None,
) -> None:
    """Write search_history.json under base_dir.

    See module docstring for the output schema and best-selection semantics.

    Args:
        base_dir: artifact dir; file lands at ``base_dir/search_history.json``.
        history: planner.history() snapshot. Mid-loop calls leave this open;
            terminal calls (after planner.ask() returned None) record the
            final trajectory.
        cfg: AdaptiveSearchSweep from the plan.
        convergence_reason: Examples include ``"max_iterations"``,
            ``"improvement_patience"``, ``"plateau_cv"``,
            ``"posterior_regret_bound"`` (Optuna terminator: Makarova 2022),
            ``"emmr"`` (Optuna terminator: Ishibashi 2023). The 1D-SLA
            planners additionally emit
            ``"monotonic_no_pass_in_range"``,
            ``"monotonic_no_failure_in_range"``,
            ``"monotonic_precision_reached"``,
            ``"smooth_isotonic_no_pass_in_range"``,
            ``"smooth_isotonic_no_failure_in_range"``,
            ``"smooth_isotonic_precision_reached"``,
            ``"smooth_isotonic_cliff_precision_reached"``, and
            ``"smooth_isotonic_pchip_fallback_bisection"``. ``None`` mid-loop.
        planner: Optional planner instance. When supplied, its
            ``boundary_summary()`` method is consulted; a non-None return is
            used in place of the history-derived computation. Planners with
            no single-boundary concept inherit the default ``None``.
    """
    iterations_payload = [
        {
            "iteration_idx": h.iteration_idx,
            "variation_values": h.variation_values,
            "objective_values": list(h.objective_values)
            if h.objective_values
            else None,
            "feasible": h.feasible,
            "non_monotonic_warning": h.non_monotonic_warning,
        }
        for h in history
    ]
    config_block = _build_config_block(cfg)
    payload: dict[str, Any] = {
        "config": config_block,
        "iterations": iterations_payload,
        "best_trials": _compute_best_trials(history, cfg),
        "boundary_summary": _resolve_boundary_summary(history, cfg, planner),
    }

    # Multi-tier extension: add tier_results, tier_metadata, config.tiers
    if planner is not None and _is_multi_tier(planner):
        config_block["tiers"] = [
            {"label": t.label, "filters": [f.model_dump() for f in t.filters]}
            for t in cfg.sla_tiers
        ]
        payload["tier_results"] = [tr.model_dump() for tr in planner.tier_results()]
        payload["tier_metadata"] = planner.tier_metadata()

    payload["recipe"] = cfg.recipe_name
    payload["convergence_reason"] = convergence_reason

    out = base_dir / "search_history.json"
    # Scrub non-finite values: orjson silently maps NaN/inf to JSON null,
    # which would erase the difference between "scorer returned NaN"
    # (objective_value=NaN) and "iteration was not scored"
    # (objective_value=None) in the on-disk trajectory.
    out.write_bytes(orjson.dumps(scrub_non_finite(payload), option=orjson.OPT_INDENT_2))


def _compute_best_trials(
    history: list[SearchIteration], cfg: AdaptiveSearchSweep
) -> list[dict] | None:
    """Compute best_trials list.

    For length-1 objectives: returns a length-1 list with the single argmax/argmin
    over feasible-then-fallback. For length>1: returns the non-dominated set
    (Pareto front) over feasible-then-fallback, with pareto_rank=0 for every
    front member. Returns None when no scored iterations exist.
    """
    from aiperf.common.enums import OptimizationDirection

    scored = [h for h in history if h.objective_values]
    feasible = [h for h in scored if h.feasible]
    ranking_pool = feasible if feasible else scored
    if not ranking_pool:
        return None

    n_obj = len(cfg.objectives)
    if n_obj == 1:
        direction = cfg.objectives[0].direction
        if direction == OptimizationDirection.MAXIMIZE:
            best = max(ranking_pool, key=lambda h: h.objective_values[0])
        else:
            best = min(ranking_pool, key=lambda h: h.objective_values[0])
        return [_serialize_trial(best, len(feasible), pareto_rank=0)]

    front = _pareto_front(ranking_pool, cfg.objectives)
    return [_serialize_trial(h, len(feasible), pareto_rank=0) for h in front]


def _serialize_trial(
    h: SearchIteration, feasible_count: int, *, pareto_rank: int
) -> dict:
    return {
        "iteration_idx": h.iteration_idx,
        "objective_values": list(h.objective_values) if h.objective_values else None,
        "variation_values": h.variation_values,
        "feasible": h.feasible,
        "feasible_count": feasible_count,
        "pareto_rank": pareto_rank,
    }


def _pareto_front(
    pool: list[SearchIteration], objectives: list
) -> list[SearchIteration]:
    """Non-dominated set under direction-aware comparison.

    A point p dominates q iff for every objective i, p is no worse than q,
    and for at least one objective p is strictly better. "Better" depends
    on each objective's direction.
    """
    from aiperf.common.enums import OptimizationDirection

    def dominates(a: SearchIteration, b: SearchIteration) -> bool:
        strictly_better_anywhere = False
        for i, obj in enumerate(objectives):
            av, bv = a.objective_values[i], b.objective_values[i]
            if obj.direction == OptimizationDirection.MAXIMIZE:
                if av < bv:
                    return False
                if av > bv:
                    strictly_better_anywhere = True
            else:
                if av > bv:
                    return False
                if av < bv:
                    strictly_better_anywhere = True
        return strictly_better_anywhere

    front: list[SearchIteration] = []
    for p in pool:
        if any(dominates(q, p) for q in pool if q is not p):
            continue
        front.append(p)
    return front


def _build_config_block(cfg: AdaptiveSearchSweep) -> dict[str, Any]:
    """Project an AdaptiveSearchSweep into the search_history.json `config` shape."""
    return {
        "planner": str(cfg.planner),
        "objectives": [
            {
                "metric": obj.metric,
                "stat": obj.stat,
                "direction": obj.direction.name,
                "threshold": obj.threshold,
            }
            for obj in cfg.objectives
        ],
        "outcome_constraints": [
            {"metric": c.metric, "op": c.op, "bound": c.bound}
            for c in cfg.outcome_constraints
        ],
        "max_iterations": cfg.max_iterations,
        "n_initial_points": cfg.n_initial_points,
        "random_seed": cfg.random_seed,
        "improvement_patience": cfg.improvement_patience,
        "plateau_window": cfg.plateau_window,
        "plateau_threshold": cfg.plateau_threshold,
        "search_space": [
            {"path": d.path, "lo": d.lo, "hi": d.hi, "kind": d.kind}
            for d in cfg.search_space
        ],
        "sla_filters": [f.model_dump() for f in cfg.sla_filters],
    }


def _resolve_boundary_summary(
    history: list[SearchIteration],
    cfg: AdaptiveSearchSweep,
    planner: Any,
) -> dict[str, Any] | None:
    """Prefer planner-precomputed boundary_summary; fall back to history-derived.

    Shape rules (mirrored in ``_compute_boundary_summary``): null on
    empty history or non-1D search-space; otherwise a dict with
    ``swept_dim_path`` plus optional ``feasible_max`` / ``infeasible_min``
    blocks. The planner-supplied path lets ``MonotonicSLASearchPlanner``
    own the truth (latched ``feasible_max``/``infeasible_min`` from per-point
    verdict logs) without forcing the exporter to re-derive feasibility.

    ``SearchPlanner.boundary_summary()`` is a concrete ABC method returning
    ``None`` by default, so the planner-precomputed branch is taken whenever
    a planner is supplied; planners with no boundary concept (Bayesian N-D)
    inherit the default ``None`` and we fall through to history derivation.
    """
    if not history or len(cfg.search_space) != 1:
        return None
    if planner is not None:
        precomputed = planner.boundary_summary()
        if precomputed is not None:
            return precomputed
    return _compute_boundary_summary(history, cfg)


def _compute_boundary_summary(
    history: list[SearchIteration], cfg: AdaptiveSearchSweep
) -> dict[str, Any] | None:
    """Derive ``boundary_summary`` from the iteration history.

    For BO-style planners (no latched bracket of its own) this scans the
    recorded iterations for the highest feasible swept value and the lowest
    infeasible swept value. The resulting block is byte-shape-identical to
    ``MonotonicSLASearchPlanner.boundary_summary()`` so downstream consumers
    don't branch on planner type.

    Returns None when no iterations were recorded; per-bound entries
    individually return None when their respective subset is empty.
    """
    swept_dim_path = cfg.search_space[0].path
    feasible_iters = [
        h for h in history if h.feasible and swept_dim_path in h.variation_values
    ]
    infeasible_iters = [
        h for h in history if not h.feasible and swept_dim_path in h.variation_values
    ]
    if not feasible_iters and not infeasible_iters:
        return None

    feasible_max: dict[str, Any] | None = None
    if feasible_iters:
        winner = max(feasible_iters, key=lambda h: h.variation_values[swept_dim_path])
        feasible_max = {
            "value": winner.variation_values[swept_dim_path],
            "iteration_idx": winner.iteration_idx,
            "objective_value": winner.objective_value,
        }

    infeasible_min: dict[str, Any] | None = None
    if infeasible_iters:
        loser = min(infeasible_iters, key=lambda h: h.variation_values[swept_dim_path])
        infeasible_min = {
            "value": loser.variation_values[swept_dim_path],
            "iteration_idx": loser.iteration_idx,
            "first_breach": first_failing_filter(loser.results, cfg.sla_filters),
        }

    return {
        "swept_dim_path": swept_dim_path,
        "feasible_max": feasible_max,
        "infeasible_min": infeasible_min,
    }


def _is_multi_tier(planner: Any) -> bool:
    """Check if planner is a MultiTierPlanner without importing it at module level."""
    return hasattr(planner, "tier_results") and hasattr(planner, "tier_metadata")
