# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental writer for search_history.json (BO trajectory log).

Called after every BO iteration so a partial trajectory survives a crash.
Sits next to sweep_aggregate/ in the artifact dir, NOT inside it.
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

    Schema:
        {
          "config": {...subset of AdaptiveSearchSweep, including sla_filters},
          "iterations": [
            {"iteration_idx": int, "variation_values": {...}, "objective_value": float | None}
          ],
          "best": {"iteration_idx": int, "objective_value": float, "variation_values": {...},
                   "feasible": bool, "feasible_count": int}
                  | null when no objectives recorded,
          "boundary_summary": {"swept_dim_path": str,
                               "feasible_max": {...} | null,
                               "infeasible_min": {..., "first_breach": {...}} | null}
                              | null when search_space dim count != 1 or no iterations,
          "recipe": str | null,  // recipe name when expanded via --search-recipe
          "convergence_reason": str | null  // why is_converged() fired, or null
        }

    Best-result selection is lexicographic feasibility-first: when at least one
    iteration satisfied every configured SLA filter, the best is chosen from
    the feasible subset; otherwise selection falls back to the full pool with
    ``feasible_count == 0`` so the reader can tell the two cases apart.

    ``boundary_summary`` (Plan-D) reports the literal SLA-feasibility boundary
    on the swept axis: ``feasible_max`` is the highest swept-dim value seen
    among feasible iterations; ``infeasible_min`` the lowest among infeasible.
    Distinct from ``best`` (the GP/objective winner). Only populated for 1D
    search spaces — multi-dim leaves it ``null`` since "highest swept value"
    has no scalar meaning across multiple axes. When ``planner`` is supplied
    and exposes ``boundary_summary()`` (e.g. ``MonotonicSLASearchPlanner``)
    the precomputed planner shape is used directly; otherwise the boundary is
    derived from ``history``.

    Args:
        base_dir: artifact dir; file lands at ``base_dir/search_history.json``.
        history: planner.history() snapshot. Mid-loop calls leave this open;
            terminal calls (after planner.ask() returned None) record the
            final trajectory.
        cfg: AdaptiveSearchSweep from the plan.
        convergence_reason: One of ``"max_iterations"``,
            ``"improvement_patience"``, ``"plateau_cv"``, or None when the
            history is being written mid-loop. Surfaced for post-run audit.
        planner: Optional planner instance. When it exposes a
            ``boundary_summary()`` method the precomputed dict is used in
            place of the history-derived computation. Pure duck-typing — no
            isinstance check, no widening of the SearchPlanner ABC required.
    """
    iterations_payload = [
        {
            "iteration_idx": h.iteration_idx,
            "variation_values": h.variation_values,
            "objective_value": h.objective_value,
            "objective_values": h.objective_values,
            "feasible": h.feasible,
            "non_monotonic_warning": h.non_monotonic_warning,
        }
        for h in history
    ]
    payload = {
        "config": _build_config_block(cfg),
        "iterations": iterations_payload,
        "best": _compute_best_payload(history, cfg),
        "best_trials": _compute_best_trials(history, cfg),
        "boundary_summary": _resolve_boundary_summary(history, cfg, planner),
        "recipe": cfg.recipe_name,
        "convergence_reason": convergence_reason,
    }
    out = base_dir / "search_history.json"
    out.write_bytes(orjson.dumps(scrub_non_finite(payload), option=orjson.OPT_INDENT_2))


def _compute_best_payload(
    history: list[SearchIteration], cfg: AdaptiveSearchSweep
) -> dict | None:
    """Lexicographic feasibility-first best-result selection.

    Prefers the best feasible iteration when any exist; falls back to the best
    of the full scored pool otherwise (with ``feasible_count == 0`` so the
    reader can distinguish the two cases). Returns ``None`` when the history
    contains no scored iterations at all.
    """
    from aiperf.orchestrator.aggregation.sweep import OptimizationDirection

    scored = [h for h in history if h.objective_value is not None]
    feasible = [h for h in scored if h.feasible]
    ranking_pool = feasible if feasible else scored
    if not ranking_pool:
        return None
    if cfg.objectives and cfg.objectives[0].direction == OptimizationDirection.MAXIMIZE:
        best = max(ranking_pool, key=lambda h: h.objective_value)
    else:
        best = min(ranking_pool, key=lambda h: h.objective_value)
    return {
        "iteration_idx": best.iteration_idx,
        "objective_value": best.objective_value,
        "variation_values": best.variation_values,
        "feasible": best.feasible,
        "feasible_count": len(feasible),
    }


def _compute_best_trials(
    history: list[SearchIteration], cfg: AdaptiveSearchSweep
) -> list[dict] | None:
    """Multi-objective best-trial list (Pareto-aware projection of ``best``).

    For single-objective configs this is a one-element list mirroring ``best``
    with ``objective_values`` (vector form). Returned as ``None`` only when
    the entire history is unscored, so consumers can always rely on
    ``best_trials[0]`` when ``len > 0``.
    """
    from aiperf.orchestrator.aggregation.sweep import OptimizationDirection

    scored = [h for h in history if h.objective_value is not None]
    feasible = [h for h in scored if h.feasible]
    ranking_pool = feasible if feasible else scored
    if not ranking_pool:
        return None
    if cfg.objectives and cfg.objectives[0].direction == OptimizationDirection.MAXIMIZE:
        best = max(ranking_pool, key=lambda h: h.objective_value)
    else:
        best = min(ranking_pool, key=lambda h: h.objective_value)
    return [
        {
            "iteration_idx": best.iteration_idx,
            "objective_value": best.objective_value,
            "objective_values": best.objective_values
            if best.objective_values is not None
            else [best.objective_value],
            "variation_values": best.variation_values,
            "feasible": best.feasible,
            "feasible_count": len(feasible),
        }
    ]


def _build_config_block(cfg: AdaptiveSearchSweep) -> dict[str, Any]:
    """Project an AdaptiveSearchSweep into the search_history.json `config` shape."""
    objectives = list(cfg.objectives or [])
    primary = objectives[0] if objectives else None
    return {
        "algorithm": str(cfg.planner),
        "planner": str(cfg.planner),
        "objective_metric": primary.metric if primary else None,
        "objective_stat": primary.stat if primary else None,
        "objective_direction": (str(primary.direction).upper() if primary else None),
        "objectives": [
            {
                "metric": o.metric,
                "stat": o.stat,
                "direction": str(o.direction).upper(),
                "threshold": o.threshold,
            }
            for o in objectives
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

    Plan-D shape rules (mirrored in ``_compute_boundary_summary``): null on
    empty history or non-1D search-space; otherwise a dict with
    ``swept_dim_path`` plus optional ``feasible_max`` / ``infeasible_min``
    blocks. The planner-supplied path lets ``MonotonicSLASearchPlanner``
    own the truth (latched ``feasible_max``/``infeasible_min`` from per-point
    verdict logs) without forcing the exporter to re-derive feasibility.
    """
    if not history or len(cfg.search_space) != 1:
        return None
    if planner is not None and hasattr(planner, "boundary_summary"):
        precomputed = planner.boundary_summary()
        if precomputed is not None:
            return precomputed
        # Planner exposed the hook but didn't latch a boundary (e.g. BO before
        # convergence); fall back to history-derivation so callers always
        # see a populated block for 1-D feasibility runs.
    return _compute_boundary_summary(history, cfg)


def _compute_boundary_summary(
    history: list[SearchIteration], cfg: AdaptiveSearchSweep
) -> dict[str, Any] | None:
    """Derive Plan-D boundary_summary from the iteration history.

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
