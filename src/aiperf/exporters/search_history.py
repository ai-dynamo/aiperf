# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental writer for search_history.json (BO trajectory log).

Called after every BO iteration so a partial trajectory survives a crash.
Sits next to sweep_aggregate/ in the artifact dir, NOT inside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.config.adaptive_search import AdaptiveSearchConfig
    from aiperf.orchestrator.search_planner.base import SearchIteration

__all__ = ["write_search_history"]


def write_search_history(
    base_dir: Path,
    history: list[SearchIteration],
    cfg: AdaptiveSearchConfig,
    *,
    convergence_reason: str | None = None,
) -> None:
    """Write search_history.json under base_dir.

    Schema:
        {
          "config": {...subset of AdaptiveSearchConfig, including sla_filters},
          "iterations": [
            {"iteration_idx": int, "variation_values": {...}, "objective_value": float | None}
          ],
          "best": {"iteration_idx": int, "objective_value": float, "variation_values": {...},
                   "feasible": bool, "feasible_count": int}
                  | null when no objectives recorded,
          "recipe": str | null,  // recipe name when expanded via --search-recipe
          "convergence_reason": str | null  // why is_converged() fired, or null
        }

    Best-result selection is lexicographic feasibility-first: when at least one
    iteration satisfied every configured SLA filter, the best is chosen from
    the feasible subset; otherwise selection falls back to the full pool with
    ``feasible_count == 0`` so the reader can tell the two cases apart.

    Args:
        base_dir: artifact dir; file lands at ``base_dir/search_history.json``.
        history: planner.history() snapshot. Mid-loop calls leave this open;
            terminal calls (after planner.ask() returned None) record the
            final trajectory.
        cfg: AdaptiveSearchConfig from the plan.
        convergence_reason: One of ``"max_iterations"``,
            ``"improvement_patience"``, ``"plateau_cv"``, or None when the
            history is being written mid-loop. Surfaced for post-run audit.
    """
    iterations_payload = [
        {
            "iteration_idx": h.iteration_idx,
            "variation_values": h.variation_values,
            "objective_value": h.objective_value,
            "feasible": h.feasible,
        }
        for h in history
    ]
    payload = {
        "config": _build_config_block(cfg),
        "iterations": iterations_payload,
        "best": _compute_best_payload(history, cfg),
        "recipe": cfg.recipe_name,
        "convergence_reason": convergence_reason,
    }
    out = base_dir / "search_history.json"
    out.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def _compute_best_payload(
    history: list[SearchIteration], cfg: AdaptiveSearchConfig
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
    if cfg.objective_direction == OptimizationDirection.MAXIMIZE:
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


def _build_config_block(cfg: AdaptiveSearchConfig) -> dict:
    """Project an AdaptiveSearchConfig into the search_history.json `config` shape."""
    return {
        "algorithm": cfg.algorithm,
        "objective_metric": cfg.objective_metric,
        "objective_stat": cfg.objective_stat,
        "objective_direction": str(cfg.objective_direction),
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
