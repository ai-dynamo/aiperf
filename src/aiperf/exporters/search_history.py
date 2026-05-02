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
          "config": {...subset of AdaptiveSearchConfig},
          "iterations": [
            {"iteration_idx": int, "variation_values": {...}, "objective_value": float | None}
          ],
          "best": {"iteration_idx": int, "objective_value": float, "variation_values": {...}}
                  | null when no objectives recorded,
          "convergence_reason": str | null  // why is_converged() fired, or null
        }

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
    from aiperf.orchestrator.aggregation.sweep import OptimizationDirection

    iterations_payload = [
        {
            "iteration_idx": h.iteration_idx,
            "variation_values": h.variation_values,
            "objective_value": h.objective_value,
        }
        for h in history
    ]
    scored = [h for h in history if h.objective_value is not None]
    if scored:
        if cfg.objective_direction == OptimizationDirection.MAXIMIZE:
            best = max(scored, key=lambda h: h.objective_value)
        else:
            best = min(scored, key=lambda h: h.objective_value)
        best_payload = {
            "iteration_idx": best.iteration_idx,
            "objective_value": best.objective_value,
            "variation_values": best.variation_values,
        }
    else:
        best_payload = None

    payload = {
        "config": {
            "algorithm": cfg.algorithm,
            "objective_metric": cfg.objective_metric,
            "objective_stat": cfg.objective_stat,
            "objective_direction": str(cfg.objective_direction),
            "max_iterations": cfg.max_iterations,
            "search_space": [
                {"path": d.path, "lo": d.lo, "hi": d.hi, "kind": d.kind}
                for d in cfg.search_space
            ],
        },
        "iterations": iterations_payload,
        "best": best_payload,
        "convergence_reason": convergence_reason,
    }
    out = base_dir / "search_history.json"
    out.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
