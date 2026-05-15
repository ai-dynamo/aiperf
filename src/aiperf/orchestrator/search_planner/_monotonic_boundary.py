# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary-summary helpers for ``MonotonicSLASearchPlanner``.

The planner owns the truth for ``feasible_max`` / ``infeasible_min``
(latched during bisection from per-point verdict logs); these helpers
project that latched state into the ``boundary_summary`` shape consumed
by ``write_search_history``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.orchestrator.search_planner._sla_helpers import first_failing_filter

if TYPE_CHECKING:
    from aiperf.orchestrator.search_planner.base import SearchIteration
    from aiperf.orchestrator.search_planner.monotonic import MonotonicSLASearchPlanner


__all__ = ["compute_boundary_summary"]


def compute_boundary_summary(
    planner: MonotonicSLASearchPlanner,
) -> dict[str, Any] | None:
    """Boundary-summary block precomputed from planner state.

    Returns None when no verdict has latched yet (zero iterations or all
    provisional). When at least one bound exists, the returned dict shape
    matches ``write_search_history``'s history-derived fallback so consumers
    don't branch on planner type.
    """
    if planner.feasible_max is None and planner.infeasible_min is None:
        return None
    return {
        "swept_dim_path": planner._dim.path,
        "feasible_max": _feasible_max_block(planner),
        "infeasible_min": _infeasible_min_block(planner),
    }


def _feasible_max_block(
    planner: MonotonicSLASearchPlanner,
) -> dict[str, Any] | None:
    """Per-iteration record at the latched ``feasible_max`` swept value."""
    if planner.feasible_max is None:
        return None
    iteration = _latest_iteration_at(planner, planner.feasible_max, feasible=True)
    if iteration is None:
        return None
    return {
        "value": planner.feasible_max,
        "iteration_idx": iteration.iteration_idx,
        "objective_value": iteration.objective_value,
    }


def _infeasible_min_block(
    planner: MonotonicSLASearchPlanner,
) -> dict[str, Any] | None:
    """Per-iteration record at the latched ``infeasible_min`` swept value."""
    if planner.infeasible_min is None:
        return None
    iteration = _latest_iteration_at(planner, planner.infeasible_min, feasible=False)
    if iteration is None:
        return None
    first_breach = first_failing_filter(iteration.results, planner._cfg.sla_filters)
    return {
        "value": planner.infeasible_min,
        "iteration_idx": iteration.iteration_idx,
        "first_breach": first_breach,
    }


def _latest_iteration_at(
    planner: MonotonicSLASearchPlanner, value: int, *, feasible: bool
) -> SearchIteration | None:
    """Return the latest iteration whose swept value matches and feasibility agrees.

    Stability-window probes can record multiple iterations at the same swept
    value with conflicting verdicts before one latches. Picking the latest
    matching-feasibility record matches the latched verdict the bracket
    reflects.
    """
    for iteration in reversed(planner._history):
        if iteration.variation_values.get(planner._dim.path) != value:
            continue
        if iteration.feasible == feasible:
            return iteration
    return None
