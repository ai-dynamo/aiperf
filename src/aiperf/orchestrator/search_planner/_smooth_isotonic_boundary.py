# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary-summary helpers for ``SmoothIsotonicSLAPlanner``.

The planner owns the latched ``feasible_max`` / ``infeasible_min`` and the
smooth-isotonic-specific reporting fields (``boundary_type``,
``binding_constraint``, ``boundary_ci_low/_hi``); these helpers project
that state into the ``boundary_summary`` shape consumed by
``write_search_history``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.orchestrator.search_planner._sla_helpers import first_failing_filter

if TYPE_CHECKING:
    from aiperf.orchestrator.search_planner.base import SearchIteration
    from aiperf.orchestrator.search_planner.smooth_isotonic import (
        SmoothIsotonicSLAPlanner,
    )


__all__ = ["compute_boundary_summary"]


def compute_boundary_summary(
    planner: SmoothIsotonicSLAPlanner,
) -> dict[str, Any] | None:
    """Boundary-summary block precomputed from planner state.

    Returns None when no verdict has latched yet. The returned dict has the
    same ``swept_dim_path`` / ``feasible_max`` / ``infeasible_min`` shape as
    the monotonic planner's, plus three optional fields specific to the
    smooth-isotonic outcome: ``boundary_type`` (smooth | cliff),
    ``binding_constraint`` (the SLO key with the worst sigma-normalized
    margin), and ``boundary_ci`` ({lo, hi} bootstrap CI when the replicate
    step ran).
    """
    if planner.feasible_max is None and planner.infeasible_min is None:
        return None
    out: dict[str, Any] = {
        "swept_dim_path": planner._dim.path,
        "feasible_max": _feasible_max_block(planner),
        "infeasible_min": _infeasible_min_block(planner),
    }
    if planner.boundary_type is not None:
        out["boundary_type"] = planner.boundary_type
    if planner.binding_constraint is not None:
        out["binding_constraint"] = planner.binding_constraint
    if planner.boundary_ci_low is not None and planner.boundary_ci_high is not None:
        out["boundary_ci"] = {
            "lo": planner.boundary_ci_low,
            "hi": planner.boundary_ci_high,
        }
    return out


def _feasible_max_block(
    planner: SmoothIsotonicSLAPlanner,
) -> dict[str, Any] | None:
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
    planner: SmoothIsotonicSLAPlanner,
) -> dict[str, Any] | None:
    if planner.infeasible_min is None:
        return None
    iteration = _latest_iteration_at(planner, planner.infeasible_min, feasible=False)
    if iteration is None:
        return None
    first_breach = first_failing_filter(iteration.results, planner._sla_filters)
    return {
        "value": planner.infeasible_min,
        "iteration_idx": iteration.iteration_idx,
        "first_breach": first_breach,
    }


def _latest_iteration_at(
    planner: SmoothIsotonicSLAPlanner, value: int, *, feasible: bool
) -> SearchIteration | None:
    for iteration in reversed(planner._history):
        if iteration.variation_values.get(planner._dim.path) != value:
            continue
        if iteration.feasible == feasible:
            return iteration
    return None
