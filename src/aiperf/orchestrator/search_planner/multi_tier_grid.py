# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-tier grid evaluation for SLO boundary discovery.

In grid mode, all tiers are evaluated at every grid point. No bracket
tracking or probe allocation is needed — per-tier boundaries are derived
post-hoc as the maximum passing grid point per tier.

This module provides :func:`evaluate_tiers_on_grid`, which takes grid sweep
results (the ``per_combination_metrics`` block from ``SweepAnalyzer.compute``)
and a list of :class:`SLOTier` definitions, and returns per-tier
:class:`TierResult` records.

Integration point: called from the ``sla_breach_knee`` post-process path
(or a sibling multi-tier post-process) when ``sla_tiers`` are present in
the recipe params.
"""

from __future__ import annotations

from typing import Any

from aiperf.common.finite import is_finite_value
from aiperf.config.sweep.adaptive import SLOTier
from aiperf.orchestrator.aggregation.sweep_sla_filter import (
    OP_TO_FN,
    read_metric_value,
)
from aiperf.orchestrator.search_planner.multi_tier_models import TierResult

__all__ = ["evaluate_tiers_on_grid"]


def evaluate_tiers_on_grid(
    per_combination_metrics: list[dict[str, Any]],
    tiers: list[SLOTier],
    swept_param: str,
    global_filters: list[Any] | None = None,
) -> list[TierResult]:
    """Evaluate all tiers at every grid point and derive per-tier boundaries.

    For each grid point (sorted ascending by swept value), evaluates every
    tier's filters against that point's metrics. The boundary for each tier
    is the maximum grid point where all of that tier's filters pass.

    Args:
        per_combination_metrics: The ``per_combination_metrics`` list from
            sweep aggregate output. Each entry has ``parameters`` (dict) and
            ``metrics`` (dict).
        tiers: List of SLO tier definitions (2-10 tiers).
        swept_param: Dotted path of the swept parameter (e.g.
            ``"phases.profiling.concurrency"``). Used to extract the grid
            point value from each combination's parameters.
        global_filters: Optional list of global SLA filters that compose
            with each tier's filters during evaluation.

    Returns:
        One :class:`TierResult` per tier with boundary derived from grid results.
    """
    leaf = swept_param.split(".")[-1]

    # Resolve which key the parameters block uses (full path or leaf)
    param_key = _resolve_param_key(per_combination_metrics, swept_param, leaf)

    # Filter to rows that have the swept param and sort ascending
    rows = [
        r for r in per_combination_metrics if param_key in (r.get("parameters") or {})
    ]
    rows.sort(key=lambda r: float(r["parameters"][param_key]))

    # Evaluate all tiers at every grid point
    results: list[TierResult] = []
    for tier in tiers:
        tier_result = _evaluate_tier_on_grid(rows, tier, param_key, global_filters)
        results.append(tier_result)

    return results


def _resolve_param_key(rows: list[dict[str, Any]], swept_param: str, leaf: str) -> str:
    """Determine whether rows key by full dotted path or leaf name."""
    if not rows:
        return leaf
    first_params = rows[0].get("parameters") or {}
    if swept_param in first_params:
        return swept_param
    return leaf


def _evaluate_tier_on_grid(
    rows: list[dict[str, Any]],
    tier: SLOTier,
    param_key: str,
    global_filters: list[Any] | None = None,
) -> TierResult:
    """Evaluate a single tier across all grid points and derive its boundary.

    Boundary = max passing grid point for this tier.
    """
    # Compose filters
    effective_filters = list(tier.filters) + (global_filters or [])

    max_passing: int | None = None
    min_failing: int | None = None
    binding_constraint: dict[str, Any] | None = None
    points_evaluated = 0

    for row in rows:
        value = row["parameters"][param_key]
        metrics = row.get("metrics") or {}
        points_evaluated += 1

        passed, first_breach = _tier_passes(effective_filters, metrics)

        if passed:
            int_value = int(float(value))
            if max_passing is None or int_value > max_passing:
                max_passing = int_value
        else:
            int_value = int(float(value))
            if min_failing is None or int_value < min_failing:
                min_failing = int_value
                binding_constraint = first_breach

    # Derive convergence status
    if max_passing is not None:
        convergence_status = "converged"
        convergence_reason = "grid_complete"
    elif points_evaluated > 0:
        convergence_status = "no_pass_in_range"
        convergence_reason = "no_pass_in_range"
    else:
        convergence_status = "partial"
        convergence_reason = "no_grid_points"

    return TierResult(
        label=tier.label,
        boundary_concurrency=max_passing,
        convergence_status=convergence_status,
        convergence_reason=convergence_reason,
        binding_constraint=binding_constraint,
        bracket_lower=max_passing,
        bracket_upper=min_failing,
        confidence_interval=None,
        probe_count=points_evaluated,
        filters=[
            {
                "metric_tag": f.metric_tag,
                "stat": f.stat,
                "op": f.op,
                "threshold": f.threshold,
            }
            for f in effective_filters
        ],
    )


def _tier_passes(
    filters: list, metrics: dict[str, Any]
) -> tuple[bool, dict[str, Any] | None]:
    """Check if all filters pass for a given metrics dict.

    Returns (passed, first_breach) where first_breach is the first failing
    filter's details (or None if all passed).
    """
    for f in filters:
        observed = read_metric_value(metrics, f.metric_tag, f.stat)
        if observed is None:
            return False, {
                "metric_tag": f.metric_tag,
                "stat": f.stat,
                "op": f.op,
                "threshold": f.threshold,
                "observed": None,
            }
        fn = OP_TO_FN[f.op]
        if not fn(observed, f.threshold):
            return False, {
                "metric_tag": f.metric_tag,
                "stat": f.stat,
                "op": f.op,
                "threshold": f.threshold,
                "observed": observed if is_finite_value(observed) else None,
            }
    return True, None
