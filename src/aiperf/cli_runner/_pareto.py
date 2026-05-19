# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pareto-axes resolution and per-cell projection for sweep aggregation.

The functions here turn a single variation's RunResults into a 2D
``(x, y)`` Pareto point when (and only when) the active recipe declares
:class:`ParetoAxesSpec` ``pareto_axes``. The cross-variation pareto-frontier
computation (which marks ``pareto_optimal`` cells on the assembled set)
lives in :mod:`aiperf.cli_runner._sweep_aggregate`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from aiperf.search_recipes._pareto_axes import ParetoAxesSpec

if TYPE_CHECKING:
    from aiperf.config import BenchmarkPlan
    from aiperf.config.sweep import SweepVariation
    from aiperf.orchestrator.models import RunResult


def _resolve_pareto_axes(plan: BenchmarkPlan) -> ParetoAxesSpec | None:
    """Return the active recipe's ``pareto_axes`` or None.

    ``BenchmarkPlan`` doesn't carry a recipe instance - it only holds the
    recipe NAME at ``plan.sweep.recipe_name`` (post-expansion) or
    ``plan.sweep.search_recipe`` (test stubs). The actual class (which is
    where ``pareto_axes`` lives as a ``ClassVar``) has to be resolved via
    the plugin registry. Centralizing this lookup keeps every call site
    honest. Tests that need to override the lookup should patch this
    function directly.
    """
    sweep = getattr(plan, "sweep", None)
    name = None
    if sweep is not None:
        # ``recipe_name`` is the post-expansion audit field (set by
        # ``_recipe_output_to_dict`` for every recipe shape, plumbed onto the
        # final sweep block by ``_apply_recipe_scenarios`` /
        # ``_apply_recipe_sweep_parameters``). The ``search_recipe`` fallback
        # below exists only for test stubs - real ``_SweepBase`` subclasses
        # have no ``search_recipe`` field, only ``recipe_name``. Try
        # recipe_name first.
        name = getattr(sweep, "recipe_name", None) or getattr(
            sweep, "search_recipe", None
        )
    if not name:
        return None
    try:
        from aiperf.plugin.enums import PluginType
        from aiperf.plugin.plugins import get_class

        recipe_cls = get_class(PluginType.SEARCH_RECIPE, name)
    except Exception:
        return None
    return getattr(recipe_cls, "pareto_axes", None)


def _extract_axis_value(
    stats: dict[str, Any],
    variation_values: dict[str, Any],
    metric: str,
    stat: str,
) -> float | None:
    """Pull an axis value from per-cell stats; fall back to variation params.

    Tries the following sources in order:
      1. ``stats[metric][stat]`` - nested form produced by
         :func:`_json_metric_to_stats` and :func:`_confidence_metric_to_stats`.
         This is the canonical path: the recipe asks for ``stat="p95"`` and
         we read the p95 field from the metric block.
      2. ``stats[f"{metric}_{stat}"]["mean"]`` - flat form some upstream
         exporters emit (e.g. ``request_latency_p95`` as its own key with
         a ``mean`` aggregator value). Kept as a fallback for forward
         compatibility with the per-cell aggregate JSON shape.
      3. ``variation_values[metric]`` - parameter-as-axis case
         (e.g. ``concurrency`` on max-concurrency-under-sla).

    Returns ``None`` when no path yields a finite float.
    """
    block = stats.get(metric)
    if block is not None and isinstance(block, dict) and stat in block:
        try:
            v = float(block[stat])
            if math.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass

    flat_block = stats.get(f"{metric}_{stat}")
    if flat_block is not None and isinstance(flat_block, dict) and "mean" in flat_block:
        try:
            v = float(flat_block["mean"])
            if math.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass

    raw = variation_values.get(metric)
    if raw is not None:
        try:
            v = float(raw)
            if math.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass
    return None


def _aggregate_one_cell(
    cell_results: list[RunResult],
    plan: BenchmarkPlan,
    variation: SweepVariation,
) -> dict[str, Any] | None:
    """Aggregate one variation's trials into a Pareto-cell dict.

    Returns ``None`` when the plan's recipe declares no ``pareto_axes`` (no
    Pareto cell to project onto) or when either axis value is unavailable.
    Used by both the orchestrator's per-cell observer callback and by the
    end-of-sweep aggregator.
    """
    # Lazy import avoids the cycle with ``_sweep_aggregate`` (which imports
    # ``_resolve_pareto_axes`` from this module at top level).
    from aiperf.cli_runner._sweep_aggregate import _aggregate_group_to_stats

    axes: ParetoAxesSpec | None = _resolve_pareto_axes(plan)
    if axes is None:
        return None
    stats = _aggregate_group_to_stats(cell_results, plan.confidence_level)
    if stats is None:
        return None
    x = _extract_axis_value(stats, variation.values, axes.x_metric, axes.x_stat)
    y = _extract_axis_value(stats, variation.values, axes.y_metric, axes.y_stat)
    if x is None or y is None:
        return None
    return {
        "params": dict(variation.values),
        "x": x,
        "y": y,
        "pareto_optimal": False,
    }
