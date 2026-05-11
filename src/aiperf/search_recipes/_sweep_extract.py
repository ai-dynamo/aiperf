# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared sweep-aggregate point extraction for post-process handlers.

Lives in its own module so post-process handlers split across sibling
files (``_ttft_curve_fit``, plus the in-place :class:`DegradationKneeDetect`
defined in ``post_process.py``) can share the projection logic without
forming an import cycle through ``post_process.py``.
"""

from __future__ import annotations

from typing import Any

from aiperf.search_recipes._post_process_shared import StatLiteral


def _extract_points(
    sweep_aggregate: dict[str, Any],
    *,
    swept_param: str,
    metric_tag: str,
    stat: StatLiteral,
) -> list[tuple[float, float]]:
    """Pull ``(swept_value, metric_value)`` pairs from the sweep aggregate.

    Supports both per-combination layouts produced by ``SweepAnalyzer.compute``:

    - Multi-trial path: keys are flattened ``<metric_tag>_<stat>`` and the
      block carries ``{mean, std, min, max, cv, ci_low, ci_high, unit}`` --
      we read ``mean`` (the multi-trial average of the stat).
    - Single-trial path: keys are the metric tag alone and the block carries
      a collapsed ``{mean, std=0, min, max, ...}``; here we use ``mean``
      (which equals the JsonMetricResult.avg) regardless of the requested
      ``stat``. Single-trial sweeps don't carry per-stat percentiles.

    Skips rows missing the swept-parameter key or the requested metric;
    raises ``ValueError`` when nothing is left after filtering so handlers
    fail loudly rather than emit an empty artifact silently.
    """
    rows = sweep_aggregate.get("per_combination_metrics") or []
    flat_key = f"{metric_tag}_{stat}"
    # Recipes pass ``swept_param`` as a full dotted path
    # (``phases.profiling.concurrency``); the user-facing per-combination
    # ``parameters`` dict uses just the leaf name (``concurrency``).
    # Accept either form so recipes don't have to know which one
    # downstream code emits.
    short_key = swept_param.rsplit(".", 1)[-1]
    points: list[tuple[float, float]] = []
    for row in rows:
        params = row.get("parameters") or {}
        metrics = row.get("metrics") or {}
        if swept_param in params:
            param_value = params[swept_param]
        elif short_key in params:
            param_value = params[short_key]
        else:
            continue
        block = metrics.get(flat_key)
        if block is None or "mean" not in block:
            block = metrics.get(metric_tag)
        if block is None or "mean" not in block:
            continue
        points.append((float(param_value), float(block["mean"])))
    if not points:
        raise ValueError(
            f"post-process: sweep aggregate has no rows with parameter "
            f"{swept_param!r} and metric {metric_tag!r} (flat key {flat_key!r}); "
            f"check that the recipe swept that parameter and that the metric is "
            f"enabled (e.g. --streaming for time_to_first_token)."
        )
    points.sort(key=lambda pair: pair[0])
    return points
