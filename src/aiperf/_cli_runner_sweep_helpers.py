# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep-aggregate helpers for ``aiperf.cli_runner``.

Lives in a sibling module to ``_cli_runner_helpers.py`` purely to keep
both files under the 500-line ergonomics cap; downstream callers should
prefer the re-export from ``_cli_runner_helpers``:

    >>> from aiperf._cli_runner_helpers import aggregate_sweep_and_export

The single public entry point is :func:`aggregate_sweep_and_export`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkPlan
    from aiperf.orchestrator.models import RunResult


VariationKey = tuple[tuple[str, Any], ...]
"""Hashable key for grouping :class:`RunResult` by variation values.

Built from ``tuple(sorted(result.variation_values.items()))``; needed
because ``dict`` itself is not hashable so cannot key a Python ``dict``.
"""


def _variation_key(values: dict[str, Any]) -> VariationKey:
    """Hashable, order-stable key for a variation's values dict.

    Example:
        >>> _variation_key({"concurrency": 10, "isl": 128})
        (('concurrency', 10), ('isl', 128))
    """
    return tuple(sorted(values.items()))


def _group_results_by_variation(
    results: list[RunResult],
) -> dict[VariationKey, list[RunResult]]:
    """Group ``results`` by their ``variation_values`` dict.

    Insertion order of the returned dict mirrors the first-seen order of
    each unique variation key; this becomes the row order of the sweep
    CSV (callers rely on it for stable diffing across reruns).

    Example:
        >>> # 3 results: 2 trials at concurrency=10, 1 at concurrency=20
        >>> # → groups[(('concurrency', 10),)] has length 2
        >>> # → groups[(('concurrency', 20),)] has length 1
    """
    groups: dict[VariationKey, list[RunResult]] = {}
    for result in results:
        key = _variation_key(result.variation_values)
        groups.setdefault(key, []).append(result)
    return groups


def _compute_sweep_parameters(
    groups: dict[VariationKey, list[RunResult]],
) -> list[dict[str, Any]]:
    """Derive ``[{"name": ..., "values": [...]}, ...]`` from grouped results.

    The values list preserves first-seen order across the variation
    keys, matching what the user's sweep config produced upstream.

    Example:
        >>> # groups keys: [(('concurrency', 10),), (('concurrency', 20),)]
        >>> _compute_sweep_parameters(groups)  # doctest: +SKIP
        [{'name': 'concurrency', 'values': [10, 20]}]
    """
    seen: dict[str, list[Any]] = {}
    for key in groups:
        for name, value in key:
            bucket = seen.setdefault(name, [])
            if value not in bucket:
                bucket.append(value)
    return [{"name": name, "values": values} for name, values in seen.items()]


def _confidence_metric_to_stats(metric: Any) -> dict[str, Any]:
    """Project a :class:`ConfidenceMetric` to the per-cell stats dict."""
    return {
        "mean": metric.mean,
        "std": metric.std,
        "min": metric.min,
        "max": metric.max,
        "cv": metric.cv,
        "ci_low": metric.ci_low,
        "ci_high": metric.ci_high,
        "unit": metric.unit,
    }


def _json_metric_to_stats(metric: Any) -> dict[str, Any]:
    """Project a :class:`JsonMetricResult` (single-trial path) to stats dict.

    ``std``/``cv``/CI fields collapse to zero when only one trial exists
    (matches main's behavior in PR #699 — a single sample has no spread).
    """
    avg = metric.avg if metric.avg is not None else 0.0
    return {
        "mean": avg,
        "std": 0.0,
        "min": metric.min if metric.min is not None else avg,
        "max": metric.max if metric.max is not None else avg,
        "cv": 0.0,
        "ci_low": avg,
        "ci_high": avg,
        "unit": metric.unit,
    }


def _aggregate_group_to_stats(
    group: list[RunResult], confidence_level: float
) -> dict[str, Any] | None:
    """Reduce a single variation group to its per-metric stats dict.

    Routes:
      - ``len(group) == 1`` → read the single result's ``summary_metrics`` directly.
      - ``len(group) > 1``  → :class:`ConfidenceAggregation` for mean/std/CI.

    Returns ``None`` when the group has no usable metrics (all trials
    failed, or single-trial run had no summary).

    Example:
        >>> # Concurrency=10, 3 trials: throughput=[100, 110, 105]
        >>> # → {"request_throughput_avg": {"mean": 105.0, "std": 5.0, ...}}
    """
    from aiperf.orchestrator.aggregation.confidence import ConfidenceAggregation

    if not group:
        return None

    if len(group) == 1:
        single = group[0]
        if not single.success or not single.summary_metrics:
            return None
        return {
            metric_name: _json_metric_to_stats(metric_result)
            for metric_name, metric_result in single.summary_metrics.items()
        }

    aggregation = ConfidenceAggregation(confidence_level=confidence_level)
    try:
        agg_result = aggregation.aggregate(group)
    except ValueError:
        return None
    return {
        metric_name: _confidence_metric_to_stats(metric)
        for metric_name, metric in agg_result.metrics.items()
    }


def _build_per_combination_stats(
    groups: dict[VariationKey, list[RunResult]], confidence_level: float
) -> dict[Any, dict[str, Any]]:
    """Build the ``per_combination_stats`` dict consumed by SweepAnalyzer.compute.

    Keys are :class:`ParameterCombination` (hashable, knows ``to_dict``);
    values are per-metric stats dicts as produced by
    :func:`_aggregate_group_to_stats`.

    Groups with no usable metrics are dropped, which keeps SweepAnalyzer
    from producing rows of nans downstream.
    """
    from aiperf.orchestrator.aggregation.sweep import ParameterCombination

    per_combination_stats: dict[Any, dict[str, Any]] = {}
    for key, group in groups.items():
        stats = _aggregate_group_to_stats(group, confidence_level)
        if stats is None:
            continue
        combo = ParameterCombination(dict(key))
        per_combination_stats[combo] = stats
    return per_combination_stats


async def aggregate_sweep_and_export(
    results: list[RunResult],
    plan: BenchmarkPlan,
    base_dir: Path,
    logger: AIPerfLogger,
) -> Path | None:
    """Group, aggregate, and export sweep results to ``base_dir/sweep_aggregate/``.

    Pipeline:

    1. Group ``results`` by ``variation_values``.
    2. For each group: aggregate trials (multi-trial) or read summary
       directly (single-trial).
    3. Run :meth:`SweepAnalyzer.compute` over the grouped stats.
    4. Write ``profile_export_aiperf_sweep.json`` and ``.csv`` via the
       sweep exporters.

    Returns the directory written to, or ``None`` if there were no
    results to aggregate (graceful no-op).

    Example:
        >>> # 3 variations × 1 trial → 1 row per variation in the CSV
        >>> # 3 variations × 3 trials → ConfidenceAggregation across each cell
        >>> await aggregate_sweep_and_export(results, plan, base_dir, logger)  # doctest: +SKIP
    """
    import asyncio

    from aiperf.exporters.aggregate import (
        AggregateExporterConfig,
        AggregateSweepCsvExporter,
        AggregateSweepJsonExporter,
    )
    from aiperf.orchestrator.aggregation.base import AggregateResult
    from aiperf.orchestrator.aggregation.sweep import SweepAnalyzer

    if not results:
        logger.info("No results to aggregate for sweep export.")
        return None

    groups = _group_results_by_variation(results)
    sweep_parameters = _compute_sweep_parameters(groups)
    per_combination_stats = _build_per_combination_stats(groups, plan.confidence_level)

    if not per_combination_stats:
        logger.warning(
            "Sweep aggregate skipped: no successful runs across all variations."
        )
        return None

    sweep_dict = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

    aggregate_dir = base_dir / "sweep_aggregate"
    await asyncio.to_thread(aggregate_dir.mkdir, parents=True, exist_ok=True)

    failed_runs = [
        {"label": r.label, "error": r.error} for r in results if not r.success
    ]
    aggregate_meta = AggregateResult(
        aggregation_type="sweep",
        num_runs=len(results),
        num_successful_runs=sum(1 for r in results if r.success),
        failed_runs=failed_runs,
    )
    exporter_config = AggregateExporterConfig(
        result=aggregate_meta, output_dir=aggregate_dir
    )
    json_exporter = AggregateSweepJsonExporter(exporter_config, sweep_dict)
    csv_exporter = AggregateSweepCsvExporter(exporter_config, sweep_dict)

    json_path, csv_path = await asyncio.gather(
        json_exporter.export(), csv_exporter.export()
    )
    logger.info(f"Sweep aggregate JSON written to: {json_path}")
    logger.info(f"Sweep aggregate CSV written to: {csv_path}")
    return aggregate_dir
