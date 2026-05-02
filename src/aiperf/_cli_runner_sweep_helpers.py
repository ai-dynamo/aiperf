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

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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


def _per_variation_aggregate_dir(
    base_dir: Path,
    variation_label: str,
    sweep_mode: Any,
) -> Path:
    """Resolve the per-variation confidence-aggregate directory.

    Mirrors origin/main's ``SweepConfidenceStrategy.export_aggregates``
    layout, keyed by mode:

    - ``SweepMode.REPEATED``  -> ``<base>/aggregate/<variation_label>/``
    - ``SweepMode.INDEPENDENT`` (default fallback) -> ``<base>/<variation_label>/aggregate/``

    The variation label is the dotted-path key produced by
    ``expand_sweep`` (e.g. ``phases.profiling.concurrency=10``); we do
    NOT sanitize it here because it already comes from a controlled
    schema path, mirroring the layout used by ``MultiRunOrchestrator``
    when it writes the trial subtrees.

    Example:
        >>> from aiperf.common.enums import SweepMode
        >>> _per_variation_aggregate_dir(
        ...     Path("/tmp/x"),
        ...     "phases.profiling.concurrency=10",
        ...     SweepMode.REPEATED,
        ... )  # doctest: +SKIP
        PosixPath('/tmp/x/aggregate/phases.profiling.concurrency=10')
    """
    from aiperf.common.enums import SweepMode

    base_dir = Path(base_dir)
    if sweep_mode == SweepMode.REPEATED:
        return base_dir / "aggregate" / variation_label
    return base_dir / variation_label / "aggregate"


async def _export_one_variation_aggregate(
    *,
    group: list[RunResult],
    key: VariationKey,
    plan: BenchmarkPlan,
    base_dir: Path,
    logger: AIPerfLogger,
) -> Path | None:
    """Aggregate one variation's runs and write the per-cell JSON+CSV pair.

    Factored out of :func:`aggregate_per_variation_and_export` to keep
    that function under the 80-line ergonomics cap; one variation's work
    is the natural per-iteration body. Returns the directory written, or
    ``None`` when the cell was skipped (insufficient successful runs or
    aggregator rejected the group).
    """
    import asyncio

    from aiperf.exporters.aggregate import (
        AggregateConfidenceCsvExporter,
        AggregateConfidenceJsonExporter,
        AggregateExporterConfig,
    )
    from aiperf.orchestrator.aggregation.confidence import ConfidenceAggregation

    successful = [r for r in group if r.success]
    # Use the first result's pre-stamped variation_label (set by
    # orchestrator._stamp_variation_metadata); fall back to a
    # reconstructed key=value form if it was never stamped (shouldn't
    # happen in practice, but keeps this helper crash-free).
    variation_label = next(
        (r.variation_label for r in group if r.variation_label),
        ",".join(f"{k}={v}" for k, v in key),
    )

    if len(successful) < 2:
        logger.warning(
            f"Skipping per-variation aggregate for {variation_label!r}: "
            f"{len(successful)} successful run(s), need at least 2."
        )
        return None

    aggregation = ConfidenceAggregation(confidence_level=plan.confidence_level)
    try:
        aggregate_result = aggregation.aggregate(group)
    except ValueError as exc:
        logger.warning(
            f"Skipping per-variation aggregate for {variation_label!r}: "
            f"ConfidenceAggregation raised {exc}"
        )
        return None
    aggregate_result.metadata["cooldown_seconds"] = plan.cooldown_seconds
    aggregate_result.metadata["variation_label"] = variation_label
    aggregate_result.metadata["variation_values"] = dict(key)
    aggregate_result.metadata["sweep_mode"] = str(plan.parameter_sweep_mode)

    aggregate_dir = _per_variation_aggregate_dir(
        base_dir, variation_label, plan.parameter_sweep_mode
    )
    await asyncio.to_thread(aggregate_dir.mkdir, parents=True, exist_ok=True)

    exporter_config = AggregateExporterConfig(
        result=aggregate_result,
        output_dir=aggregate_dir,
    )
    json_exporter = AggregateConfidenceJsonExporter(exporter_config)
    csv_exporter = AggregateConfidenceCsvExporter(exporter_config)
    json_path, csv_path = await asyncio.gather(
        json_exporter.export(), csv_exporter.export()
    )
    logger.info(
        f"Per-variation aggregate ({variation_label}) JSON: {json_path}; CSV: {csv_path}"
    )
    return aggregate_dir


async def aggregate_per_variation_and_export(
    results: list[RunResult],
    plan: BenchmarkPlan,
    base_dir: Path,
    logger: AIPerfLogger,
) -> list[Path]:
    """Write a per-variation confidence aggregate (JSON+CSV) for each cell.

    Sweep version of ``aggregate_and_export`` from
    ``_cli_runner_helpers``: groups ``results`` by ``variation_values``
    and writes one ``profile_export_aiperf_aggregate.{json,csv}`` pair
    per variation that has >=2 successful runs. Variations with fewer
    successful runs are skipped with a warning (single-trial sweeps,
    runs that all failed in one cell, etc.) -- the per-cell run
    artifacts are still on disk, and the sweep aggregate runs
    independently downstream.

    The output path matches origin/main's
    ``SweepConfidenceStrategy.export_aggregates`` layout via
    :func:`_per_variation_aggregate_dir`, branching on
    ``plan.parameter_sweep_mode``.

    Returns the list of directories written (in group-iteration order),
    so the caller can log them. Empty list when no variation cleared
    the >=2-successful-run gate.

    Example:
        >>> # 2 variations x 3 trials, mode=independent
        >>> # writes:
        >>> #   <base>/phases.profiling.concurrency=10/aggregate/profile_export_aiperf_aggregate.json
        >>> #   <base>/phases.profiling.concurrency=10/aggregate/profile_export_aiperf_aggregate.csv
        >>> #   <base>/phases.profiling.concurrency=20/aggregate/profile_export_aiperf_aggregate.json
        >>> #   <base>/phases.profiling.concurrency=20/aggregate/profile_export_aiperf_aggregate.csv
        >>> await aggregate_per_variation_and_export(results, plan, base, logger)  # doctest: +SKIP
    """
    if not results:
        return []

    groups = _group_results_by_variation(results)
    written: list[Path] = []

    for key, group in groups.items():
        aggregate_dir = await _export_one_variation_aggregate(
            group=group,
            key=key,
            plan=plan,
            base_dir=base_dir,
            logger=logger,
        )
        if aggregate_dir is not None:
            written.append(aggregate_dir)

    return written


def _build_sweep_aggregate_result(
    results: list[RunResult],
    sweep_dict: dict[str, Any],
) -> Any:
    """Assemble the :class:`AggregateResult` consumed by the sweep exporters.

    Mirrors origin/main's ``SweepConfidenceStrategy.aggregate`` shape:
    stuffs the sweep sections (``best_configurations``, ``pareto_optimal``)
    into ``metadata`` and the per-cell rows into ``metrics``, so the
    exporters share their constructor with the sibling confidence
    exporters. Factored out of :func:`aggregate_sweep_and_export` to keep
    that function under the 80-line ergonomics cap.
    """
    from aiperf.orchestrator.aggregation.base import AggregateResult

    failed_runs = [
        {"label": r.label, "error": r.error} for r in results if not r.success
    ]
    sweep_metadata = dict(sweep_dict.get("metadata", {}))
    sweep_metadata["best_configurations"] = sweep_dict.get("best_configurations", {})
    sweep_metadata["pareto_optimal"] = sweep_dict.get("pareto_optimal", [])
    return AggregateResult(
        aggregation_type="sweep",
        num_runs=len(results),
        num_successful_runs=sum(1 for r in results if r.success),
        failed_runs=failed_runs,
        metadata=sweep_metadata,
        metrics=sweep_dict.get("per_combination_metrics", []),
    )


def _log_sweep_summary(aggregate_result: Any, logger: AIPerfLogger) -> None:
    """Stdout summary of best configurations and Pareto frontier.

    Factored out of :func:`aggregate_sweep_and_export` so the export
    pipeline stays inside the line budget; the stdout format is unchanged.
    """
    best_configs = aggregate_result.metadata.get("best_configurations", {})
    if best_configs:
        logger.info("")
        logger.info("Best Configurations:")
        if "best_throughput" in best_configs:
            bt = best_configs["best_throughput"]
            params_str = ", ".join(f"{k}={v}" for k, v in bt["parameters"].items())
            logger.info(
                f"  Best throughput: {params_str} ({bt['metric']:.2f} {bt['unit']})"
            )
        if "best_latency_p99" in best_configs:
            bl = best_configs["best_latency_p99"]
            params_str = ", ".join(f"{k}={v}" for k, v in bl["parameters"].items())
            logger.info(
                f"  Best latency (p99): {params_str} ({bl['metric']:.2f} {bl['unit']})"
            )

    pareto_optimal = aggregate_result.metadata.get("pareto_optimal", [])
    if pareto_optimal:
        logger.info(f"  Pareto optimal points: {pareto_optimal}")


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
    3. Run :meth:`SweepAnalyzer.compute` over the grouped stats (with SLA
       filters when ``plan.sla_filters`` is non-empty).
    4. Write ``profile_export_aiperf_sweep.json`` and ``.csv`` via the
       sweep exporters; run the recipe's post-process hook when set.

    Returns the directory written to, or ``None`` if there were no
    results to aggregate (graceful no-op).

    Example:
        >>> # 3 variations × 1 trial → 1 row per variation in the CSV
        >>> # 3 variations × 3 trials → ConfidenceAggregation across each cell
        >>> await aggregate_sweep_and_export(results, plan, base_dir, logger)  # doctest: +SKIP
    """
    import asyncio

    from aiperf._cli_runner_post_process import (
        export_sweep_aggregate,
        run_post_process_hook,
    )
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

    sweep_dict = SweepAnalyzer.compute(
        per_combination_stats,
        sweep_parameters,
        sla_filters=list(plan.sla_filters) if plan.sla_filters else None,
    )

    aggregate_dir = base_dir / "sweep_aggregate"
    await asyncio.to_thread(aggregate_dir.mkdir, parents=True, exist_ok=True)

    aggregate_result = _build_sweep_aggregate_result(results, sweep_dict)
    await export_sweep_aggregate(aggregate_result, aggregate_dir, logger)
    _log_sweep_summary(aggregate_result, logger)

    if plan.post_process is not None:
        await asyncio.to_thread(
            run_post_process_hook,
            plan.post_process,
            sweep_dict,
            aggregate_dir,
            logger,
        )

    return aggregate_dir
