# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep-aggregate helpers for ``aiperf.cli_runner``.

Cross-variation aggregation: takes the per-variation RunResults from the
full sweep and emits a single sweep-wide JSON (with pareto-optimal cells
when a recipe declares ``pareto_axes``) plus per-variation confidence
aggregates.

The two public entry points are :func:`aggregate_per_variation_and_export`
and :func:`aggregate_sweep_and_export`; the confidence aggregation for a
single configuration (multi-trial) lives in
:mod:`aiperf.cli_runner._aggregate`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.cli_runner._pareto import _resolve_pareto_axes
from aiperf.orchestrator.models import VariationKey, _variation_key

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkPlan
    from aiperf.orchestrator.models import RunResult


def _resolve_model_name_for_variation(
    plan: BenchmarkPlan, key: VariationKey
) -> str | None:
    """Resolve the first model name for the variation identified by ``key``.

    Matches ``plan.variations[i].label`` against ``_key_label(key)`` and
    returns ``plan.configs[i].models.items[0].name``. Falls back to
    ``configs[0]`` when no variation matches (non-sweep plans or label
    mismatch), and returns ``None`` if the resolved config has no model
    items.

    The aggregate exporter stamps this onto ``metadata["model"]`` so the
    plot loader can recover the model name for aggregate-only runs
    (``profile_export_aiperf_aggregate.json`` carries no
    ``input_config`` block).
    """
    if not plan.configs:
        return None

    target_label = _key_label(key)
    config = plan.configs[0]
    for variation in plan.variations:
        if variation.label == target_label and 0 <= variation.index < len(plan.configs):
            config = plan.configs[variation.index]
            break

    items = getattr(getattr(config, "models", None), "items", None) or []
    if items and getattr(items[0], "name", None):
        return items[0].name
    return None


def _plan_iteration_order(plan: BenchmarkPlan) -> Any:
    """Resolve the sweep iteration order, defaulting to REPEATED outside grids."""
    from aiperf.common.enums import SweepMode
    from aiperf.config.sweep import _GridSweepBase

    if isinstance(plan.sweep, _GridSweepBase):
        return plan.sweep.iteration_order
    return SweepMode.REPEATED


def _plan_sla_filters(plan: BenchmarkPlan) -> list[Any]:
    """Resolve SLA filters from plan.sweep, empty list when no sweep."""
    if plan.sweep is None:
        return []
    return list(plan.sweep.sla_filters)


def _plan_post_process(plan: BenchmarkPlan) -> Any:
    """Resolve post-process spec from plan.sweep, None when no sweep."""
    if plan.sweep is None:
        return None
    return plan.sweep.post_process


def _key_values(key: VariationKey) -> tuple[tuple[str, Any], ...]:
    """Return the values-tuple half of a :data:`VariationKey`."""
    return key[1]


def _key_label(key: VariationKey) -> str:
    """Return the label half of a :data:`VariationKey`."""
    return key[0]


def _variation_dir_name(
    key: VariationKey, variation_label: str, group: list[RunResult]
) -> str:
    """Per-variation directory name, readable even for nested overrides.

    Scenario sweeps without an explicit ``values:`` block carry nested
    override dicts in ``variation_values``; ``_hashable_value`` serializes
    those to long JSON strings, and :func:`_format_dir_name` would turn
    that into an unreadable on-disk path (e.g.
    ``benchmark_{datasets[{namedefault,prompts{isl{mean1000}}}]}``). When
    any variation value is non-scalar, fall back to the human-authored
    ``variation_label`` (e.g. ``aa-1k``), which is the natural cell
    identity. Scalar sweeps keep the ``{leaf}_{value}`` form
    (e.g. ``concurrency_10``).

    Example:
        >>> # nested override -> label
        >>> # scalar override  -> "concurrency_10"
    """
    from aiperf.config.sweep import _format_dir_name

    has_nested = any(
        isinstance(value, (dict, list))
        for result in group
        for value in result.variation_values.values()
    )
    if has_nested:
        return variation_label
    return _format_dir_name(dict(_key_values(key))) or variation_label


def _group_results_by_variation(
    results: list[RunResult],
) -> dict[VariationKey, list[RunResult]]:
    """Group ``results`` by ``(variation_label, sorted_values)``.

    Insertion order of the returned dict mirrors the first-seen order of
    each unique variation key; this becomes the row order of the sweep
    CSV (callers rely on it for stable diffing across reruns).

    Critically: keys distinguish cells by ``variation_label``, not just
    by ``variation_values``. QMC samplers can sample two distinct rows
    that round to the same integer dict; those are independent cells
    that must each get their own aggregate dir, not be pooled.

    Example:
        >>> # 3 results: 2 trials at sobol_0001 (concurrency=3) + 1 at sobol_0006 (also concurrency=3)
        >>> # → groups[("sobol_0001", (('concurrency', 3),))] has length 2
        >>> # → groups[("sobol_0006", (('concurrency', 3),))] has length 1
    """
    groups: dict[VariationKey, list[RunResult]] = {}
    for result in results:
        key = _variation_key(result.variation_label, result.variation_values)
        groups.setdefault(key, []).append(result)
    return groups


def _short_param_name(dotted: str) -> str:
    """Strip dotted-path prefix to expose just the leaf parameter name.

    Sweep parameter names internally are dotted paths like
    ``phases.profiling.concurrency``. The user-facing
    ``per_combination_metrics`` / ``best_configurations`` blocks use the
    leaf name (``concurrency``) so consumers can index by the same short
    keys they used on the CLI (``--concurrency 10``).
    """
    return dotted.rsplit(".", 1)[-1]


def _parameter_display_names(
    groups: dict[VariationKey, list[RunResult]],
) -> dict[str, str]:
    names: list[str] = []
    for key in groups:
        for name, _ in _key_values(key):
            if name not in names:
                names.append(name)
    leaf_counts: dict[str, int] = {}
    for name in names:
        leaf = _short_param_name(name)
        leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
    return {
        name: _short_param_name(name)
        if leaf_counts[_short_param_name(name)] == 1
        else name
        for name in names
    }


def _short_values_dict(
    values: tuple[tuple[str, Any], ...], display_names: dict[str, str]
) -> dict[str, Any]:
    """Map dotted-path keys to collision-safe display parameter names."""
    return {display_names[name]: value for name, value in values}


def _compute_sweep_parameters(
    groups: dict[VariationKey, list[RunResult]],
) -> list[dict[str, Any]]:
    """Derive ``[{"name": ..., "values": [...]}, ...]`` from grouped results.

    The values list preserves first-seen order across the variation
    keys, matching what the user's sweep config produced upstream.
    Parameter names are projected to their leaf form
    (``phases.profiling.concurrency`` -> ``concurrency``) so downstream
    consumers can index by the same short keys they used on the CLI.

    Example:
        >>> # groups keys: [(('concurrency', 10),), (('concurrency', 20),)]
        >>> _compute_sweep_parameters(groups)  # doctest: +SKIP
        [{'name': 'concurrency', 'values': [10, 20]}]
    """
    display_names = _parameter_display_names(groups)
    seen: dict[str, list[Any]] = {}
    for key in groups:
        for name, value in _key_values(key):
            display = display_names[name]
            bucket = seen.setdefault(display, [])
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
    (a single sample has no spread).

    Includes percentile fields (p1..p99) and the canonical ``avg`` alias when
    present on the source metric so downstream consumers (per-cell observers,
    recipe post-processes) can request stats other than ``mean``. The recipe
    asks for ``stat="p95"``; this projection makes that lookup work without
    flat-key gymnastics in the consumer.
    """
    avg = metric.avg if metric.avg is not None else 0.0
    out: dict[str, Any] = {
        "mean": avg,
        "avg": avg,
        "std": 0.0,
        "min": metric.min if metric.min is not None else avg,
        "max": metric.max if metric.max is not None else avg,
        "cv": 0.0,
        "ci_low": avg,
        "ci_high": avg,
        "unit": metric.unit,
    }
    # Carry through every percentile field that's set on the source.
    for pct_field in ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"):
        v = getattr(metric, pct_field, None)
        if v is not None:
            out[pct_field] = v
    # Carry through schema-1.1 size fields when present so downstream
    # readers see the same shape as profile_export_aiperf.json.
    for size_field in ("count", "sum"):
        v = getattr(metric, size_field, None)
        if v is not None:
            out[size_field] = v
    return out


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

    display_names = _parameter_display_names(groups)
    per_combination_stats: dict[Any, dict[str, Any]] = {}
    for key, group in groups.items():
        stats = _aggregate_group_to_stats(group, confidence_level)
        if stats is None:
            continue
        combo = ParameterCombination(
            _short_values_dict(_key_values(key), display_names)
        )
        per_combination_stats[combo] = stats
    return per_combination_stats


def _per_variation_aggregate_dir(
    base_dir: Path,
    variation_dir_name: str,
    sweep_mode: Any,
) -> Path:
    """Resolve the per-variation confidence-aggregate directory.

    Layout, keyed by mode:

    - ``SweepMode.REPEATED``  -> ``<base>/aggregate/<variation_dir_name>/``
    - ``SweepMode.INDEPENDENT`` (default fallback) -> ``<base>/<variation_dir_name>/aggregate/``

    ``variation_dir_name`` is the ``{last_seg}_{value}`` form
    (e.g. ``concurrency_10``) produced by
    :attr:`aiperf.config.sweep.SweepVariation.dir_name`. It is NOT the
    dotted-path variation label; downstream consumers (plotters,
    dashboards) depend on this form.

    Example:
        >>> from aiperf.common.enums import SweepMode
        >>> _per_variation_aggregate_dir(
        ...     Path("/tmp/x"),
        ...     "concurrency_10",
        ...     SweepMode.REPEATED,
        ... )  # doctest: +SKIP
        PosixPath('/tmp/x/aggregate/concurrency_10')
    """
    from aiperf.common.enums import SweepMode

    base_dir = Path(base_dir)
    if sweep_mode == SweepMode.REPEATED:
        return base_dir / "aggregate" / variation_dir_name
    return base_dir / variation_dir_name / "aggregate"


def _sweep_aggregate_dir(base_dir: Path, plan: BenchmarkPlan) -> Path:
    """Resolve the sweep-level aggregate directory.

    Aggregate locations:

    - sweep + multi-run REPEATED -> ``<base>/aggregate/sweep_aggregate/``
    - everything else (sweep-only, sweep + INDEPENDENT) ->
      ``<base>/sweep_aggregate/``

    Multi-run REPEATED interleaves trials across variations, so the
    cross-variation summary lives under the same ``aggregate/`` umbrella
    as the per-variation aggregates. The other modes complete each
    variation cell before moving on, so the sweep summary sits at the
    top of the artifact tree.
    """
    from aiperf.common.enums import SweepMode

    base_dir = Path(base_dir)
    if plan.trials > 1 and _plan_iteration_order(plan) == SweepMode.REPEATED:
        return base_dir / "aggregate" / "sweep_aggregate"
    return base_dir / "sweep_aggregate"


async def _export_one_variation_aggregate(
    *,
    group: list[RunResult],
    key: VariationKey,
    plan: BenchmarkPlan,
    base_dir: Path,
    logger: AIPerfLogger,
) -> Path | None:
    """Aggregate one variation's runs and write the per-cell JSON+CSV pair.

    Returns the directory written, or ``None`` when the cell was skipped
    (insufficient successful runs or aggregator rejected the group).
    """
    import asyncio

    from aiperf.exporters.aggregate import (
        AggregateConfidenceCsvExporter,
        AggregateConfidenceJsonExporter,
        AggregateExporterConfig,
    )
    from aiperf.orchestrator.aggregation.confidence import ConfidenceAggregation

    successful = [r for r in group if r.success]
    # Prefer the key's label (it's the cell-identity half of VariationKey
    # and is guaranteed unique across QMC cells); fall back to any
    # stamped result label, then to a reconstructed key=value form.
    variation_label = _key_label(key) or next(
        (r.variation_label for r in group if r.variation_label),
        ",".join(f"{k}={v}" for k, v in _key_values(key)),
    )

    if not successful:
        logger.warning(
            f"Skipping per-variation aggregate for {variation_label!r}: "
            f"0 successful runs."
        )
        return None

    # Aligned with sweep-aggregate gating: ConfidenceAggregation has a
    # documented single-run degraded mode (std=0, CI collapsed to mean,
    # ``single_run: True`` in metadata). Letting it through here keeps
    # per-variation drill-downs in lockstep with the sweep summary row,
    # which already accepts single-success cells via the same path.
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
    aggregate_result.metadata["variation_values"] = dict(_key_values(key))
    aggregate_result.metadata["sweep_mode"] = str(_plan_iteration_order(plan))
    if model := _resolve_model_name_for_variation(plan, key):
        aggregate_result.metadata["model"] = model

    variation_dir_name = _variation_dir_name(key, variation_label, group)
    aggregate_dir = _per_variation_aggregate_dir(
        base_dir, variation_dir_name, _plan_iteration_order(plan)
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
    ``aiperf.cli_runner._aggregate``: groups ``results`` by ``variation_values``
    and writes one ``profile_export_aiperf_aggregate.{json,csv}`` pair
    per variation that has >=1 successful run. Single-success cells use
    ``ConfidenceAggregation``'s degraded mode (std=0, CI collapsed to
    mean, ``single_run: True`` in metadata) -- see the comment at
    :func:`_export_one_variation_aggregate`. Variations with zero
    successful runs are skipped with a warning; the per-cell run
    artifacts are still on disk, and the sweep aggregate runs
    independently downstream.

    The output path is computed by
    :func:`_per_variation_aggregate_dir`, branching on
    the sweep's ``iteration_order``.

    Returns the list of directories written (in group-iteration order),
    so the caller can log them. Empty list when every variation had
    zero successful runs.

    Example:
        >>> # 2 variations x 3 trials, mode=independent
        >>> # writes:
        >>> #   <base>/concurrency_10/aggregate/profile_export_aiperf_aggregate.json
        >>> #   <base>/concurrency_10/aggregate/profile_export_aiperf_aggregate.csv
        >>> #   <base>/concurrency_20/aggregate/profile_export_aiperf_aggregate.json
        >>> #   <base>/concurrency_20/aggregate/profile_export_aiperf_aggregate.csv
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

    Stuffs the sweep sections (``best_configurations``, ``pareto_optimal``)
    into ``metadata`` and the per-cell rows into ``metrics``, so the
    exporters share their constructor with the sibling confidence
    exporters.
    """
    from aiperf.orchestrator.aggregation.base import AggregateResult

    failed_runs = [
        {"label": r.label, "error": r.error} for r in results if not r.success
    ]
    sweep_metadata = dict(sweep_dict.get("metadata", {}))
    # ``aggregation_type`` is duplicated into the
    # ``metadata`` block so consumers that key off
    # ``output["metadata"]["aggregation_type"]`` work without first
    # checking the top-level key. The base exporter still emits the
    # top-level field too.
    sweep_metadata["aggregation_type"] = "sweep"
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


def _log_sweep_summary(
    aggregate_result: Any,
    logger: AIPerfLogger,
) -> None:
    """Stdout summary of best configurations and Pareto frontier."""
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
       filters when ``plan.sweep.sla_filters`` is non-empty).
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

    from aiperf.cli_runner._post_process import (
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
        sla_filters=list(_plan_sla_filters(plan)) if _plan_sla_filters(plan) else None,
    )
    # Stamp run-shape metadata onto the sweep block so downstream
    # consumers can re-derive iteration-order and per-cell trial counts
    # from the sweep JSON without re-reading the plan.
    sweep_dict.setdefault("metadata", {})
    sweep_dict["metadata"]["sweep_mode"] = str(_plan_iteration_order(plan))
    sweep_dict["metadata"]["confidence_level"] = plan.confidence_level
    sweep_dict["metadata"]["num_trials_per_value"] = max(
        (len(g) for g in groups.values()), default=0
    )
    axes = _resolve_pareto_axes(plan)
    if axes is not None:
        sweep_dict["metadata"]["pareto_axes"] = axes.model_dump()

    aggregate_dir = _sweep_aggregate_dir(base_dir, plan)
    await asyncio.to_thread(aggregate_dir.mkdir, parents=True, exist_ok=True)

    aggregate_result = _build_sweep_aggregate_result(results, sweep_dict)
    await export_sweep_aggregate(aggregate_result, aggregate_dir, logger)
    _log_sweep_summary(aggregate_result, logger)

    post_process = _plan_post_process(plan)
    if post_process is not None:
        await asyncio.to_thread(
            run_post_process_hook,
            post_process,
            sweep_dict,
            aggregate_dir,
            logger,
        )

    return aggregate_dir
