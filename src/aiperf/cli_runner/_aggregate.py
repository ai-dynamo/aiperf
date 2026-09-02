# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-cell confidence aggregation + console summary for cli_runner.

:func:`aggregate_and_export` is the multi-run entry point for combining
the per-trial RunResults of a *single configuration* into a confidence
aggregate (mean/std/CI per metric) and writing the JSON/CSV/detailed
exports. The sweep-wide aggregation (across variations) lives in the
sibling :mod:`aiperf.cli_runner._sweep_aggregate`.

:func:`print_aggregate_summary` writes the human-readable summary block
that ends every multi-run benchmark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkPlan
    from aiperf.orchestrator.aggregation.base import AggregateResult
    from aiperf.orchestrator.strategies import ExecutionStrategy


async def aggregate_and_export(
    results: list,
    plan: BenchmarkPlan,
    *,
    strategy: ExecutionStrategy,
    base_dir: Path,
    logger: AIPerfLogger,
) -> None:
    """Aggregate ``results`` and write JSON/CSV/detailed artifacts.

    Async to allow ``asyncio.gather`` on the exporter coroutines (JSON,
    CSV, and detailed-JSON exports run concurrently). Callers reach
    this via ``asyncio.run(...)`` from ``cli_runner``.
    """
    import asyncio

    from aiperf.exporters.aggregate import (
        AggregateConfidenceCsvExporter,
        AggregateConfidenceJsonExporter,
        AggregateDetailedJsonExporter,
        AggregateExporterConfig,
    )
    from aiperf.orchestrator.aggregation.confidence import ConfidenceAggregation

    aggregation = ConfidenceAggregation(confidence_level=plan.confidence_level)
    aggregate_result = aggregation.aggregate(results)
    aggregate_result.metadata["cooldown_seconds"] = plan.cooldown_seconds
    _stamp_scenario_submission_metadata(aggregate_result, results, plan)

    aggregate_dir = strategy.get_aggregate_path(base_dir)

    exporter_config = AggregateExporterConfig(
        result=aggregate_result,
        output_dir=aggregate_dir,
    )

    detailed_result = _maybe_compute_detailed(plan, results)

    await asyncio.to_thread(aggregate_dir.mkdir, parents=True, exist_ok=True)
    json_exporter = AggregateConfidenceJsonExporter(exporter_config)
    csv_exporter = AggregateConfidenceCsvExporter(exporter_config)

    tasks = [json_exporter.export(), csv_exporter.export()]

    if detailed_result is not None:
        detailed_config = AggregateExporterConfig(
            result=detailed_result,
            output_dir=aggregate_dir,
        )
        detailed_exporter = AggregateDetailedJsonExporter(detailed_config)
        tasks.append(detailed_exporter.export())

    export_paths = await asyncio.gather(*tasks)

    logger.info(f"Aggregate JSON written to: {export_paths[0]}")
    logger.info(f"Aggregate CSV written to: {export_paths[1]}")
    if plan.use_adaptive and len(export_paths) > 2:
        logger.info(f"Collated aggregate JSON written to: {export_paths[2]}")

    print_aggregate_summary(aggregate_result, logger)


def _sum_runtime_response_counts(results: list) -> tuple[int, int]:
    """Sum (total_responses, context_overflow_count) across successful runs.

    Reads per-run averaged summary metrics. Total responses is
    ``request_count + error_request_count + skipped_context_overflow_count``:
    metric-path overflows are already inside ``error_request_count``
    (``ContextOverflowCountMetric`` is ERROR_ONLY), while AGENTIC_REPLAY
    skip-path overflows are only in the ``skipped_context_overflow_count``
    side channel. The overflow numerator is the (possibly merged)
    ``context_overflow_count`` metric. Missing metrics contribute 0.
    """
    from aiperf.common.finite import is_finite_value

    def _avg(run, tag: str) -> int:
        m = run.summary_metrics.get(tag)
        if m is None or not is_finite_value(m.avg):
            return 0
        return int(m.avg)

    total_responses = 0
    context_overflow_count = 0
    for run in results:
        if not run.success:
            continue
        overflow = _avg(run, "context_overflow_count")
        skipped = _avg(run, "skipped_context_overflow_count")
        context_overflow_count += overflow
        total_responses += (
            _avg(run, "request_count") + _avg(run, "error_request_count") + skipped
        )
    return total_responses, context_overflow_count


def _stamp_scenario_submission_metadata(
    aggregate: AggregateResult,
    results: list,
    plan: BenchmarkPlan,
) -> None:
    """Inject scenario-submission carrier keys onto ``aggregate.metadata``.

    The ``AggregateConfidenceJsonExporter`` pops these underscore-prefixed keys
    to compute ``submission_valid`` / ``submission_invalid_reasons`` for the
    aggregate JSON. Dataset provenance is stamped for every public-dataset run;
    the scenario carrier keys are no-op when the base config sets no ``--scenario``.

    The static validator outcome lives on ``run.resolved.scenario_outcome``
    (set by ScenarioResolver). The per-run scenario_outcome is computed in each
    benchmark subprocess and is not returned through ``RunResult``; the
    invariant lock is deterministic from config, so we re-resolve it here off
    the base config rather than threading a new return channel. Runtime totals
    are summed from the per-run summary metrics.
    """
    base_config = plan.configs[0]

    from aiperf.dataset.provenance import public_dataset_provenance

    dataset = public_dataset_provenance(base_config)
    if dataset is not None:
        aggregate.metadata["dataset"] = dataset

    scenario_name = getattr(base_config, "scenario", None)
    if scenario_name is None:
        return

    outcome = None
    reresolve_failed = False
    try:
        from aiperf.cli_runner import _make_benchmark_run
        from aiperf.common.scenario import apply_scenario

        run = _make_benchmark_run(base_config)
        apply_scenario(run)
        outcome = getattr(run.resolved, "scenario_outcome", None)
    except Exception as exc:
        # Fail closed: do not stamp submission_valid=True when the invariant
        # lock could not be re-checked. Aggregation/export must still proceed.
        from aiperf.common.aiperf_logger import AIPerfLogger

        AIPerfLogger(__name__).warning(
            f"Scenario re-resolution failed during aggregate stamp; "
            f"marking submission_valid=False: {exc}"
        )
        reresolve_failed = True

    if reresolve_failed:
        submission_valid = False
        submission_invalid_reasons: list[str] = ["scenario_reresolve_failed"]
    elif outcome is None:
        submission_valid = True
        submission_invalid_reasons = []
    else:
        submission_valid = bool(getattr(outcome, "submission_valid", True))
        submission_invalid_reasons = list(
            getattr(outcome, "submission_invalid_reasons", []) or []
        )

    # Scenario + sweep only reaches aggregation under --unsafe-override: the
    # envelope validator (_reject_scenario_with_sweep) hard-errors otherwise.
    # apply_scenario above re-resolves a SINGLE expanded variation and cannot
    # see the sweep, so it returns a clean per-variation outcome. Carry the
    # envelope-level override violation forward here so a scenario sweep is
    # never stamped as a valid submission. (scenario_name is non-None here.)
    if getattr(plan, "is_sweep", False):
        submission_valid = False
        if "scenario_with_sweep" not in submission_invalid_reasons:
            submission_invalid_reasons.append("scenario_with_sweep")

    total_responses, context_overflow_count = _sum_runtime_response_counts(results)

    aggregate.metadata["_scenario_name"] = scenario_name
    aggregate.metadata["_validator_submission_valid"] = submission_valid
    aggregate.metadata["_validator_submission_invalid_reasons"] = (
        submission_invalid_reasons
    )
    aggregate.metadata["_total_responses"] = total_responses
    aggregate.metadata["_context_overflow_count"] = context_overflow_count
    aggregate.metadata["_was_cancelled"] = any(
        getattr(r, "was_cancelled", False) for r in results
    )


def _maybe_compute_detailed(
    plan: BenchmarkPlan, results: list
) -> AggregateResult | None:
    """Return detailed aggregation result when adaptive mode is enabled."""
    if not plan.use_adaptive:
        return None

    from aiperf.orchestrator.aggregation.detailed import DetailedAggregation
    from aiperf.orchestrator.jsonl_loader import DEFAULT_JSONL_FILENAME

    detailed_aggregation = DetailedAggregation(
        jsonl_filename=plan.export_jsonl_file or DEFAULT_JSONL_FILENAME,
    )
    detailed_result = detailed_aggregation.aggregate(results)
    detailed_result.metadata["cooldown_seconds"] = plan.cooldown_seconds
    return detailed_result


_PRIORITY_METRICS = (
    "request_throughput",
    "time_to_first_token",
    "inter_token_latency",
    "request_latency",
)
_PRIORITY_STAT_SUFFIXES = ("_avg", "_p99", "_max", "_p50")


def _collect_priority_metrics(
    aggregate_result: AggregateResult,
) -> list[tuple[str, str]]:
    """Return (metric_key, display_name) pairs for available priority metrics."""
    selected: list[tuple[str, str]] = []
    for base_metric in _PRIORITY_METRICS:
        for suffix in _PRIORITY_STAT_SUFFIXES:
            metric_key = f"{base_metric}{suffix}"
            if metric_key not in aggregate_result.metrics:
                continue
            display_name = base_metric.replace("_", " ").title()
            stat_name = suffix[1:].upper()
            if stat_name == "AVG":
                stat_name = "Avg"
            elif stat_name.startswith("P"):
                stat_name = f"P{stat_name[1:]}"
            else:
                stat_name = stat_name.capitalize()
            selected.append((metric_key, f"{display_name} ({stat_name})"))
            break
    return selected


def _print_metric_block(
    metric: Any, display_name: str, confidence_level: float, logger: AIPerfLogger
) -> None:
    """Log mean/std/min/max/cv/CI lines for a single metric."""
    logger.info(f"\n{display_name}:")
    logger.info(f"  Mean:    {metric.mean:>12.4f} {metric.unit}")
    logger.info(f"  Std Dev: {metric.std:>12.4f} {metric.unit}")
    logger.info(f"  Min:     {metric.min:>12.4f} {metric.unit}")
    logger.info(f"  Max:     {metric.max:>12.4f} {metric.unit}")
    logger.info(f"  CV:      {metric.cv:>12.2%}")
    logger.info(
        f"  {confidence_level:.0%} CI: [{metric.ci_low:.4f}, {metric.ci_high:.4f}] {metric.unit}"
    )


def _print_interpretation_guide(confidence_level: float, logger: AIPerfLogger) -> None:
    """Log the CV / CI interpretation footer block."""
    logger.info("")
    logger.info("-" * 80)
    logger.info("Coefficient of Variation (CV) Interpretation Guide:")
    logger.info("  CV < 5%:   Excellent repeatability (low variance)")
    logger.info("  CV 5-10%:  Good repeatability (moderate variance)")
    logger.info("  CV 10-20%: Fair repeatability (consider more runs)")
    logger.info("  CV > 20%:  High variance (investigate or increase runs)")
    logger.info("")
    logger.info("Confidence Interval (CI) Interpretation:")
    logger.info(
        f"  The {confidence_level:.0%} CI indicates the range where the true mean"
    )
    logger.info(f"  is likely to fall with {confidence_level:.0%} confidence.")
    logger.info("  Narrower intervals indicate more precise estimates.")
    logger.info("=" * 80)


def print_aggregate_summary(
    aggregate_result: AggregateResult, logger: AIPerfLogger
) -> None:
    """Print a comprehensive summary of aggregate statistics to console."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("AGGREGATE STATISTICS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Aggregation Type: {aggregate_result.aggregation_type}")
    logger.info(f"Total Runs: {aggregate_result.num_runs}")
    logger.info(f"Successful Runs: {aggregate_result.num_successful_runs}")

    if aggregate_result.failed_runs:
        logger.warning(f"Failed Runs ({len(aggregate_result.failed_runs)}):")
        for failed in aggregate_result.failed_runs:
            logger.warning(f"  - {failed['label']}: {failed['error']}")

    confidence_level = aggregate_result.metadata.get("confidence_level", 0.95)
    logger.info(f"Confidence Level: {confidence_level:.0%}")

    logger.info("")
    logger.info("Key Metrics:")
    logger.info("-" * 80)

    metrics_to_display = _collect_priority_metrics(aggregate_result)
    for metric_key, display_name in metrics_to_display:
        _print_metric_block(
            aggregate_result.metrics[metric_key],
            display_name,
            confidence_level,
            logger,
        )

    if not metrics_to_display:
        logger.warning("No key metrics found in aggregate results")

    _print_interpretation_guide(confidence_level, logger)
