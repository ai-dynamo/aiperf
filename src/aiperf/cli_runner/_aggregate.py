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

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkPlan
    from aiperf.orchestrator.aggregation.base import AggregateResult
    from aiperf.orchestrator.models import RunResult
    from aiperf.orchestrator.strategies import ExecutionStrategy


SCENARIO_SUBMISSION_CARRIER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "_scenario_name",
        "_validator_submission_valid",
        "_validator_submission_invalid_reasons",
        "_total_responses",
        "_context_overflow_count",
        "_was_cancelled",
    }
)
"""The underscore-prefixed carrier keys ``_stamp_scenario_submission_metadata``
writes onto ``AggregateResult.metadata``.

Single source of truth shared by the writer (this module) and every aggregate
exporter that must keep the keys out of its public output: the
``AggregateConfidenceJsonExporter`` pops them individually while folding the
submission verdict; the ``AggregateConfidenceCsvExporter`` (which performs no
fold) strips them via :func:`strip_scenario_submission_carrier_keys`.
"""


def strip_scenario_submission_carrier_keys(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``metadata`` without the scenario-submission carrier keys.

    Exporters that do not fold the submission verdict themselves (e.g. the
    aggregate confidence CSV) use this so the internal carrier keys never leak
    into user-facing output.

    Args:
        metadata: The ``AggregateResult.metadata`` dict (not mutated).

    Returns:
        A shallow copy with every :data:`SCENARIO_SUBMISSION_CARRIER_KEYS`
        entry removed.
    """
    return {
        key: value
        for key, value in metadata.items()
        if key not in SCENARIO_SUBMISSION_CARRIER_KEYS
    }


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


def _stamp_scenario_submission_metadata(
    aggregate: AggregateResult,
    results: list[RunResult],
    plan: BenchmarkPlan,
) -> None:
    """Stamp scenario-submission carrier keys onto ``aggregate.metadata``.

    The ``AggregateConfidenceJsonExporter``
    reader (``aggregate_confidence_json_exporter.py`` ``_build_submission_metadata``)
    pops these underscore-prefixed keys to re-derive ``submission_valid`` /
    ``submission_invalid_reasons`` via ``compute_submission_outcome``; without this
    writer the reader sees no ``_scenario_name`` and emits an empty verdict.

    Stamps the carrier keys the reader expects EXACTLY:
    ``_scenario_name`` / ``_validator_submission_valid`` /
    ``_validator_submission_invalid_reasons`` / ``_total_responses`` /
    ``_context_overflow_count`` / ``_was_cancelled`` -- the
    :data:`SCENARIO_SUBMISSION_CARRIER_KEYS` set, which non-folding exporters
    strip via :func:`strip_scenario_submission_carrier_keys`.

    The validator (lock-level) outcome is read from the REAL per-run
    ``ScenarioOutcome`` carried on ``RunResult.submission_valid`` /
    ``RunResult.submission_invalid_reasons`` (stamped by ``apply_scenario`` in
    the parent via ``local_executor._carried_scenario_outcome``). This
    is the lock-only verdict -- ``(True, [])`` for a clean lock (INCLUDING a clean
    config that merely carries ``--unsafe-override`` with no violation), or
    ``(False, [...])`` under ``--unsafe-override`` with violations. It is NOT
    re-derived from the ``unsafe_override`` flag (that flag alone never determines
    validity) and NOT read from the per-run JSON's ``submission_valid`` (already
    runtime-folded). Runtime-only signals (cross-run context-overflow rate and
    cancellation) are folded on top by the reader. No-op when no scenario is set.

    Args:
        aggregate: The confidence ``AggregateResult`` whose ``metadata`` dict is
            mutated in place with the carrier keys.
        results: All per-run ``RunResult`` objects from the orchestrator. Only
            successful runs contribute to the cross-run response sums; the
            cancellation OR folds over ALL runs (a graceful Ctrl+C that completed
            zero requests is ``success=False`` but still wrote ``was_cancelled``).
        plan: The ``BenchmarkPlan`` carrying the static scenario name on its
            first config.
    """
    config = plan.configs[0]
    scenario_name = getattr(config, "scenario", None)
    if scenario_name is None:
        return

    validator_submission_valid, validator_reasons = _carried_validator_outcome(results)

    successful_runs = [r for r in results if r.success]
    total_responses, context_overflow_count = _sum_runtime_response_counts(
        successful_runs
    )
    json_name = config.artifacts.profile_export_json_file.name
    was_cancelled = any(_read_run_was_cancelled(run, json_name) for run in results)

    carriers: dict[str, Any] = {
        "_scenario_name": str(scenario_name),
        "_validator_submission_valid": validator_submission_valid,
        "_validator_submission_invalid_reasons": validator_reasons,
        "_total_responses": total_responses,
        "_context_overflow_count": context_overflow_count,
        "_was_cancelled": was_cancelled,
    }
    if set(carriers) != SCENARIO_SUBMISSION_CARRIER_KEYS:
        raise RuntimeError(
            "scenario-submission carrier keys drifted from "
            "SCENARIO_SUBMISSION_CARRIER_KEYS; update the constant so exporters "
            "keep stripping every stamped key"
        )
    aggregate.metadata.update(carriers)


def _carried_validator_outcome(
    results: list[RunResult],
) -> tuple[bool, list[str]]:
    """Resolve the lock-only validator verdict carried on the per-run results.

    Reads ``RunResult.submission_valid`` / ``RunResult.submission_invalid_reasons``
    -- the real ``ScenarioOutcome`` stamped by ``apply_scenario`` in the parent
    and carried through ``local_executor``. The lock verdict is config-
    deterministic across the trials of a single cell (every run shares the same
    config), so the first run carrying a non-None ``submission_valid`` defines it.
    Falls back to ``(True, [])`` when no run carries a verdict, so the
    reader's runtime fold still applies on a clean lock.

    Args:
        results: All per-run ``RunResult`` objects from the orchestrator.

    Returns:
        A ``(validator_submission_valid, validator_reasons)`` tuple. The bool is
        the lock-only verdict (True for a clean lock, including clean +
        ``--unsafe-override``); the list is the lock-only reason tags.
    """
    for run in results:
        if run.submission_valid is not None:
            return run.submission_valid, list(run.submission_invalid_reasons)
    return True, []


def _sum_runtime_response_counts(
    successful_runs: list[RunResult],
) -> tuple[int, int]:
    """Sum total responses and context-overflow counts across successful runs.

    Each ``RunResult.summary_metrics`` is a tag -> ``JsonMetricResult`` map; the
    per-run total lives on the count metric's ``avg``. Total responses =
    ``request_count`` (valid) + ``error_request_count`` (non-overflow failures)
    + ``context_overflow_count`` (overflow records skipped from normal metrics),
    matching the InferenceX AgentX spec §4.8 / §7 denominator (all responses
    received). Returns ``(0, 0)`` when no successful runs exist.

    Args:
        successful_runs: The successful ``RunResult`` objects to sum over.

    Returns:
        A ``(total_responses, context_overflow_count)`` tuple of non-negative ints.
    """
    total_responses = 0
    context_overflow_count = 0
    for result in successful_runs:
        metrics = result.summary_metrics or {}
        for tag in ("request_count", "error_request_count"):
            metric = metrics.get(tag)
            if metric is not None and metric.avg is not None:
                total_responses += int(metric.avg)
        overflow_metric = metrics.get("context_overflow_count")
        if overflow_metric is not None and overflow_metric.avg is not None:
            overflow_count = int(overflow_metric.avg)
            context_overflow_count += overflow_count
            total_responses += overflow_count
    return total_responses, context_overflow_count


def _read_run_was_cancelled(run: RunResult, json_name: str) -> bool:
    """Read the top-level ``was_cancelled`` flag from a run's profile export JSON.

    A run that exits 0 after a graceful Ctrl+C still writes its export file (with
    partial metrics) and marks it ``was_cancelled: true``; scenario submissions
    must treat such runs as invalid. ``RunResult`` carries no cancellation
    flag, so the signal is read back from the
    per-run JSON at ``run.artifacts_path / json_name``.

    Args:
        run: The ``RunResult`` whose ``artifacts_path`` locates the export dir.
        json_name: The profile-export JSON filename (honors ``--profile-export-prefix``).

    Returns:
        True when the run was cancelled early, False otherwise (including when the
        artifacts path or export file is missing or unreadable).
    """
    import orjson

    artifacts_path = run.artifacts_path
    if artifacts_path is None:
        return False
    json_file = artifacts_path / json_name
    if not json_file.exists():
        return False
    try:
        data = orjson.loads(json_file.read_bytes())
    except (OSError, ValueError, orjson.JSONDecodeError):
        return False
    return bool(data.get("was_cancelled", False))


def _maybe_compute_detailed(
    plan: BenchmarkPlan, results: list
) -> AggregateResult | None:
    """Return detailed aggregation result when adaptive mode is enabled."""
    if not plan.use_adaptive:
        return None

    from aiperf.orchestrator.aggregation.detailed import DetailedAggregation

    detailed_aggregation = DetailedAggregation(
        jsonl_filename=plan.export_jsonl_file or "",
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
