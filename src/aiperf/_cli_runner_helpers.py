# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for ``aiperf.cli_runner``.

Split out purely to keep ``cli_runner.py`` below the file/function size
ergonomics limits; the helpers here are not part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkPlan
    from aiperf.orchestrator.aggregation.base import AggregateResult
    from aiperf.orchestrator.strategies import ExecutionStrategy


def validate_convergence_config(plan: BenchmarkPlan) -> None:
    """Raise ValueError for invalid adaptive/convergence plan configurations."""
    from aiperf.common.enums import ConvergenceMode, ExportLevel

    if not plan.use_adaptive:
        return
    if plan.trials <= 1:
        raise ValueError(
            "--convergence-metric requires --num-profile-runs > 1. "
            "Set --num-profile-runs to at least 2 to enable adaptive convergence."
        )
    if (
        plan.convergence_mode == ConvergenceMode.DISTRIBUTION
        and plan.export_level == ExportLevel.SUMMARY
    ):
        raise ValueError(
            "--convergence-mode distribution requires per-request JSONL data, "
            "but --export-level is set to 'summary'. "
            "Use --export-level records or --export-level raw."
        )


def log_multi_run_banner(
    plan: BenchmarkPlan, total_runs: int, logger: AIPerfLogger
) -> None:
    """Emit the banner describing a multi-run benchmark's configuration."""
    from aiperf.common.enums import ConvergenceMode

    logger.info("=" * 80)
    logger.info("Starting Multi-Run Benchmark")
    logger.info(f"  Configurations: {len(plan.configs)}")
    logger.info(f"  Trials per config: {plan.trials}")
    logger.info(f"  Total runs: {total_runs}")
    logger.info(f"  Confidence level: {plan.confidence_level:.0%}")
    logger.info(f"  Cooldown between runs: {plan.cooldown_seconds}s")
    if plan.use_adaptive:
        logger.info(f"  Convergence mode: {plan.convergence_mode}")
        logger.info(f"  Convergence metric: {plan.convergence_metric}")
        logger.info(f"  Convergence threshold: {plan.convergence_threshold}")
        if plan.convergence_mode == ConvergenceMode.DISTRIBUTION:
            logger.info(
                "  Note: distribution mode converges when KS p-value > threshold "
                "(higher threshold = stricter, opposite of ci_width/cv)"
            )
    logger.info("=" * 80)


def build_strategy(plan: BenchmarkPlan, logger: AIPerfLogger) -> ExecutionStrategy:
    """Construct the per-trial execution strategy (adaptive or fixed)."""
    from aiperf.orchestrator.strategies import FixedTrialsStrategy

    if not plan.use_adaptive:
        return FixedTrialsStrategy(
            num_trials=plan.trials,
            cooldown_seconds=plan.cooldown_seconds,
            auto_set_seed=plan.set_consistent_seed,
            disable_warmup_after_first=plan.disable_warmup_after_first,
        )

    from aiperf.orchestrator.strategies import AdaptiveStrategy

    criterion = _build_convergence_criterion(plan)

    effective_min_runs = min(3, plan.trials)
    if effective_min_runs < 3:
        logger.warning(
            f"--num-profile-runs={plan.trials} is below the recommended minimum of 3. "
            "Convergence checks will have reduced statistical power."
        )

    return AdaptiveStrategy(
        criterion=criterion,
        min_runs=effective_min_runs,
        max_runs=plan.trials,
        cooldown_seconds=plan.cooldown_seconds,
        auto_set_seed=plan.set_consistent_seed,
        disable_warmup_after_first=plan.disable_warmup_after_first,
    )


def _build_convergence_criterion(plan: BenchmarkPlan):  # noqa: ANN202
    """Pick the convergence criterion matching ``plan.convergence_mode``."""
    from aiperf.common.enums import ConvergenceMode
    from aiperf.orchestrator.convergence import (
        CIWidthConvergence,
        CVConvergence,
        DistributionConvergence,
    )

    mode = plan.convergence_mode
    threshold = plan.convergence_threshold

    if mode == ConvergenceMode.CI_WIDTH:
        return CIWidthConvergence(
            metric=plan.convergence_metric,
            stat=plan.convergence_stat,
            threshold=threshold,
            confidence_level=plan.confidence_level,
        )
    if mode == ConvergenceMode.CV:
        return CVConvergence(
            metric=plan.convergence_metric,
            threshold=threshold,
            stat=plan.convergence_stat,
        )
    return DistributionConvergence(
        metric=plan.convergence_metric,
        p_value_threshold=threshold,
        jsonl_filename=plan.export_jsonl_file or "",
    )


def aggregate_and_export(
    results: list,
    plan: BenchmarkPlan,
    *,
    strategy: ExecutionStrategy,
    base_dir: Path,
    logger: AIPerfLogger,
) -> None:
    """Aggregate ``results`` and write JSON/CSV/detailed artifacts."""
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

    aggregate_dir = strategy.get_aggregate_path(base_dir)

    exporter_config = AggregateExporterConfig(
        result=aggregate_result,
        output_dir=aggregate_dir,
    )

    detailed_result = _maybe_compute_detailed(plan, results)

    async def export_artifacts():
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

        return await asyncio.gather(*tasks)

    export_paths = asyncio.run(export_artifacts())

    logger.info(f"Aggregate JSON written to: {export_paths[0]}")
    logger.info(f"Aggregate CSV written to: {export_paths[1]}")
    if plan.use_adaptive and len(export_paths) > 2:
        logger.info(f"Collated aggregate JSON written to: {export_paths[2]}")

    print_aggregate_summary(aggregate_result, logger)


def _maybe_compute_detailed(plan: BenchmarkPlan, results: list):  # noqa: ANN202
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
    metric, display_name: str, confidence_level: float, logger: AIPerfLogger
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
