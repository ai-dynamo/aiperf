# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from aiperf.cli_utils import raise_startup_error_and_exit
from aiperf.plugin.enums import ServiceType, UIType

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkConfig, BenchmarkPlan, BenchmarkRun
    from aiperf.orchestrator.aggregation.base import AggregateResult


def run_benchmark(plan: BenchmarkPlan) -> None:
    """Run benchmarks from a BenchmarkPlan.

    For single-config single-trial plans, runs directly (Dashboard works).
    For multi-config or multi-trial plans, uses the MultiRunOrchestrator.
    """
    if plan.use_adaptive and plan.trials <= 1:
        raise ValueError(
            "--convergence-metric requires --num-profile-runs > 1. "
            "Set --num-profile-runs to at least 2 to enable adaptive convergence."
        )

    if plan.is_single_run:
        run = _make_benchmark_run(plan.configs[0])
        _run_single_benchmark(run)
    else:
        _run_multi_benchmark(plan)


def _make_benchmark_run(
    config: BenchmarkConfig,
    *,
    benchmark_id: str | None = None,
    trial: int = 0,
    artifact_dir: Path | None = None,
) -> BenchmarkRun:
    """Wrap a BenchmarkConfig into a BenchmarkRun."""
    from aiperf.config import BenchmarkRun

    return BenchmarkRun(
        benchmark_id=benchmark_id or uuid4().hex[:12],
        cfg=config,
        trial=trial,
        artifact_dir=artifact_dir or config.artifacts.dir,
    )


def _run_single_benchmark(run: BenchmarkRun) -> None:
    """Run a single benchmark."""

    import multiprocessing
    import platform

    from aiperf.common.environment import Environment

    config = run.cfg
    is_macos = platform.system() == "Darwin"
    using_dashboard = config.ui_type == UIType.DASHBOARD

    # NOTE: On macOS, when using the Textual UI with multiprocessing, terminal corruption
    # (ASCII garbage, freezing) can occur when mouse events interfere with child processes.
    # We apply multiple layers of protection:
    # 1. Set spawn method early (before any multiprocessing operations)
    # 2. Create log_queue before any UI initialization
    # 3. Set FD_CLOEXEC on terminal file descriptors
    # 4. Close terminal FDs in child processes (done in bootstrap.py)
    #
    # Env override takes precedence for all platforms.
    if Environment.SERVICE.MULTIPROCESSING_START_METHOD:
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method(
                Environment.SERVICE.MULTIPROCESSING_START_METHOD, force=True
            )
    elif is_macos and using_dashboard:
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method("spawn", force=True)

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.config.resolvers import build_default_resolver_chain

    logger = AIPerfLogger(__name__)

    # Create queues before UI initialization to minimize FD inheritance issues.
    # Error queue is always created; log queue only for Dashboard UI.
    from aiperf.common.error_queue import get_global_error_queue

    get_global_error_queue()

    log_queue = None
    if using_dashboard:
        from aiperf.common.logging import get_global_log_queue

        log_queue = get_global_log_queue()

        if is_macos:
            import fcntl
            import sys

            try:
                for fd in [
                    sys.stdin.fileno(),
                    sys.stdout.fileno(),
                    sys.stderr.fileno(),
                ]:
                    flags = fcntl.fcntl(fd, fcntl.F_GETFD)
                    fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
                logger.debug("Set FD_CLOEXEC on terminal file descriptors for macOS")
            except (OSError, ValueError, AttributeError) as e:
                # Non-fatal if this fails, other layers will protect
                logger.debug(f"Could not set FD_CLOEXEC on terminal descriptors: {e}")
    else:
        from aiperf.common.logging import setup_rich_logging

        setup_rich_logging(config)

    logger.info("Starting AIPerf System")

    try:
        chain = build_default_resolver_chain()
        chain.resolve_all(run)
    except Exception as e:
        logger.exception("Configuration resolution failed")
        raise_startup_error_and_exit(
            f"Configuration resolution failed: {e}",
            title="Configuration Error",
        )

    exit_code = 0
    try:
        bootstrap_and_run_service(
            service_type=ServiceType.SYSTEM_CONTROLLER,
            run=run,
            log_queue=log_queue,
        )
    except SystemExit as e:
        exit_code = int(e.code) if e.code is not None else 0
    except Exception:
        logger.exception("Error running AIPerf System")
        exit_code = 1
    finally:
        logger.debug("AIPerf System exited")

    # Bypass Python's normal teardown: multiprocessing atexit handlers,
    # leftover ZMQ contexts, and daemon threads can otherwise block the
    # interpreter from exiting — which is fatal under pytest-xdist where
    # the parent waits on communicate(). The controller already flushed
    # logs and wrote artifacts; killing the interpreter here is safe.
    import os as _os
    import sys as _sys

    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(exit_code)


def _run_multi_benchmark(plan: BenchmarkPlan) -> None:
    """Run multiple benchmarks from a BenchmarkPlan.

    Executes trials x configs benchmarks, then aggregates results and
    computes confidence statistics. When convergence flags are set, uses
    AdaptiveStrategy for early stopping and runs both ConfidenceAggregation
    and DetailedAggregation.
    """
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.enums import ConvergenceMode, ExportLevel
    from aiperf.common.logging import setup_rich_logging
    from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
    from aiperf.orchestrator.strategies import FixedTrialsStrategy

    first_config = plan.configs[0]

    if first_config.ui_type == UIType.DASHBOARD:
        raise ValueError(
            "Dashboard UI is not supported with sweep/multi-run mode. "
            "Please use '--ui simple' or '--ui none' instead."
        )

    setup_rich_logging(first_config)
    logger = AIPerfLogger(__name__)

    total_runs = len(plan.configs) * plan.trials

    # Validate convergence configuration
    if plan.use_adaptive:
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

    from aiperf.config import BenchmarkRun
    from aiperf.config.resolvers import ArtifactDirResolver, TimingResolver

    probe_run = BenchmarkRun(
        benchmark_id="probe",
        cfg=first_config,
        artifact_dir=first_config.artifacts.dir,
    )
    ArtifactDirResolver().resolve(probe_run)
    TimingResolver().resolve(probe_run)

    base_dir = probe_run.artifact_dir

    per_run_duration = probe_run.resolved.total_expected_duration
    if per_run_duration is not None:
        total_benchmark = per_run_duration * total_runs
        total_with_cooldown = total_benchmark + plan.cooldown_seconds * max(
            total_runs - 1, 0
        )
        logger.info(f"  Estimated duration: {total_with_cooldown:.0f}s")

    # Create strategy
    if plan.use_adaptive:
        from aiperf.orchestrator.convergence import (
            CIWidthConvergence,
            CVConvergence,
            DistributionConvergence,
        )
        from aiperf.orchestrator.strategies import AdaptiveStrategy

        mode = plan.convergence_mode
        threshold = plan.convergence_threshold

        if mode == ConvergenceMode.CI_WIDTH:
            criterion = CIWidthConvergence(
                metric=plan.convergence_metric,
                stat=plan.convergence_stat,
                threshold=threshold,
                confidence_level=plan.confidence_level,
            )
        elif mode == ConvergenceMode.CV:
            criterion = CVConvergence(
                metric=plan.convergence_metric,
                threshold=threshold,
                stat=plan.convergence_stat,
            )
        else:
            criterion = DistributionConvergence(
                metric=plan.convergence_metric,
                p_value_threshold=threshold,
                jsonl_filename=plan.export_jsonl_file or "",
            )

        effective_min_runs = min(3, plan.trials)
        if effective_min_runs < 3:
            logger.warning(
                f"--num-profile-runs={plan.trials} is below the recommended minimum of 3. "
                "Convergence checks will have reduced statistical power."
            )

        strategy = AdaptiveStrategy(
            criterion=criterion,
            min_runs=effective_min_runs,
            max_runs=plan.trials,
            cooldown_seconds=plan.cooldown_seconds,
            auto_set_seed=plan.set_consistent_seed,
            disable_warmup_after_first=plan.disable_warmup_after_first,
        )
    else:
        strategy = FixedTrialsStrategy(
            num_trials=plan.trials,
            cooldown_seconds=plan.cooldown_seconds,
            auto_set_seed=plan.set_consistent_seed,
            disable_warmup_after_first=plan.disable_warmup_after_first,
        )

    orchestrator = MultiRunOrchestrator(base_dir=base_dir)

    try:
        results = orchestrator.execute(first_config, strategy)
    except Exception:
        logger.exception("Error executing multi-run benchmark")
        raise

    successful_runs = [r for r in results if r.success]
    failed_runs = [r for r in results if not r.success]

    logger.info("=" * 80)
    logger.info(f"All runs complete: {len(successful_runs)}/{total_runs} successful")
    if failed_runs:
        logger.warning(f"Failed runs: {', '.join(r.label for r in failed_runs)}")
    logger.info("=" * 80)

    if len(successful_runs) >= 2:
        logger.info("Computing aggregate statistics...")

        from aiperf.orchestrator.aggregation.confidence import ConfidenceAggregation

        aggregation = ConfidenceAggregation(confidence_level=plan.confidence_level)
        aggregate_result = aggregation.aggregate(results)
        aggregate_result.metadata["cooldown_seconds"] = plan.cooldown_seconds

        aggregate_dir = strategy.get_aggregate_path(base_dir)

        from aiperf.exporters.aggregate import (
            AggregateConfidenceCsvExporter,
            AggregateConfidenceJsonExporter,
            AggregateDetailedJsonExporter,
            AggregateExporterConfig,
        )

        exporter_config = AggregateExporterConfig(
            result=aggregate_result,
            output_dir=aggregate_dir,
        )

        import asyncio

        # Compute detailed aggregation synchronously before async export
        detailed_result = None
        if plan.use_adaptive:
            from aiperf.orchestrator.aggregation.detailed import DetailedAggregation

            detailed_aggregation = DetailedAggregation(
                jsonl_filename=plan.export_jsonl_file or "",
            )
            detailed_result = detailed_aggregation.aggregate(results)
            detailed_result.metadata["cooldown_seconds"] = plan.cooldown_seconds

        async def export_artifacts():
            await asyncio.to_thread(aggregate_dir.mkdir, parents=True, exist_ok=True)
            json_exporter = AggregateConfidenceJsonExporter(exporter_config)
            csv_exporter = AggregateConfidenceCsvExporter(exporter_config)

            tasks = [
                json_exporter.export(),
                csv_exporter.export(),
            ]

            if detailed_result is not None:
                detailed_config = AggregateExporterConfig(
                    result=detailed_result,
                    output_dir=aggregate_dir,
                )
                detailed_exporter = AggregateDetailedJsonExporter(detailed_config)
                tasks.append(detailed_exporter.export())

            return await asyncio.gather(*tasks)

        export_paths = asyncio.run(export_artifacts())

        json_path = export_paths[0]
        csv_path = export_paths[1]
        logger.info(f"Aggregate JSON written to: {json_path}")
        logger.info(f"Aggregate CSV written to: {csv_path}")
        if plan.use_adaptive and len(export_paths) > 2:
            logger.info(f"Collated aggregate JSON written to: {export_paths[2]}")

        _print_aggregate_summary(aggregate_result, logger)
    elif len(successful_runs) == 1:
        logger.warning(
            "Only 1 successful run - cannot compute confidence statistics. "
            "At least 2 successful runs are required."
        )
        sys.exit(1)
    else:
        logger.error(
            "All runs failed - cannot compute aggregate statistics. "
            "Please check the error messages above."
        )
        sys.exit(1)


def _print_aggregate_summary(
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

    priority_metrics = [
        "request_throughput",
        "time_to_first_token",
        "inter_token_latency",
        "request_latency",
    ]

    metrics_to_display = []
    for base_metric in priority_metrics:
        for suffix in ["_avg", "_p99", "_max", "_p50"]:
            metric_key = f"{base_metric}{suffix}"
            if metric_key in aggregate_result.metrics:
                display_name = base_metric.replace("_", " ").title()
                stat_name = suffix[1:].upper()
                if stat_name == "AVG":
                    stat_name = "Avg"
                elif stat_name.startswith("P"):
                    stat_name = f"P{stat_name[1:]}"
                else:
                    stat_name = stat_name.capitalize()

                metrics_to_display.append((metric_key, f"{display_name} ({stat_name})"))
                break

    metrics_found = 0
    for metric_key, display_name in metrics_to_display:
        metric = aggregate_result.metrics[metric_key]
        logger.info(f"\n{display_name}:")
        logger.info(f"  Mean:    {metric.mean:>12.4f} {metric.unit}")
        logger.info(f"  Std Dev: {metric.std:>12.4f} {metric.unit}")
        logger.info(f"  Min:     {metric.min:>12.4f} {metric.unit}")
        logger.info(f"  Max:     {metric.max:>12.4f} {metric.unit}")
        logger.info(f"  CV:      {metric.cv:>12.2%}")
        logger.info(
            f"  {confidence_level:.0%} CI: [{metric.ci_low:.4f}, {metric.ci_high:.4f}] {metric.unit}"
        )
        metrics_found += 1

    if metrics_found == 0:
        logger.warning("No key metrics found in aggregate results")

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
