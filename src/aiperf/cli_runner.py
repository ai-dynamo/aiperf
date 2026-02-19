# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import sys
from typing import TYPE_CHECKING

from aiperf.cli_utils import raise_startup_error_and_exit
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.gpu_telemetry.metrics_config import MetricsConfigLoader
from aiperf.plugin.enums import ServiceType, UIType

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.orchestrator.aggregation.base import AggregateResult


def run_system_controller(
    user_config: UserConfig,
    service_config: ServiceConfig,
) -> None:
    """Run the system controller with the given configuration.

    If num_profile_runs > 1, runs multi-run orchestration for confidence reporting.
    Otherwise, runs a single benchmark (backward compatibility).
    """
    # Check if multi-run mode is enabled
    if user_config.loadgen.num_profile_runs > 1:
        _run_multi_benchmark(user_config, service_config)
    else:
        _run_single_benchmark(user_config, service_config)


def _run_single_benchmark(
    user_config: UserConfig,
    service_config: ServiceConfig,
) -> None:
    """Run a single benchmark (original behavior)."""

    # NOTE: On macOS, when using the Textual UI with multiprocessing, terminal corruption
    # (ASCII garbage, freezing) can occur when mouse events interfere with child processes.
    # We apply multiple layers of protection:
    # 1. Set spawn method early (before any multiprocessing operations)
    # 2. Create log_queue before any UI initialization
    # 3. Set FD_CLOEXEC on terminal file descriptors
    # 4. Close terminal FDs in child processes (done in bootstrap.py)

    import multiprocessing
    import platform

    is_macos = platform.system() == "Darwin"
    using_dashboard = service_config.ui_type == UIType.DASHBOARD

    # Force spawn method on macOS to prevent fork-related issues.
    # This should already be the default, but we'll set it explicitly just in case.
    if is_macos and using_dashboard:
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method("spawn", force=True)

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.common.tokenizer_validator import validate_tokenizer_early

    logger = AIPerfLogger(__name__)

    # Create log_queue before UI initialization to minimize FD inheritance issues.
    log_queue = None
    if using_dashboard:
        from aiperf.common.logging import get_global_log_queue

        log_queue = get_global_log_queue()

        # Set FD_CLOEXEC on terminal file descriptors on macOS.
        # This ensures terminal FDs are closed when child processes spawn.
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

        setup_rich_logging(user_config, service_config)

    # Create and start the system controller
    logger.info("Starting AIPerf System")

    # Validate tokenizer early (before spawning services) to fail fast.
    user_config.tokenizer.resolved_names = validate_tokenizer_early(user_config, logger)

    # Validate custom GPU metrics CSV file
    if user_config.gpu_telemetry_metrics_file:
        try:
            csv_path = user_config.gpu_telemetry_metrics_file
            logger.info(f"Custom GPU metrics file configured: {csv_path}")

            loader = MetricsConfigLoader()
            custom_metrics, _ = loader.build_custom_metrics_from_csv(csv_path)
            logger.info(
                f"Validated {len(custom_metrics)} custom metrics from {csv_path}"
            )
        except Exception as e:
            logger.exception("Error validating custom GPU metrics file")
            raise_startup_error_and_exit(
                f"Invalid custom GPU metrics file: {e}",
                title="GPU Metrics Configuration Error",
            )

    try:
        bootstrap_and_run_service(
            service_type=ServiceType.SYSTEM_CONTROLLER,
            service_config=service_config,
            user_config=user_config,
            log_queue=log_queue,
        )
    except Exception:
        logger.exception("Error running AIPerf System")
        raise
    finally:
        logger.debug("AIPerf System exited")


def _run_multi_benchmark(
    user_config: UserConfig,
    service_config: ServiceConfig,
) -> None:
    """Run multiple benchmarks for confidence reporting.

    Executes num_profile_runs benchmarks with the same configuration,
    then aggregates results and computes confidence statistics.
    """
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.logging import setup_rich_logging
    from aiperf.exporters.aggregate import (
        AggregateConfidenceCsvExporter,
        AggregateConfidenceJsonExporter,
        AggregateExporterConfig,
    )
    from aiperf.orchestrator.aggregation.confidence import ConfidenceAggregation
    from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
    from aiperf.orchestrator.strategies import FixedTrialsStrategy

    # Validate and adjust UI type for multi-run mode
    if (
        "ui_type" in service_config.model_fields_set
        and service_config.ui_type == UIType.DASHBOARD
    ):
        raise ValueError(
            "Dashboard UI is not supported with multi-run mode (--num-profile-runs > 1) "
            "due to terminal control limitations. "
            "Please use '--ui simple' or '--ui none' instead."
        )

    # Set default to simple if ui_type wasn't explicitly set
    if "ui_type" not in service_config.model_fields_set:
        service_config.ui_type = UIType.SIMPLE

    # Set up logging so output is visible
    setup_rich_logging(user_config, service_config)

    logger = AIPerfLogger(__name__)

    # Inform user about UI mode (now that logging is set up)
    if "ui_type" not in service_config.model_fields_set:
        logger.info(
            "Multi-run mode: UI automatically set to 'simple' "
            "(use '--ui none' to disable UI output)"
        )

    # Print multi-run banner
    num_runs = user_config.loadgen.num_profile_runs
    confidence_level = user_config.loadgen.confidence_level
    cooldown = user_config.loadgen.profile_run_cooldown_seconds

    logger.info("=" * 80)
    logger.info("Starting Multi-Run Confidence Reporting")
    logger.info(f"  Number of runs: {num_runs}")
    logger.info(f"  Confidence level: {confidence_level:.0%}")
    logger.info(f"  Cooldown between runs: {cooldown}s")
    logger.info("=" * 80)

    # Create strategy
    strategy = FixedTrialsStrategy(
        num_trials=num_runs,
        cooldown_seconds=cooldown,
        auto_set_seed=user_config.loadgen.set_consistent_seed,
        disable_warmup_after_first=user_config.loadgen.profile_run_disable_warmup_after_first,
    )

    # Create orchestrator
    orchestrator = MultiRunOrchestrator(
        base_dir=user_config.output.artifact_directory, service_config=service_config
    )

    # Execute runs
    try:
        results = orchestrator.execute(user_config, strategy)
    except Exception:
        logger.exception("Error executing multi-run benchmark")
        raise

    # Count successful runs
    successful_runs = [r for r in results if r.success]
    failed_runs = [r for r in results if not r.success]

    logger.info("=" * 80)
    logger.info(f"All runs complete: {len(successful_runs)}/{num_runs} successful")
    if failed_runs:
        logger.warning(f"Failed runs: {', '.join(r.label for r in failed_runs)}")
    logger.info("=" * 80)

    # Aggregate results if we have at least 2 successful runs
    if len(successful_runs) >= 2:
        logger.info("Computing aggregate statistics...")

        aggregation = ConfidenceAggregation(confidence_level=confidence_level)
        aggregate_result = aggregation.aggregate(results)

        # Add cooldown to metadata
        aggregate_result.metadata["cooldown_seconds"] = cooldown

        # Write aggregate artifacts using exporters
        aggregate_dir = strategy.get_aggregate_path(
            user_config.output.artifact_directory
        )

        # Create exporter config
        exporter_config = AggregateExporterConfig(
            result=aggregate_result,
            output_dir=aggregate_dir,
        )

        # Export both JSON and CSV in a single async context
        # This avoids multiple asyncio.run() calls and is more efficient
        import asyncio

        async def export_artifacts():
            """Export aggregate artifacts asynchronously."""
            # Create directory asynchronously
            await asyncio.to_thread(aggregate_dir.mkdir, parents=True, exist_ok=True)

            # Export JSON and CSV concurrently
            json_exporter = AggregateConfidenceJsonExporter(exporter_config)
            csv_exporter = AggregateConfidenceCsvExporter(exporter_config)

            json_path, csv_path = await asyncio.gather(
                json_exporter.export(),
                csv_exporter.export(),
            )

            return json_path, csv_path

        json_path, csv_path = asyncio.run(export_artifacts())

        logger.info(f"Aggregate JSON written to: {json_path}")
        logger.info(f"Aggregate CSV written to: {csv_path}")

        # Print summary
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
    aggregate_result: "AggregateResult", logger: "AIPerfLogger"
) -> None:
    """Print a comprehensive summary of aggregate statistics to console.

    Args:
        aggregate_result: AggregateResult with computed statistics
        logger: Logger instance for output
    """

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

    # Get confidence level from metadata
    confidence_level = aggregate_result.metadata.get("confidence_level", 0.95)
    logger.info(f"Confidence Level: {confidence_level:.0%}")

    logger.info("")
    logger.info("Key Metrics:")
    logger.info("-" * 80)

    # Define priority metrics to display (in order of preference)
    # We'll look for these base metric names with _avg, _p99, _max suffixes
    priority_metrics = [
        "request_throughput",
        "time_to_first_token",
        "inter_token_latency",
        "request_latency",
    ]

    # Build list of metrics to display by finding available stat variants
    metrics_to_display = []
    for base_metric in priority_metrics:
        # Look for _avg first (most common), then _p99, then _max
        for suffix in ["_avg", "_p99", "_max", "_p50"]:
            metric_key = f"{base_metric}{suffix}"
            if metric_key in aggregate_result.metrics:
                # Create display name (e.g., "Request Throughput (Avg)")
                display_name = base_metric.replace("_", " ").title()
                stat_name = suffix[1:].upper()  # Remove leading underscore
                if stat_name == "AVG":
                    stat_name = "Avg"
                elif stat_name.startswith("P"):
                    stat_name = f"P{stat_name[1:]}"  # P99, P50, etc.
                else:
                    stat_name = stat_name.capitalize()

                metrics_to_display.append((metric_key, f"{display_name} ({stat_name})"))
                break  # Only show one stat variant per base metric

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
