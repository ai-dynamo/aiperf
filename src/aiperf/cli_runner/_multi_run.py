# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-run execution path for aiperf.cli_runner.

``_run_multi_benchmark`` is the entry point for sweeps and multi-trial
plans. It wires up the orchestrator + executor + (optional) search
planner, drives execution to completion, then aggregates and summarizes.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from aiperf.cli_runner._aggregate import aggregate_and_export
from aiperf.cli_runner._banner import _log_search_planner_active, log_multi_run_banner
from aiperf.cli_runner._callbacks import (
    CompletedRun,
    OnComplete,
    _invoke_callbacks,
)
from aiperf.cli_runner._strategy import (
    _build_search_planner,
    build_strategy,
    validate_convergence_config,
)
from aiperf.cli_runner._sweep_aggregate import (
    aggregate_per_variation_and_export,
    aggregate_sweep_and_export,
)
from aiperf.orchestrator.models import _variation_key
from aiperf.plugin.enums import UIType

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkConfig, BenchmarkPlan
    from aiperf.orchestrator.models import RunResult
    from aiperf.orchestrator.strategies import ExecutionStrategy


def _run_multi_benchmark(
    plan: BenchmarkPlan,
    *,
    on_complete: list[OnComplete] | None = None,
) -> None:
    """Run multiple benchmarks from a BenchmarkPlan.

    Executes trials x configs benchmarks, then aggregates results and
    computes confidence statistics. When convergence flags are set, uses
    AdaptiveStrategy for early stopping and runs both ConfidenceAggregation
    and DetailedAggregation.

    Args:
        plan: BenchmarkPlan describing the configs/trials to execute.
        on_complete: Optional list of callbacks invoked in list order after
            the orchestrator returns successfully. Skipped if execution
            raises. Each callback is isolated by ``_invoke_callbacks``:
            an exception is logged, the exit code is forced non-zero, and
            remaining callbacks still run. ``AIPERF_RAISE_ON_CALLBACK_ERROR=true``
            opts into re-raising the first failure.
    """
    from aiperf.cli_runner import _make_benchmark_run
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.logging import setup_rich_logging

    _validate_multi_benchmark_plan(plan)
    first_config = plan.configs[0]

    setup_rich_logging(_make_benchmark_run(first_config))
    logger = AIPerfLogger(__name__)

    total_runs = len(plan.configs) * plan.trials

    validate_convergence_config(plan)
    log_multi_run_banner(plan, total_runs, logger)

    base_dir = _estimate_and_log_duration(plan, first_config, total_runs, logger)

    # Strategy is rebuilt per-cell inside the orchestrator; this top-level
    # instance is kept solely so aggregate_and_export() can resolve aggregate
    # paths and seed/warmup helpers consistently with what the runs used.
    strategy = build_strategy(plan, logger)

    results = _execute_multi_benchmark(plan, base_dir, logger)

    exit_code = _summarize_and_export(
        plan,
        results,
        total_runs=total_runs,
        strategy=strategy,
        base_dir=base_dir,
        logger=logger,
    )

    # Run callbacks whenever ANY run produced artifacts, even on a partial
    # failure path: with one successful trial the per-run JSONL/CSV/JSON are
    # on disk and downstream hooks (auto-plot, exporters) can still consume
    # them. Only skip when zero runs succeeded.
    successful_runs = [r for r in results if r.success]
    if on_complete and successful_runs:
        completed = CompletedRun(artifact_dir=plan.configs[0].artifacts.dir)
        exit_code = _invoke_callbacks(on_complete, completed, exit_code, logger)

    # Match _run_single_benchmark's hang-protection: bypass Python's normal
    # teardown so multiprocessing atexit handlers and leftover ZMQ contexts
    # cannot block the interpreter from exiting (multi-run has MORE
    # subprocesses than single-run, so this is at least as critical here).
    # The orchestrator already flushed logs and wrote artifacts; killing
    # the interpreter is safe.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
    # Production never reaches here (``os._exit`` terminates the process).
    # The component-integration test harness mocks ``os._exit`` to a no-op,
    # so re-raise via ``sys.exit`` to surface the failure as a SystemExit
    # the harness can catch.
    if exit_code != 0:
        sys.exit(exit_code)


def _execute_multi_benchmark(
    plan: BenchmarkPlan,
    base_dir: Path,
    logger: AIPerfLogger,
) -> list[RunResult]:
    """Build the orchestrator + executor + planner and run the plan to completion.

    Wraps the asyncio run in a try/except that re-raises after logging;
    exceptions are owned by the caller's exit-code path.
    """
    import asyncio as _asyncio

    from aiperf.cli_runner._sweep_table import (
        SweepTableLogger,
        _should_emit_sweep_table,
    )
    from aiperf.orchestrator.local_executor import LocalSubprocessExecutor
    from aiperf.orchestrator.orchestrator import MultiRunOrchestrator

    no_flag = plan.no_sweep_table
    table_logger = (
        SweepTableLogger(plan, logger)
        if _should_emit_sweep_table(plan, no_sweep_table=no_flag)
        else None
    )
    orchestrator = MultiRunOrchestrator(base_dir=base_dir, cell_callback=table_logger)
    executor = LocalSubprocessExecutor(base_dir=base_dir)
    search_planner = _build_search_planner(plan)
    _log_search_planner_active(plan, search_planner, logger)

    try:
        return _asyncio.run(
            orchestrator.execute(plan, executor, search_planner=search_planner)
        )
    except Exception:
        logger.exception("Error executing multi-run benchmark")
        raise


def _estimate_and_log_duration(
    plan: BenchmarkPlan,
    first_config: BenchmarkConfig,
    total_runs: int,
    logger: AIPerfLogger,
) -> Path:
    """Resolve artifact/timing for a probe run, log duration, return base_dir."""
    from aiperf.config import BenchmarkRun
    from aiperf.config.resolution.resolvers import ArtifactDirResolver, TimingResolver

    probe_run = BenchmarkRun(
        benchmark_id="probe",
        cfg=first_config,
        artifact_dir=first_config.artifacts.dir,
        variables=dict(plan.variables),
    )
    ArtifactDirResolver().resolve(probe_run, for_probe=True)
    TimingResolver().resolve(probe_run)

    per_run_duration = probe_run.resolved.total_expected_duration
    if per_run_duration is not None:
        total_benchmark = per_run_duration * total_runs
        total_with_cooldown = total_benchmark + plan.cooldown_seconds * max(
            total_runs - 1, 0
        )
        logger.info(f"  Estimated duration: {total_with_cooldown:.0f}s")

    return probe_run.artifact_dir


def _validate_multi_benchmark_plan(plan: BenchmarkPlan) -> None:
    """Reject configurations multi-run can't honor before any setup work."""
    _reject_in_process_sweep_under_operator(plan)

    first_config = plan.configs[0]

    if first_config.ui_type == UIType.DASHBOARD:
        raise ValueError(
            "Dashboard UI is not supported with sweep/multi-run mode. "
            "Please use '--ui simple' or '--ui none' instead."
        )


def _reject_in_process_sweep_under_operator(plan: BenchmarkPlan) -> None:
    """Block in-process grid sweep when running inside an operator-managed pod.

    The k8s operator drives grid sweeps cluster-wide via the AIPerfSweep CR
    (one AIPerfJob per variation, controller pod sees a single-config plan).
    Adaptive outer loops, in contrast, run inside the controller pod itself
    via ``BayesianSearchPlanner`` - the controller proposes each variation
    one at a time, so the in-process adaptive path is allowed under the
    operator and is not blocked here.
    """
    if os.environ.get("AIPERF_OPERATOR_MANAGED") != "1":
        return
    if plan.is_sweep:
        swept_params = sorted(
            {
                k
                for variation in plan.variations
                if variation is not None
                for k in variation.values
            }
        )
        raise SystemExit(
            f"In-process parameter sweep ({len(plan.configs)} variations across "
            f"{swept_params or '<unknown>'}) is not supported in operator-managed "
            f"runs (AIPERF_OPERATOR_MANAGED=1). Use the AIPerfSweep CRD "
            f"(cluster-scope) for cross-job sweeps - see docs/kubernetes/sweeps.md "
            f"- or submit one AIPerfJob per variation. To run as a single point "
            f"benchmark, drop the comma in --concurrency / other magic-list flags."
        )


def _summarize_and_export(
    plan: BenchmarkPlan,
    results: list[RunResult],
    *,
    total_runs: int,
    strategy: ExecutionStrategy,
    base_dir: Path,
    logger: AIPerfLogger,
) -> int:
    """Log success/failure summary and run confidence + sweep aggregation.

    Returns an exit code (0 on full success, 1 when fewer than 2 runs
    succeeded). Does not call ``sys.exit`` - the caller is responsible for
    propagating the code so that registered ``on_complete`` callbacks still
    run on whatever per-run artifacts were produced.
    """
    import asyncio as _asyncio

    successful_runs = [r for r in results if r.success]
    failed_runs = [r for r in results if not r.success]

    logger.info("=" * 80)
    if not plan.is_sweep:
        logger.info(
            f"All runs complete: {len(successful_runs)}/{total_runs} successful"
        )
    if failed_runs:
        if plan.is_sweep:
            _log_failed_sweep_variations(failed_runs, logger)
        else:
            logger.warning(f"Failed runs: {', '.join(r.label for r in failed_runs)}")
    logger.info("=" * 80)

    if len(successful_runs) >= 2:
        logger.info("Computing aggregate statistics...")
        if plan.is_sweep:
            # Per-variation confidence aggregates (one JSON+CSV per cell with
            # >=2 successful runs) and the cross-variation sweep aggregate
            # are independent; run concurrently.
            async def _aggregate_sweep() -> None:
                await _asyncio.gather(
                    aggregate_per_variation_and_export(results, plan, base_dir, logger),
                    aggregate_sweep_and_export(results, plan, base_dir, logger),
                )

            _asyncio.run(_aggregate_sweep())
        else:
            _asyncio.run(
                aggregate_and_export(
                    results, plan, strategy=strategy, base_dir=base_dir, logger=logger
                )
            )
        return 0
    if len(successful_runs) == 1:
        if plan.is_sweep:
            logger.warning(
                "Only 1 variation succeeded - cannot compute sweep aggregate "
                "statistics. At least 2 successful variations are required."
            )
        else:
            logger.warning(
                "Only 1 successful run - cannot compute confidence statistics. "
                "At least 2 successful runs are required."
            )
        return 1
    logger.error(
        "All runs failed - cannot compute aggregate statistics. "
        "Please check the error messages above."
    )
    return 1


def _log_failed_sweep_variations(
    failed_runs: list[RunResult], logger: AIPerfLogger
) -> None:
    """Log per-variation failures for a sweep, grouped by (label, sorted values).

    Keying by label too is required so QMC cells with collision-prone integer
    values (Sobol/LHS) don't get pooled into one row of the summary; mirrors
    ``cli_runner._sweep_aggregate``.
    """
    by_variation: dict[tuple, list[RunResult]] = {}
    for r in failed_runs:
        key = _variation_key(r.variation_label or "", r.variation_values or {})
        by_variation.setdefault(key, []).append(r)

    def _format_key(label: str, params: tuple) -> str:
        kvs = ", ".join(f"{k}={v}" for k, v in params)
        return f"{label}: {kvs}" if label else kvs

    failed_values_str = [_format_key(label, params) for label, params in by_variation]
    logger.warning(f"Some sweep variations failed: {failed_values_str}")
    for (label, params), group in by_variation.items():
        params_str = _format_key(label, params)
        for r in group:
            error_msg = r.error or "(no error message)"
            logger.warning(f"  {params_str}: {error_msg}")
