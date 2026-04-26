# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-run orchestrator for AIPerf benchmarks.

Iterates variations x trials from a BenchmarkPlan via a pluggable RunExecutor.
Strategy decisions (when to stop a cell, what config to run next) are made
per-variation with a fresh strategy instance, so AdaptiveStrategy convergence
state does not leak across cells.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.orchestrator.models import RunResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.config.benchmark import BenchmarkPlan
    from aiperf.orchestrator.executor import RunExecutor


logger = logging.getLogger(__name__)

__all__ = [
    "MultiRunOrchestrator",
]


class MultiRunOrchestrator:
    """Orchestrates execution of multiple benchmark runs across variations x trials.

    Each (variation, trial) pair is executed via the injected RunExecutor.
    Strategy state is per-cell: a fresh ExecutionStrategy is built for each
    variation so adaptive convergence operates on cell-local results only.
    """

    def __init__(self, base_dir: Path) -> None:
        """Initialize MultiRunOrchestrator.

        Args:
            base_dir: Base directory for all artifacts.
        """
        self.base_dir = Path(base_dir)

    async def execute(
        self,
        plan: BenchmarkPlan,
        executor: RunExecutor,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[RunResult]:
        """Execute all (variation, trial) runs in the plan.

        Args:
            plan: BenchmarkPlan with configs[], variations[], trials, convergence config.
            executor: Concrete RunExecutor (LocalSubprocessExecutor or K8sChildJobExecutor).
            cancel_check: Optional callable polled before each variation and each
                trial inside a variation. When it returns True, the orchestrator
                returns the partial results gathered so far without starting any
                further runs.

        Returns:
            Flat list of RunResult, ordered by (variation_index, trial_index).
        """
        from aiperf._cli_runner_helpers import build_strategy
        from aiperf.config.benchmark import BenchmarkRun

        all_results: list[RunResult] = []
        logger.info(
            f"Starting multi-run benchmark: {len(plan.configs)} variations, "
            f"{plan.trials} trials per variation"
        )

        for var_idx, (cfg, variation) in enumerate(
            zip(plan.configs, plan.variations, strict=False)
        ):
            if cancel_check is not None and cancel_check():
                logger.info(f"Sweep cancelled at variation {var_idx}; aborting")
                return all_results
            strategy = build_strategy(plan, logger)  # fresh per-cell strategy
            strategy.validate_config(cfg)
            cell_results: list[RunResult] = []
            trial = 0

            while strategy.should_continue(cell_results):
                if cancel_check is not None and cancel_check():
                    logger.info(
                        f"Sweep cancelled mid-cell at v{var_idx} t{trial}; aborting"
                    )
                    all_results.extend(cell_results)
                    return all_results
                next_cfg = strategy.get_next_config(cfg, cell_results)
                label = strategy.get_run_label(trial)
                cell_dir = self.base_dir / variation.label
                artifact_dir = strategy.get_run_path(cell_dir, trial)

                run = BenchmarkRun(
                    benchmark_id=executor.derive_id(plan, var_idx, trial),
                    cfg=next_cfg,
                    variation=variation,
                    trial=trial,
                    label=label,
                    artifact_dir=artifact_dir,
                )
                logger.info(f"[v{var_idx} t{trial}] Executing {label}...")
                result = await executor.execute(run)
                cell_results.append(result)
                trial += 1

                if self._sweep_failure_threshold_exceeded(
                    all_results + cell_results, plan
                ):
                    logger.warning("Failure threshold exceeded; aborting sweep")
                    all_results.extend(cell_results)
                    return all_results

                if strategy.should_continue(cell_results):
                    cooldown = strategy.get_cooldown_seconds()
                    if cooldown > 0:
                        logger.info(f"Cooldown: {cooldown}s")
                        await asyncio.sleep(cooldown)

            all_results.extend(cell_results)

        successful = sum(1 for r in all_results if r.success)
        logger.info(f"All runs complete: {successful}/{len(all_results)} successful")
        return all_results

    @staticmethod
    def _sweep_failure_threshold_exceeded(
        results: list[RunResult], plan: BenchmarkPlan
    ) -> bool:
        """Return True if the sweep should abort due to failure-policy limits."""
        failure_policy = getattr(plan, "failure_policy", None)
        if failure_policy is None:
            return False
        if getattr(failure_policy, "on_child_failure", "continue") == "abort":
            return any(not r.success for r in results)
        max_fail = getattr(failure_policy, "max_failures", 0)
        if max_fail > 0:
            failed = sum(1 for r in results if not r.success)
            return failed >= max_fail
        return False
