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

    from aiperf.config.benchmark import BenchmarkPlan, BenchmarkRun
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

        Iteration order honors plan.parameter_sweep_mode:

        - INDEPENDENT (default): variations outer, trials inner. Artifact
          tree is <base>/<variation>/profile_runs/run_NNNN/.
        - REPEATED: trials outer, variations inner. Artifact tree is
          <base>/profile_runs/trial_NNNN/<variation>/profile_runs/run_0001/.
          The trailing run_0001 segment comes from the per-cell strategy
          and is unconditional; trial-NNNN is the orchestrator's prefix.

        Args:
            plan: BenchmarkPlan with configs[], variations[], trials, convergence config.
            executor: Concrete RunExecutor (LocalSubprocessExecutor or K8sChildJobExecutor).
            cancel_check: Optional callable polled before each variation and each
                trial inside a variation. When it returns True, the orchestrator
                returns the partial results gathered so far without starting any
                further runs.

        Returns:
            Flat list of RunResult, ordered by the active iteration order.
        """
        from aiperf.common.enums import SweepMode

        if plan.parameter_sweep_mode == SweepMode.REPEATED:
            return await self._execute_repeated(
                plan, executor, cancel_check=cancel_check
            )
        return await self._execute_independent(
            plan, executor, cancel_check=cancel_check
        )

    async def _execute_independent(
        self,
        plan: BenchmarkPlan,
        executor: RunExecutor,
        *,
        cancel_check: Callable[[], bool] | None,
    ) -> list[RunResult]:
        """Variations-outer, trials-inner iteration (the default mode).

        Each variation gets a fresh ExecutionStrategy; adaptive convergence
        operates on cell-local results only. Artifact tree:
        <base>/<variation>/profile_runs/run_NNNN/.
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
            if var_idx > 0 and plan.parameter_sweep_cooldown_seconds > 0:
                logger.debug(
                    f"Inter-variation cooldown: "
                    f"{plan.parameter_sweep_cooldown_seconds}s before v{var_idx}"
                )
                await asyncio.sleep(plan.parameter_sweep_cooldown_seconds)
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
                self._stamp_variation_metadata(result, run, trial)
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

    async def _execute_repeated(
        self,
        plan: BenchmarkPlan,
        executor: RunExecutor,
        *,
        cancel_check: Callable[[], bool] | None,
    ) -> list[RunResult]:
        """Trials-outer, variations-inner iteration (repeated mode).

        Each variation has one strategy reused across trials, called once
        per trial with that variation's growing prior-results history.
        Artifact tree:
        <base>/profile_runs/trial_NNNN/<variation>/profile_runs/run_0001/.
        """
        from aiperf._cli_runner_helpers import build_strategy
        from aiperf.config.benchmark import BenchmarkRun

        if plan.use_adaptive:
            raise ValueError(
                "parameter_sweep_mode='repeated' is incompatible with "
                "convergence-based stopping (--convergence-metric). The "
                "trial-outer iteration order has no place to evaluate "
                "convergence per-cell. Use 'independent' for adaptive "
                "sweeps, or remove --convergence-metric."
            )

        all_results: list[RunResult] = []
        logger.info(
            f"Starting multi-run benchmark (repeated): {plan.trials} trials x "
            f"{len(plan.configs)} variations"
        )

        # Why we track per-variation history:
        # FixedTrialsStrategy.get_next_config keys disable_warmup_after_first
        # off `len(prior_results) > 0`. In repeated mode each (variation,
        # trial) cell fires exactly once, so the natural per-cell results
        # list is always [] and warmup would re-enable on every trial -
        # silently diverging from main's _execute_trials_then_sweep
        # semantics where warmup runs only on trial 1 across all
        # variations. We thread per-variation history across the outer
        # trial loop so the strategy sees prior_results growing as it
        # would in independent mode. Strategy contract only inspects
        # len(prior); contents are not read - so we never have to keep
        # this list pruned or even successful-only. Do NOT replace with
        # `[]` per call: the silent re-enable is unobservable in unit
        # tests but corrupts wall-clock comparisons across modes.
        strategies = [build_strategy(plan, logger) for _ in plan.configs]
        for strategy, cfg in zip(strategies, plan.configs, strict=True):
            strategy.validate_config(cfg)
        per_variation_history: list[list[RunResult]] = [[] for _ in plan.configs]

        for trial in range(plan.trials):
            if cancel_check is not None and cancel_check():
                logger.info(f"Sweep cancelled at trial {trial}; aborting")
                return all_results
            for var_idx, (cfg, variation) in enumerate(
                zip(plan.configs, plan.variations, strict=False)
            ):
                if cancel_check is not None and cancel_check():
                    logger.info(
                        f"Sweep cancelled mid-trial at [v{var_idx} t{trial}]; aborting"
                    )
                    return all_results
                strategy = strategies[var_idx]
                next_cfg = strategy.get_next_config(cfg, per_variation_history[var_idx])
                label = strategy.get_run_label(trial)
                cell_dir = (
                    self.base_dir
                    / "profile_runs"
                    / f"trial_{trial + 1:04d}"
                    / variation.label
                )
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
                self._stamp_variation_metadata(result, run, trial)
                all_results.append(result)
                per_variation_history[var_idx].append(result)

                if self._sweep_failure_threshold_exceeded(all_results, plan):
                    logger.warning("Failure threshold exceeded; aborting sweep")
                    return all_results

                if (
                    var_idx + 1 < len(plan.configs)
                    and plan.parameter_sweep_cooldown_seconds > 0
                ):
                    logger.debug(
                        f"Inter-variation cooldown (within trial {trial}): "
                        f"{plan.parameter_sweep_cooldown_seconds}s"
                    )
                    await asyncio.sleep(plan.parameter_sweep_cooldown_seconds)

            if trial + 1 < plan.trials:
                cooldown = strategies[0].get_cooldown_seconds()
                if cooldown > 0:
                    logger.info(f"Inter-trial cooldown: {cooldown}s")
                    await asyncio.sleep(cooldown)

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

    @staticmethod
    def _stamp_variation_metadata(
        result: RunResult, run: BenchmarkRun, trial_index: int
    ) -> None:
        """Populate sweep-aggregation fields on result from the originating run."""
        if run.variation is not None:
            result.variation_label = run.variation.label
            result.variation_values = dict(run.variation.values)
        result.trial_index = trial_index
