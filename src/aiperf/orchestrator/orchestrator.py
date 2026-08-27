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
from typing import TYPE_CHECKING, Any

from aiperf.orchestrator.models import RunResult, _variation_key

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun
    from aiperf.config.sweep import SweepVariation
    from aiperf.orchestrator.executor import RunExecutor


logger = logging.getLogger(__name__)

__all__ = [
    "MultiRunOrchestrator",
]


def _resolve_artifact_dir(
    base_dir: Path,
    plan: BenchmarkPlan,
    variation: Any,
    trial_index: int,
    *,
    iteration_order: Any | None = None,
) -> Path:
    """Compute the artifact dir for one (variation, trial) run.

    Five cases for grid runs, branching on (is_sweep, trials > 1,
    iteration order); one extra case for adaptive (BO) runs:

    | sweep | trials | order       | layout                                          |
    |-------|--------|-------------|-------------------------------------------------|
    | no    | 1      | -           | ``<base>/``                                     |
    | no    | >1     | -           | ``<base>/profile_runs/run_NNNN/``               |
    | yes   | 1      | -           | ``<base>/<dir_name>/``                          |
    | yes   | >1     | REPEATED    | ``<base>/profile_runs/trial_NNNN/<dir_name>/``  |
    | yes   | >1     | INDEPENDENT | ``<base>/<dir_name>/profile_runs/trial_NNNN/``  |
    | adaptive | any | -           | ``<base>/search_iter_NNNN/profile_runs/run_NNNN/`` |

    The adaptive (BO) row uses ``variation.label`` (which the search
    planners populate as ``search_iter_NNNN``) instead of
    ``variation.dir_name`` so each BO iteration writes into its own
    iteration-numbered tree - ``variation.dir_name`` would name the dir
    after the proposed coordinates and collide when the planner
    re-proposes a nearby point.

    Note the asymmetric inner-dir naming for grid runs: ``run_NNNN`` for
    the no-sweep multi-run case, ``trial_NNNN`` for the sweep +
    INDEPENDENT multi-run case. Downstream consumers (plotters,
    dashboards) depend on this asymmetry.

    ``trial_index`` is zero-based; emitted dir names are 1-based and
    zero-padded to 4 digits.
    """
    from aiperf.common.enums import SweepMode

    if plan.is_adaptive_search:
        return (
            base_dir / variation.label / "profile_runs" / f"run_{trial_index + 1:04d}"
        )

    is_sweep = plan.is_sweep
    multi_run = plan.trials > 1
    if iteration_order is None:
        from aiperf.cli_runner._sweep_aggregate import _plan_iteration_order

        iteration_order = _plan_iteration_order(plan)

    if not is_sweep and not multi_run:
        return base_dir
    if not is_sweep and multi_run:
        return base_dir / "profile_runs" / f"run_{trial_index + 1:04d}"
    if is_sweep and not multi_run:
        return base_dir / variation.dir_name
    if iteration_order == SweepMode.REPEATED:
        return (
            base_dir
            / "profile_runs"
            / f"trial_{trial_index + 1:04d}"
            / variation.dir_name
        )
    return (
        base_dir / variation.dir_name / "profile_runs" / f"trial_{trial_index + 1:04d}"
    )


def _plan_cooldown_seconds(plan: BenchmarkPlan) -> float:
    """Inter-variation cooldown lives on plan.sweep; 0.0 outside a sweep."""
    return plan.sweep.cooldown_seconds if plan.sweep is not None else 0.0


def resolve_run_seed(
    plan: BenchmarkPlan, variation: SweepVariation, trial: int = 0
) -> int | None:
    # Why: grid/zip/scenario sweeps pre-compute the full per-variation seed
    # list at plan-build (`base + idx`). Adaptive sweeps
    # discover variations at runtime, so `variation.index` can exceed the
    # plan-time list length — fall back to SHA-256 derivation over
    # `(envelope_seed, variation.label)` so iter > 0 doesn't silently drop the
    # seed and the same proposal label always yields the same workload.
    # When `multi_run.vary_seed_per_trial` is set, derive a distinct per-trial
    # seed instead — single SHA over `(envelope_seed, variation.label, trial)`
    # so the same (variation, trial) coordinate always reproduces.
    from aiperf.common.random_generator import derive_variation_seed

    if plan.multi_run.vary_seed_per_trial and plan.random_seed is not None:
        return derive_variation_seed(
            plan.random_seed, f"{variation.label}:trial:{trial}"
        )
    if variation.index < len(plan.variation_seeds):
        return plan.variation_seeds[variation.index]
    return derive_variation_seed(plan.random_seed, variation.label)


def _build_strategy(plan: BenchmarkPlan) -> Any:
    """Construct a per-variation execution strategy from a BenchmarkPlan."""
    from aiperf.cli_runner._strategy import build_strategy

    return build_strategy(plan, logger)


class MultiRunOrchestrator:
    """Orchestrates execution of multiple benchmark runs across variations x trials.

    Each (variation, trial) pair is executed via the injected RunExecutor.
    Strategy state is per-cell: a fresh ExecutionStrategy is built for each
    variation so adaptive convergence operates on cell-local results only.
    """

    def __init__(
        self,
        base_dir: Path,
        *,
        cell_callback: Callable[[tuple, dict], None] | None = None,
    ) -> None:
        """Initialize MultiRunOrchestrator.

        Args:
            base_dir: Base directory for all artifacts.
            cell_callback: Optional per-cell observer fired after each variation
                finishes its trials. Receives ``(variation_key, cell)`` where
                ``variation_key`` is the hashable
                :func:`aiperf.orchestrator.models._variation_key` output
                ``(label, sorted_values_tuple)`` (nested override values are
                canonicalized to JSON strings so the key stays hashable)
                and ``cell`` is the dict produced by
                :func:`aiperf.cli_runner._pareto._aggregate_one_cell`.
                Useful for live observers (e.g. a streaming Pareto tracker).
                Exceptions raised by the callback are caught and logged at
                WARNING so a buggy observer cannot break the sweep.
        """
        self.base_dir = Path(base_dir)
        self._cell_callback = cell_callback

    def _fire_cell_callback(
        self,
        plan: BenchmarkPlan,
        variation: Any,
        cell_results: list[RunResult],
    ) -> None:
        """Invoke the per-cell observer callback if one is registered.

        Catches all exceptions from the callback so a buggy observer can
        never break the sweep. Logs at WARNING. No-op when ``cell_callback``
        was not supplied. When the recipe declares no ``pareto_axes``
        (``_aggregate_one_cell`` returns ``None``), a minimal cell dict
        with ``params`` populated and ``x``/``y`` set to ``None`` is
        synthesized so sweep-mode-agnostic consumers (e.g.
        ``SweepTableLogger``) still receive every variation.
        """
        if self._cell_callback is None:
            return
        try:
            from aiperf.cli_runner._pareto import _aggregate_one_cell

            cell = _aggregate_one_cell(cell_results, plan, variation)
            if cell is None:
                # Recipe declares no pareto_axes: build a minimal cell so
                # consumers (e.g. SweepTableLogger) still receive params
                # and the trial-result list. Pareto x/y stay None — only
                # consumers that opt into them must check.
                cell = {
                    "params": dict(variation.values),
                    "x": None,
                    "y": None,
                    "pareto_optimal": False,
                }
            cell["_cell_results"] = cell_results  # opaque pass-through for consumers
            variation_key = _variation_key(
                getattr(variation, "label", None) or "", variation.values
            )
            self._cell_callback(variation_key, cell)
        except Exception:
            logger.warning("cell_callback raised; suppressing", exc_info=True)

    def _maybe_write_sampling_design(self, plan: BenchmarkPlan) -> None:
        """Write `sweep_aggregate/sampling_design.json` for QMC sweeps.

        Records the actually-executed sample values, sourced from
        ``plan.variations`` (populated upstream by ``expand_qmc_sweep``).
        Re-instantiating a fresh QMC engine here would draw a *second*,
        unrelated sample set whenever ``sweep.seed`` is None (the default),
        producing an audit trail that does not match the variants the
        orchestrator runs.

        No-op for non-QMC sweeps. Called before any cells so that a
        crashed sweep still leaves a faithful design record on disk.
        """
        from aiperf.config.sweep import LatinHypercubeSweep, SobolSweep

        sweep = getattr(plan, "sweep", None)
        if not isinstance(sweep, (SobolSweep, LatinHypercubeSweep)):
            return

        import math

        import orjson

        agg_dir = self.base_dir / "sweep_aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)

        # Pull mapped values directly from the already-expanded variations
        # so the audit file matches the variants that actually ran.
        dim_paths = [d.path for d in sweep.dimensions]
        samples_mapped: list[list[Any]] = []
        for variation in plan.variations:
            row: list[Any] = []
            for path in dim_paths:
                value = variation.values[path]
                # orjson 3.x silently coerces nan/inf to null. SamplingDimension
                # validators should reject non-finite lo/hi, but be defensive
                # so we never write a misleading null into the audit.
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(
                        f"non-finite value {value!r} in variation "
                        f"{variation.label!r} at dim {path!r}; refusing to "
                        f"write a misleading sampling_design.json"
                    )
                row.append(value)
            samples_mapped.append(row)

        design = {
            "type": sweep.type,
            "samples": sweep.samples,
            "seed": sweep.seed,
            "scramble": getattr(sweep, "scramble", None),
            "optimization": getattr(sweep, "optimization", None),
            "dimensions": [
                {
                    "path": d.path,
                    "lo": d.lo,
                    "hi": d.hi,
                    "scale": d.scale,
                    "kind": d.kind,
                    "choices": d.choices,
                }
                for d in sweep.dimensions
            ],
            "samples_mapped": samples_mapped,
        }
        (agg_dir / "sampling_design.json").write_bytes(
            orjson.dumps(design, option=orjson.OPT_INDENT_2, default=str)
        )

    async def execute(
        self,
        plan: BenchmarkPlan,
        executor: RunExecutor,
        *,
        cancel_check: Callable[[], bool] | None = None,
        search_planner: Any = None,
    ) -> list[RunResult]:
        """Execute all (variation, trial) runs in the plan.

        Iteration order:

        - When ``plan.is_adaptive_search`` is True, dispatches to
          :meth:`execute_adaptive_search` (BO / adaptive). ``search_planner``
          must be supplied in this case.
        - Otherwise honors the grid sweep's iteration_order:

          - INDEPENDENT: variations outer, trials inner.
          - REPEATED (default): trials outer, variations inner; one run
            per (variation, trial) cell.

        Artifact tree branches on (is_sweep, trials > 1, iteration order)
        - see :func:`_resolve_artifact_dir` for the full table.

        Args:
            plan: BenchmarkPlan with configs[], variations[], trials, convergence config.
            executor: Concrete RunExecutor (LocalSubprocessExecutor or K8sChildJobExecutor).
            cancel_check: Optional callable polled before each variation and each
                trial inside a variation. When it returns True, the orchestrator
                returns the partial results gathered so far without starting any
                further runs.
            search_planner: Outer-loop planner instance (e.g.
                ``BayesianSearchPlanner``). Required when ``plan.is_adaptive_search``;
                ignored otherwise.

        Returns:
            Flat list of RunResult, ordered by the active iteration order.
        """
        self._maybe_write_sampling_design(plan)

        from aiperf.cli_runner._sweep_aggregate import _plan_iteration_order
        from aiperf.common.enums import SweepMode

        if plan.is_adaptive_search:
            if search_planner is None:
                raise ValueError(
                    "plan.sweep is an AdaptiveSearchSweep but no search_planner was passed to execute(). "
                    "The CLI runner is expected to instantiate one and forward it."
                )
            return await self.execute_adaptive_search(
                plan, executor, search_planner, cancel_check=cancel_check
            )

        if _plan_iteration_order(plan) == SweepMode.REPEATED:
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
        """Variations-outer, trials-inner iteration.

        Each variation gets a fresh ExecutionStrategy; adaptive convergence
        operates on cell-local results only. See
        :func:`_resolve_artifact_dir` for the full layout table.
        """
        all_results: list[RunResult] = []
        logger.info(
            f"Starting multi-run benchmark (independent): {len(plan.configs)} variations x "
            f"{plan.trials} trials per variation"
        )

        for var_idx, (cfg, variation) in enumerate(
            zip(plan.configs, plan.variations, strict=True)
        ):
            if cancel_check is not None and cancel_check():
                logger.info(f"Sweep cancelled at variation {var_idx}; aborting")
                return all_results
            if var_idx > 0 and _plan_cooldown_seconds(plan) > 0:
                cooldown = _plan_cooldown_seconds(plan)
                logger.debug(f"Inter-variation cooldown: {cooldown}s before v{var_idx}")
                await asyncio.sleep(cooldown)
            strategy = _build_strategy(plan)  # fresh per-cell strategy
            strategy.validate_config(cfg)

            cell_results, aborted = await self._run_independent_cell(
                plan,
                executor,
                strategy=strategy,
                cfg=cfg,
                variation=variation,
                var_idx=var_idx,
                prior_all_results=all_results,
                cancel_check=cancel_check,
            )
            all_results.extend(cell_results)
            if aborted:
                return all_results

        successful = sum(1 for r in all_results if r.success)
        if plan.is_sweep:
            logger.info(
                f"Independent mode complete: {successful}/{len(all_results)} runs successful"
            )
        else:
            logger.info(
                f"All runs complete: {successful}/{len(all_results)} successful"
            )
        return all_results

    async def _run_independent_cell(
        self,
        plan: BenchmarkPlan,
        executor: RunExecutor,
        *,
        strategy: Any,
        cfg: Any,
        variation: Any,
        var_idx: int,
        prior_all_results: list[RunResult],
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[list[RunResult], bool]:
        """Run all trials for one variation cell in independent mode.

        Returns ``(cell_results, aborted)`` where ``aborted`` signals the
        caller to stop iterating further variations (cancel-check fired
        mid-cell, or sweep failure threshold tripped).
        """
        from aiperf.config.resolution.plan import BenchmarkRun

        cell_results: list[RunResult] = []
        trial = 0
        while strategy.should_continue(cell_results):
            if cancel_check is not None and cancel_check():
                logger.info(
                    f"Sweep cancelled mid-cell at v{var_idx} t{trial}; aborting"
                )
                return cell_results, True
            next_cfg = strategy.get_next_config(cfg, cell_results)
            label = strategy.get_run_label(trial)
            artifact_dir = _resolve_artifact_dir(self.base_dir, plan, variation, trial)

            run = BenchmarkRun(
                benchmark_id=executor.derive_id(plan, var_idx, trial),
                sweep_id=plan.sweep_id,
                cfg=next_cfg,
                variation=variation,
                trial=trial,
                label=label,
                artifact_dir=artifact_dir,
                random_seed=resolve_run_seed(plan, variation, trial),
                variables=dict(plan.variables),
            )
            logger.info(f"[v{var_idx} t{trial}] Executing {label}...")
            result = await executor.execute(run)
            self._stamp_variation_metadata(result, run, trial)
            cell_results.append(result)
            trial += 1

            if self._sweep_failure_threshold_exceeded(
                prior_all_results + cell_results, plan
            ):
                logger.warning("Failure threshold exceeded; aborting sweep")
                return cell_results, True

            if strategy.should_continue(cell_results):
                cooldown = strategy.get_cooldown_seconds()
                if cooldown > 0:
                    logger.info(f"Cooldown: {cooldown}s")
                    await asyncio.sleep(cooldown)

        self._fire_cell_callback(plan, variation, cell_results)
        return cell_results, False

    async def execute_adaptive_search(
        self,
        plan: BenchmarkPlan,
        executor: RunExecutor,
        planner: Any,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[RunResult]:
        """Drive an adaptive outer loop (e.g. BO).

        Each iteration: ask planner for a (cfg, variation), run all trials
        for it via :meth:`_run_independent_cell`, feed results back to the
        planner, write search_history.json incrementally.
        """
        from aiperf.exporters.search_history import write_search_history

        all_results: list[RunResult] = []
        sweep = plan.sweep
        assert sweep is not None  # guaranteed by plan.is_adaptive_search
        logger.info(
            f"Starting adaptive outer-loop benchmark "
            f"({sweep.planner}, max_iterations={sweep.max_iterations}, "
            f"trials per point={plan.trials})"
        )

        def _flush_history(reason: str | None) -> None:
            write_search_history(
                self.base_dir,
                planner.history(),
                sweep,
                convergence_reason=reason,
                planner=planner,
            )

        while True:
            if cancel_check is not None and cancel_check():
                logger.info(
                    f"Adaptive outer loop cancelled after {planner.iter_count} iterations"
                )
                _flush_history("cancelled")
                return all_results

            proposal = planner.ask()
            if proposal is None:
                reason = planner.convergence_reason() or "unknown"
                logger.info(
                    "Adaptive outer loop terminated after %d iterations (reason=%s)",
                    planner.iter_count,
                    reason,
                )
                _flush_history(reason)
                return all_results
            cfg, variation = proposal
            strategy = _build_strategy(plan)
            strategy.validate_config(cfg)

            logger.info(f"[search iter {variation.index}] proposing {variation.values}")
            cell_results, aborted = await self._run_independent_cell(
                plan,
                executor,
                strategy=strategy,
                cfg=cfg,
                variation=variation,
                var_idx=variation.index,
                prior_all_results=all_results,
                cancel_check=cancel_check,
            )
            planner.tell(variation, cell_results)
            all_results.extend(cell_results)
            _flush_history(None)

            if aborted:
                logger.warning(
                    f"Outer-loop cell at iter {variation.index} aborted; halting BO"
                )
                _flush_history("aborted")
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
        One run per (variation, trial) cell. See
        :func:`_resolve_artifact_dir` for the full layout table.
        """
        all_results: list[RunResult] = []
        logger.info(
            f"Starting multi-run benchmark (repeated): {plan.trials} trials x "
            f"{len(plan.configs)} variations"
        )
        strategies, per_variation_history = self._build_repeated_state(plan)

        for trial in range(plan.trials):
            if cancel_check is not None and cancel_check():
                logger.info(f"Sweep cancelled at trial {trial}; aborting")
                return all_results
            cancelled = await self._run_repeated_trial(
                plan,
                executor,
                strategies=strategies,
                trial=trial,
                per_variation_history=per_variation_history,
                all_results=all_results,
                cancel_check=cancel_check,
            )
            if cancelled:
                return all_results
            if trial + 1 < plan.trials:
                cooldown = strategies[0].get_cooldown_seconds()
                if cooldown > 0:
                    logger.info(f"Inter-trial cooldown: {cooldown}s")
                    await asyncio.sleep(cooldown)

        successful = sum(1 for r in all_results if r.success)
        if plan.is_sweep:
            logger.info(
                f"Repeated mode complete: {successful}/{len(all_results)} runs successful"
            )
        else:
            logger.info(
                f"All runs complete: {successful}/{len(all_results)} successful"
            )
        return all_results

    @staticmethod
    def _build_repeated_state(
        plan: BenchmarkPlan,
    ) -> tuple[list[Any], list[list[RunResult]]]:
        """Build per-variation strategies and prior-results history for repeated mode.

        Why we track per-variation history:
        FixedTrialsStrategy.get_next_config keys disable_warmup_after_first
        off `len(prior_results) > 0`. In repeated mode each (variation,
        trial) cell fires exactly once, so the natural per-cell results
        list is always [] and warmup would re-enable on every trial. The
        invariant we want is: warmup runs only on trial 1 across all
        variations. We thread per-variation history across the outer
        trial loop so the strategy sees prior_results growing as it
        would in independent mode. Strategy contract only inspects
        len(prior); contents are not read - so we never have to keep
        this list pruned or even successful-only. Do NOT replace with
        `[]` per call: the silent re-enable is unobservable in production
        logs but corrupts wall-clock comparisons across modes.
        Regression-locked by tests/unit/orchestrator/test_multi_run_orchestrator.py
        ::test_repeated_mode_passes_growing_prior_results_to_strategy.
        """
        strategies = [_build_strategy(plan) for _ in plan.configs]
        for strategy, cfg in zip(strategies, plan.configs, strict=True):
            strategy.validate_config(cfg)
        per_variation_history: list[list[RunResult]] = [[] for _ in plan.configs]
        return strategies, per_variation_history

    async def _run_repeated_trial(
        self,
        plan: BenchmarkPlan,
        executor: RunExecutor,
        *,
        strategies: list[Any],
        trial: int,
        per_variation_history: list[list[RunResult]],
        all_results: list[RunResult],
        cancel_check: Callable[[], bool] | None,
    ) -> bool:
        """Run all variations for one trial in repeated mode.

        The per-trial body owns the inner variation loop, the cancel/threshold
        checks, and the inter-variation cooldown. Mutates ``all_results`` and
        ``per_variation_history`` in place. Returns True when the caller must
        abort the outer trial loop (cancelled, or sweep failure threshold
        tripped).
        """
        from aiperf.config.resolution.plan import BenchmarkRun

        for var_idx, (cfg, variation) in enumerate(
            zip(plan.configs, plan.variations, strict=True)
        ):
            if cancel_check is not None and cancel_check():
                logger.info(
                    f"Sweep cancelled mid-trial at [v{var_idx} t{trial}]; aborting"
                )
                return True
            strategy = strategies[var_idx]
            next_cfg = strategy.get_next_config(cfg, per_variation_history[var_idx])
            label = strategy.get_run_label(trial)
            artifact_dir = _resolve_artifact_dir(self.base_dir, plan, variation, trial)

            run = BenchmarkRun(
                benchmark_id=executor.derive_id(plan, var_idx, trial),
                sweep_id=plan.sweep_id,
                cfg=next_cfg,
                variation=variation,
                trial=trial,
                label=label,
                artifact_dir=artifact_dir,
                random_seed=resolve_run_seed(plan, variation, trial),
                variables=dict(plan.variables),
            )
            logger.info(f"[v{var_idx} t{trial}] Executing {label}...")
            result = await executor.execute(run)
            self._stamp_variation_metadata(result, run, trial)
            all_results.append(result)
            per_variation_history[var_idx].append(result)
            # Fire the cell callback when this variation has gathered all its
            # trials (under trials-outer/variations-inner this is detectable
            # by ``len(per_variation_history[var_idx]) == plan.trials``).
            # Firing earlier would emit a partial cell; firing only at the
            # last trial keeps a single canonical event per variation.
            if len(per_variation_history[var_idx]) >= plan.trials:
                self._fire_cell_callback(
                    plan, variation, list(per_variation_history[var_idx])
                )

            if self._sweep_failure_threshold_exceeded(all_results, plan):
                logger.warning("Failure threshold exceeded; aborting sweep")
                return True

            if var_idx + 1 < len(plan.configs) and _plan_cooldown_seconds(plan) > 0:
                cooldown = _plan_cooldown_seconds(plan)
                logger.debug(
                    f"Inter-variation cooldown (within trial {trial}): {cooldown}s"
                )
                await asyncio.sleep(cooldown)

        return False

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
            result.variation_index = run.variation.index
        result.trial_index = trial_index
