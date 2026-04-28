# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution strategies for multi-run orchestration."""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.common.config import UserConfig
from aiperf.orchestrator.models import RunResult

if TYPE_CHECKING:
    from aiperf.orchestrator.aggregation.base import AggregateResult
    from aiperf.orchestrator.convergence.base import ConvergenceCriterion

logger = logging.getLogger(__name__)

__all__ = [
    "AdaptiveStrategy",
    "ExecutionStrategy",
    "FixedTrialsStrategy",
    "ParameterSweepStrategy",
    "SweepConfidenceStrategy",
    "SweepMode",
]


def _sanitize_label(label: str) -> str:
    """Sanitize label to prevent path traversal attacks.

    Args:
        label: Raw label string

    Returns:
        Sanitized label safe for filesystem paths
    """
    sanitized = re.sub(r"[/\\]|\.\.", "", label)
    sanitized = re.sub(r'[<>:"|?*]', "", sanitized)
    return sanitized


class SweepMode(str, Enum):
    """Execution order for sweep + confidence composition.

    REPEATED: for trial in trials: for value in values
        All values tested per trial before moving to next trial.
        Path: base_dir/profile_runs/trial_0001/concurrency_10/

    INDEPENDENT: for value in values: for trial in trials
        All trials completed per value before moving to next value.
        Path: base_dir/concurrency_10/profile_runs/trial_0001/
    """

    REPEATED = "repeated"
    INDEPENDENT = "independent"


class ExecutionStrategy(ABC):
    """Base class for execution strategies.

    Strategies decide:
    1. What config to run next (based on results so far)
    2. Whether to continue or stop
    3. How to label runs for artifact organization
    4. Where to store artifacts (path structure)
    5. Cooldown duration between runs
    6. How to aggregate results
    7. How to export aggregates

    Strategies that need custom execution loops (e.g., composed strategies)
    can override execute() to own their iteration. The orchestrator checks
    for this and falls back to its generic loop when execute() returns None.
    """

    def validate_config(self, config: UserConfig) -> None:  # noqa: B027
        """Validate that config is suitable for this strategy.

        Override this method to add strategy-specific validation.
        Called by orchestrator before starting execution.

        Args:
            config: User configuration to validate
        """
        pass

    @abstractmethod
    def should_continue(self, results: list[RunResult]) -> bool:
        """Decide whether to run another trial.

        Args:
            results: Results from runs executed so far

        Returns:
            True if should run another trial, False to stop
        """
        pass

    @abstractmethod
    def get_next_config(
        self, base_config: UserConfig, results: list[RunResult]
    ) -> UserConfig:
        """Generate config for next run.

        Args:
            base_config: Base benchmark configuration
            results: Results from runs executed so far

        Returns:
            Configuration for next run
        """
        pass

    @abstractmethod
    def get_run_label(self, run_index: int) -> str:
        """Generate label for run at given index.

        Args:
            run_index: Zero-based index of run

        Returns:
            Label for run (e.g., "run_0001")
        """
        pass

    @abstractmethod
    def get_cooldown_seconds(self) -> float:
        """Return cooldown duration between runs."""
        pass

    @abstractmethod
    def get_run_path(self, base_dir: Path, run_index: int) -> Path:
        """Build path for a run's artifacts.

        Args:
            base_dir: Base artifact directory (Path)
            run_index: Zero-based run index

        Returns:
            Path where this run's artifacts should be stored
        """
        pass

    @abstractmethod
    def get_aggregate_path(self, base_dir: Path) -> Path:
        """Build path for aggregate artifacts.

        Args:
            base_dir: Base artifact directory (Path)

        Returns:
            Path where aggregate artifacts should be stored
        """
        pass

    @abstractmethod
    def aggregate(
        self, results: list[RunResult], config: UserConfig
    ) -> AggregateResult | None:
        """Compute aggregate statistics from run results.

        Args:
            results: List of run results
            config: User configuration used for execution

        Returns:
            AggregateResult, or None if no aggregation is needed
        """
        pass

    @abstractmethod
    def export_aggregates(self, aggregate: AggregateResult, base_dir: Path) -> None:
        """Export aggregate artifacts to disk.

        Args:
            aggregate: Aggregate result to export
            base_dir: Base artifact directory
        """
        pass

    def tag_result(self, result: RunResult, run_index: int) -> RunResult:
        """Tag a result with execution context metadata.

        Called by the orchestrator after each run. Default is a no-op.
        Strategies that need metadata tagging (e.g., sweep strategies)
        override this to stamp results at creation time instead of
        reverse-engineering from metadata during aggregation.

        Args:
            result: Run result to tag
            run_index: Zero-based index of this run

        Returns:
            The tagged result (may be the same object, mutated)
        """
        return result

    def collect_failed_values(self, results: list[RunResult]) -> list[dict[str, Any]]:
        """Collect information about failed runs.

        Default implementation returns an empty list. Strategies with
        sweep parameters override this to report which values failed.

        Args:
            results: List of all run results

        Returns:
            List of failed value info dicts
        """
        return []

    def execute(
        self,
        config: UserConfig,
        run_fn: Callable[[UserConfig, str, Path], RunResult],
        base_dir: Path,
    ) -> list[RunResult] | None:
        """Optional: run a custom execution loop.

        Strategies that need custom iteration (e.g., composed strategies,
        adaptive strategies) override this to own their loop. The orchestrator
        calls this first; if it returns None, the orchestrator uses its
        generic loop driven by should_continue/get_next_config.

        Args:
            config: Base user configuration
            run_fn: Callback to execute a single run: (config, label, path) -> RunResult
            base_dir: Base artifact directory

        Returns:
            List of RunResult if this strategy owns its loop, or None to
            use the orchestrator's generic loop.
        """
        return None


class FixedTrialsStrategy(ExecutionStrategy):
    """Strategy for fixed number of trials with identical config.

    Used for confidence reporting: run same benchmark N times to quantify variance.

    Attributes:
        num_trials: Number of trials to run
        cooldown_seconds: Sleep duration between trials
        auto_set_seed: Auto-set random seed if not specified
    """

    DEFAULT_SEED = 42

    def __init__(
        self,
        num_trials: int,
        cooldown_seconds: float = 0.0,
        auto_set_seed: bool = True,
        disable_warmup_after_first: bool = True,
    ) -> None:
        """Initialize FixedTrialsStrategy.

        Args:
            num_trials: Number of trials to run (must be between 1 and 10)
            cooldown_seconds: Sleep duration between trials (must be >= 0)
            auto_set_seed: Auto-set random seed if not specified
            disable_warmup_after_first: Disable warmup for runs after the first.

        Raises:
            ValueError: If cooldown_seconds < 0
        """
        if cooldown_seconds < 0:
            raise ValueError(
                f"Invalid cooldown duration: {cooldown_seconds} seconds. "
                f"Cooldown must be non-negative (0 or greater). "
                f"Use 0 for no cooldown, or a positive value like 10 for a 10-second pause between runs."
            )

        self.num_trials = num_trials
        self.cooldown_seconds = cooldown_seconds
        self.auto_set_seed = auto_set_seed
        self.disable_warmup_after_first = disable_warmup_after_first

    def validate_config(self, config: UserConfig) -> None:
        """Validate that config is suitable for this strategy."""
        if (
            self.num_trials > 1
            and config.input.random_seed is None
            and not self.auto_set_seed
        ):
            logger.warning(
                "No random seed specified and auto_set_seed is disabled. "
                "This may result in different workloads across runs, "
                "making confidence statistics less meaningful. "
                "Consider setting --random-seed explicitly."
            )

    def should_continue(self, results: list[RunResult]) -> bool:
        """Continue until we've run num_trials."""
        return len(results) < self.num_trials

    def get_next_config(
        self, base_config: UserConfig, results: list[RunResult]
    ) -> UserConfig:
        """Return config for next trial.

        For first trial: ensure random seed is set (for workload consistency).
        For subsequent trials: optionally disable warmup based on strategy settings.
        """
        config = base_config

        if self.auto_set_seed:
            config = self._ensure_random_seed(config)

        if len(results) > 0 and self.disable_warmup_after_first:
            config = self._disable_warmup(config)

        return config

    def get_run_label(self, run_index: int) -> str:
        """Generate zero-padded label: run_0001, run_0002, etc."""
        label = f"run_{run_index + 1:04d}"
        return _sanitize_label(label)

    def get_cooldown_seconds(self) -> float:
        """Return configured cooldown duration."""
        return self.cooldown_seconds

    def get_run_path(self, base_dir: Path, run_index: int) -> Path:
        """Build path: base_dir/profile_runs/run_NNNN/"""
        base_dir = Path(base_dir)
        label = self.get_run_label(run_index)
        return base_dir / "profile_runs" / label

    def get_aggregate_path(self, base_dir: Path) -> Path:
        """Build path: base_dir/aggregate/"""
        base_dir = Path(base_dir)
        return base_dir / "aggregate"

    def aggregate(
        self, results: list[RunResult], config: UserConfig
    ) -> AggregateResult | None:
        """Compute confidence statistics across trials.

        Args:
            results: List of run results from all trials
            config: User configuration (used for confidence_level)

        Returns:
            AggregateResult with confidence statistics
        """
        from aiperf.orchestrator.aggregation.confidence import ConfidenceAggregation

        successful = [r for r in results if r.success]
        if len(successful) < 2:
            logger.warning(
                f"Insufficient successful runs for confidence aggregation: "
                f"{len(successful)} (need at least 2)"
            )
            return None

        aggregation = ConfidenceAggregation(
            confidence_level=config.loadgen.confidence_level
        )
        result = aggregation.aggregate(results)
        result.metadata["cooldown_seconds"] = (
            config.loadgen.profile_run_cooldown_seconds
        )
        return result

    def export_aggregates(self, aggregate: AggregateResult, base_dir: Path) -> None:
        """Export confidence aggregate to JSON and CSV."""
        from aiperf.orchestrator.export_helpers import export_confidence

        aggregate_dir = self.get_aggregate_path(base_dir)
        export_confidence(aggregate, aggregate_dir)

    def _ensure_random_seed(self, config: UserConfig) -> UserConfig:
        """Ensure config has random seed set."""
        if config.input.random_seed is None:
            logger.info(
                f"No --random-seed specified. Using default seed {self.DEFAULT_SEED} "
                f"for multi-run consistency. All runs will use identical workloads."
            )
            config = config.model_copy(deep=True)
            config.input.random_seed = self.DEFAULT_SEED
        return config

    def _disable_warmup(self, config: UserConfig) -> UserConfig:
        """Create config copy with warmup disabled."""
        config = config.model_copy(deep=True)
        config.loadgen.disable_warmup()
        return config


class AdaptiveStrategy(ExecutionStrategy):
    """Strategy that stops early when a convergence criterion is satisfied.

    Composes with any ConvergenceCriterion to decide when metrics have
    stabilized, bounded by configurable min/max run counts. Delegates
    run labeling, path structure, seed handling, and warmup disabling to
    a FixedTrialsStrategy instance for artifact compatibility.
    """

    DEFAULT_SEED = FixedTrialsStrategy.DEFAULT_SEED

    def __init__(
        self,
        criterion: ConvergenceCriterion,
        min_runs: int = 3,
        max_runs: int = 10,
        cooldown_seconds: float = 0.0,
        auto_set_seed: bool = True,
        disable_warmup_after_first: bool = True,
    ) -> None:
        if min_runs < 1:
            raise ValueError(f"Invalid min_runs: {min_runs}. Must be at least 1.")
        if max_runs < min_runs:
            raise ValueError(
                f"Invalid max_runs: {max_runs}. Must be >= min_runs ({min_runs})."
            )

        self.criterion = criterion
        self.min_runs = min_runs
        self.max_runs = max_runs
        self._delegate = FixedTrialsStrategy(
            num_trials=max_runs,
            cooldown_seconds=cooldown_seconds,
            auto_set_seed=auto_set_seed,
            disable_warmup_after_first=disable_warmup_after_first,
        )

    @property
    def cooldown_seconds(self) -> float:
        return self._delegate.cooldown_seconds

    @property
    def auto_set_seed(self) -> bool:
        return self._delegate.auto_set_seed

    @property
    def disable_warmup_after_first(self) -> bool:
        return self._delegate.disable_warmup_after_first

    def should_continue(self, results: list[RunResult]) -> bool:
        """Continue unless max reached or criterion converged (after min)."""
        n = len(results)
        if n >= self.max_runs:
            return False
        if n < self.min_runs:
            return True
        try:
            converged = self.criterion.is_converged(results)
        except Exception:
            logger.exception(
                "Convergence criterion raised an error; treating as not converged"
            )
            converged = False
        return not converged

    def get_next_config(
        self, base_config: UserConfig, results: list[RunResult]
    ) -> UserConfig:
        """Return config for next run, delegating to FixedTrialsStrategy."""
        return self._delegate.get_next_config(base_config, results)

    def get_run_label(self, run_index: int) -> str:
        """Generate zero-padded label matching FixedTrialsStrategy: run_0001, etc."""
        return self._delegate.get_run_label(run_index)

    def get_cooldown_seconds(self) -> float:
        """Return configured cooldown duration."""
        return self._delegate.get_cooldown_seconds()

    def get_run_path(self, base_dir: Path, run_index: int) -> Path:
        """Build path for a run's artifacts: base_dir/profile_runs/run_NNNN/."""
        return self._delegate.get_run_path(base_dir, run_index)

    def get_aggregate_path(self, base_dir: Path) -> Path:
        """Build path for aggregate artifacts: base_dir/aggregate/."""
        return self._delegate.get_aggregate_path(base_dir)

    def aggregate(
        self, results: list[RunResult], config: UserConfig
    ) -> AggregateResult | None:
        """Compute confidence aggregation and detailed (collated) aggregation.

        Detailed aggregation pools per-request JSONL data across runs for
        combined percentiles. Stored in metadata for export_aggregates().
        """
        confidence_agg = self._delegate.aggregate(results, config)
        if confidence_agg is None:
            return None

        # Compute detailed aggregation from per-request JSONL data
        from aiperf.orchestrator.aggregation.detailed import DetailedAggregation

        detailed_aggregation = DetailedAggregation()
        detailed_result = detailed_aggregation.aggregate(results)
        detailed_result.metadata["cooldown_seconds"] = (
            config.loadgen.profile_run_cooldown_seconds
        )
        confidence_agg.metadata["_detailed_result"] = detailed_result

        return confidence_agg

    def export_aggregates(self, aggregate: AggregateResult, base_dir: Path) -> None:
        """Export confidence aggregate and detailed/collated aggregate."""
        from aiperf.orchestrator.export_helpers import (
            export_confidence,
            export_detailed,
        )

        aggregate_dir = self.get_aggregate_path(base_dir)
        export_confidence(aggregate, aggregate_dir)

        detailed_result = aggregate.metadata.pop("_detailed_result", None)
        if detailed_result is not None:
            export_detailed(detailed_result, aggregate_dir)


class ParameterSweepStrategy(ExecutionStrategy):
    """Strategy for sweeping a single parameter across multiple values.

    This strategy is COMPLETELY INDEPENDENT - it knows nothing about confidence
    trials or FixedTrialsStrategy. It ONLY handles varying a parameter.

    Attributes:
        parameter_name: Name of parameter to sweep (e.g., "concurrency")
        parameter_values: List of values to test
        cooldown_seconds: Cooldown between parameter values
        same_seed: Use same seed for all values (default: derive different seeds)
        auto_set_seed: Auto-set base seed if not specified
        base_seed: Base seed for derivation (set on first get_next_config call)
    """

    DEFAULT_SEED = 42

    def __init__(
        self,
        parameter_name: str,
        parameter_values: list[int],
        cooldown_seconds: float = 0.0,
        same_seed: bool = False,
        auto_set_seed: bool = True,
    ) -> None:
        """Initialize parameter sweep strategy.

        Args:
            parameter_name: Name of parameter to sweep (e.g., "concurrency")
            parameter_values: List of values to test
            cooldown_seconds: Cooldown between parameter values (must be >= 0)
            same_seed: Use same seed for all values
            auto_set_seed: Auto-set base seed if not specified

        Raises:
            ValueError: If cooldown_seconds < 0 or parameter_values is empty
        """
        if cooldown_seconds < 0:
            raise ValueError(
                f"Invalid cooldown duration: {cooldown_seconds} seconds. "
                f"Cooldown must be non-negative (0 or greater). "
                f"Use 0 for no cooldown, or a positive value like 10 for a 10-second pause between parameter values."
            )
        if not parameter_values:
            raise ValueError(
                "Parameter sweep requires at least one value to test. "
                "Provide a comma-separated list of values: --concurrency 10,20,30. "
                "For a single value, use: --concurrency 10 (no comma)."
            )

        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.cooldown_seconds = cooldown_seconds
        self.same_seed = same_seed
        self.auto_set_seed = auto_set_seed
        self.base_seed: int | None = None

    def should_continue(self, results: list[RunResult]) -> bool:
        """Continue until all parameter values are tested."""
        return len(results) < len(self.parameter_values)

    def get_next_config(
        self, base_config: UserConfig, results: list[RunResult]
    ) -> UserConfig:
        """Generate config for next parameter value.

        Sets the parameter value and derives appropriate random seed.
        """
        value_index = len(results)
        value = self.parameter_values[value_index]

        config = base_config.model_copy(deep=True)
        setattr(config.loadgen, self.parameter_name, value)
        config.loadgen.parameter_sweep_mode = "repeated"
        config.loadgen.parameter_sweep_cooldown_seconds = 0.0
        config.loadgen.parameter_sweep_same_seed = False
        config.loadgen.model_fields_set.discard("parameter_sweep_mode")
        config.loadgen.model_fields_set.discard("parameter_sweep_cooldown_seconds")
        config.loadgen.model_fields_set.discard("parameter_sweep_same_seed")

        if self.base_seed is None:
            if config.input.random_seed is not None:
                self.base_seed = config.input.random_seed
            elif self.auto_set_seed:
                self.base_seed = self.DEFAULT_SEED
                logger.info(
                    f"No --random-seed specified. Using default seed {self.DEFAULT_SEED} "
                    f"for parameter sweep consistency."
                )

        if self.base_seed is not None:
            if self.same_seed:
                seed = self.base_seed
                if value_index == 0:
                    logger.info(
                        f"Using same seed ({seed}) across all sweep values "
                        f"(--parameter-sweep-same-seed enabled)."
                    )
            else:
                seed = self.base_seed + value_index
                if value_index == 0:
                    logger.info(
                        f"Deriving different seeds per sweep value from base seed {self.base_seed}."
                    )
            config.input.random_seed = seed

        return config

    def get_run_label(self, run_index: int) -> str:
        """Generate label: concurrency_10, concurrency_20, etc."""
        value = self.parameter_values[run_index]
        label = f"{self.parameter_name}_{value}"
        return _sanitize_label(label)

    def get_cooldown_seconds(self) -> float:
        """Return cooldown between parameter values."""
        return self.cooldown_seconds

    def get_run_path(self, base_dir: Path, run_index: int) -> Path:
        """Build path: base_dir/concurrency_10/"""
        base_dir = Path(base_dir)
        label = self.get_run_label(run_index)
        return base_dir / label

    def get_aggregate_path(self, base_dir: Path) -> Path:
        """Build path: base_dir/sweep_aggregate/"""
        base_dir = Path(base_dir)
        return base_dir / "sweep_aggregate"

    def tag_result(self, result: RunResult, run_index: int) -> RunResult:
        """Tag result with sweep parameter metadata."""
        value = self.parameter_values[run_index]
        result.metadata.update(
            {
                "value_index": run_index,
                self.parameter_name: value,
            }
        )
        return result

    def collect_failed_values(self, results: list[RunResult]) -> list[dict[str, Any]]:
        """Collect information about failed sweep values."""
        failed = []
        seen: set[tuple[str, Any]] = set()
        for result in results:
            if not result.success and self.parameter_name in result.metadata:
                value = result.metadata[self.parameter_name]
                key = (self.parameter_name, value)
                if key not in seen:
                    seen.add(key)
                    failed.append(
                        {
                            "value": value,
                            "parameter_name": self.parameter_name,
                            "error": result.error or "Unknown error",
                        }
                    )
        return failed

    def aggregate(
        self, results: list[RunResult], config: UserConfig
    ) -> AggregateResult | None:
        """Compute sweep-level aggregation (Pareto + best configs) from single-run metrics.

        For sweep-only mode, each value has exactly one run. Metrics are extracted
        directly from summary_metrics without confidence wrapping.

        Args:
            results: List of run results (one per sweep value)
            config: User configuration

        Returns:
            AggregateResult with sweep statistics, or None if insufficient data
        """
        from aiperf.orchestrator.aggregation.base import AggregateResult
        from aiperf.orchestrator.aggregation.sweep import (
            SweepAnalyzer,
        )

        per_combination_stats = self._build_per_combination_stats(results)
        if not per_combination_stats:
            logger.warning("No successful sweep results to aggregate")
            return None

        sweep_parameters = [
            {"name": self.parameter_name, "values": sorted(self.parameter_values)}
        ]
        sweep_dict = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        total_runs = len(results)
        successful_runs = len([r for r in results if r.success])
        failed_run_details = [
            {"label": r.label, "error": str(r.error) if r.error else "Unknown error"}
            for r in results
            if not r.success
        ]

        sweep_dict["metadata"]["aggregation_type"] = "sweep"

        sweep_result = AggregateResult(
            aggregation_type="sweep",
            num_runs=total_runs,
            num_successful_runs=successful_runs,
            failed_runs=failed_run_details,
            metadata=sweep_dict["metadata"],
            metrics=sweep_dict["per_combination_metrics"],
        )
        sweep_result.metadata["best_configurations"] = sweep_dict["best_configurations"]
        sweep_result.metadata["pareto_optimal"] = sweep_dict["pareto_optimal"]

        return sweep_result

    def export_aggregates(self, aggregate: AggregateResult, base_dir: Path) -> None:
        """Export sweep aggregate to JSON and CSV."""
        from aiperf.orchestrator.export_helpers import export_sweep

        sweep_dir = self.get_aggregate_path(base_dir)
        export_sweep(aggregate, sweep_dir)

        best_configs = aggregate.metadata.get("best_configurations", {})
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

        pareto_optimal = aggregate.metadata.get("pareto_optimal", [])
        if pareto_optimal:
            logger.info(f"  Pareto optimal points: {pareto_optimal}")

    def _build_per_combination_stats(self, results: list[RunResult]) -> dict[Any, dict]:
        """Convert single-run JsonMetricResult into SweepAnalyzer input format.

        For sweep-only, each value has exactly one run. Metrics are flattened to
        ``{metric_name}_{stat_key}`` so the keys match what confidence aggregation
        produces and what SweepAnalyzer.compute() expects.
        """
        from aiperf.common.constants import STAT_KEYS
        from aiperf.orchestrator.aggregation.sweep import ParameterCombination

        stats: dict[Any, dict] = {}
        for result in results:
            if not result.success:
                continue
            if self.parameter_name not in result.metadata:
                continue
            value = result.metadata[self.parameter_name]
            coord = ParameterCombination({self.parameter_name: value})
            metrics_dict: dict[str, dict] = {}
            for metric_name, metric_result in result.summary_metrics.items():
                for stat_key in STAT_KEYS:
                    stat_value = getattr(metric_result, stat_key, None)
                    if stat_value is not None:
                        flattened_key = f"{metric_name}_{stat_key}"
                        metrics_dict[flattened_key] = {
                            "mean": stat_value,
                            "unit": metric_result.unit,
                        }
            stats[coord] = metrics_dict
        return stats


class SweepConfidenceStrategy(ExecutionStrategy):
    """Strategy for parameter sweep with confidence trials at each value.

    Composes ParameterSweepStrategy and FixedTrialsStrategy. Owns the
    nested execution loop (Option B), per-value confidence aggregation,
    and cross-value sweep aggregation.

    This strategy overrides execute() to run its own nested loops with
    proper cooldown logic (sweep cooldown between values, trial cooldown
    between trials within a value). The orchestrator delegates entirely
    to this strategy's execute() method.

    Attributes:
        sweep: The parameter sweep strategy
        confidence: The fixed trials strategy
        mode: Execution order (REPEATED or INDEPENDENT)
    """

    def __init__(
        self,
        sweep: ParameterSweepStrategy,
        confidence: FixedTrialsStrategy,
        mode: SweepMode,
    ) -> None:
        self.sweep = sweep
        self.confidence = confidence
        self.mode = mode

    # Required by the ABC but won't be called when execute() is used.

    def should_continue(self, results: list[RunResult]) -> bool:
        """Not used — execute() owns the loop."""
        total = len(self.sweep.parameter_values) * self.confidence.num_trials
        return len(results) < total

    def get_next_config(
        self, base_config: UserConfig, results: list[RunResult]
    ) -> UserConfig:
        """Not used — execute() owns the loop."""
        raise NotImplementedError(
            "SweepConfidenceStrategy uses execute() for custom iteration"
        )

    def get_run_label(self, run_index: int) -> str:
        """Not used — execute() owns the loop."""
        return f"run_{run_index + 1:04d}"

    def get_cooldown_seconds(self) -> float:
        """Not used — execute() handles cooldowns internally."""
        return 0.0

    def get_run_path(self, base_dir: Path, run_index: int) -> Path:
        """Not used — execute() builds paths from inner strategies."""
        return Path(base_dir) / f"run_{run_index + 1:04d}"

    def get_aggregate_path(self, base_dir: Path) -> Path:
        """Build path for sweep-level aggregate."""
        if self.mode == SweepMode.REPEATED:
            return Path(base_dir) / "aggregate" / "sweep_aggregate"
        else:
            return Path(base_dir) / "sweep_aggregate"

    def execute(
        self,
        config: UserConfig,
        run_fn: Callable[[UserConfig, str, Path], RunResult],
        base_dir: Path,
    ) -> list[RunResult] | None:
        """Execute nested sweep + confidence loops.

        The loop order depends on self.mode:
        - REPEATED:    for trial in trials: for value in values
        - INDEPENDENT: for value in values: for trial in trials

        Args:
            config: Base user configuration
            run_fn: Callback to execute a single run
            base_dir: Base artifact directory

        Returns:
            List of all RunResult
        """
        self.sweep.validate_config(config)
        self.confidence.validate_config(config)

        if self.mode == SweepMode.REPEATED:
            return self._execute_trials_then_sweep(config, run_fn, base_dir)
        else:
            return self._execute_sweep_then_trials(config, run_fn, base_dir)

    def _execute_trials_then_sweep(
        self,
        config: UserConfig,
        run_fn: Callable[[UserConfig, str, Path], RunResult],
        base_dir: Path,
    ) -> list[RunResult]:
        """Repeated mode: for trial in trials: for value in values.

        Path: base_dir/profile_runs/trial_0001/concurrency_10/
        """
        all_results: list[RunResult] = []
        trial_results: list[RunResult] = []

        n_values = len(self.sweep.parameter_values)
        n_trials = self.confidence.num_trials

        logger.info(
            f"Starting repeated mode: {n_trials} trials × {n_values} sweep values"
        )

        while self.confidence.should_continue(trial_results):
            trial_index = len(trial_results)
            trial_config = self.confidence.get_next_config(config, trial_results)
            trial_label = _sanitize_label(f"trial_{trial_index + 1:04d}")
            trial_dir = Path(base_dir) / "profile_runs" / trial_label

            logger.info(f"[Trial {trial_index + 1}/{n_trials}] Starting {trial_label}")

            sweep_results: list[RunResult] = []
            while self.sweep.should_continue(sweep_results):
                value_index = len(sweep_results)
                run_config = self.sweep.get_next_config(trial_config, sweep_results)
                run_path = self.sweep.get_run_path(trial_dir, value_index)
                sweep_label = self.sweep.get_run_label(value_index)
                label = f"{trial_label}_{sweep_label}"

                logger.info(
                    f"  [{value_index + 1}/{n_values}] Executing {sweep_label}..."
                )

                result = run_fn(run_config, label, run_path)
                # Tag at creation time
                result.metadata.update(
                    {
                        "trial_index": trial_index,
                        "value_index": value_index,
                        self.sweep.parameter_name: getattr(
                            run_config.loadgen, self.sweep.parameter_name
                        ),
                        "sweep_mode": self.mode.value,
                    }
                )
                sweep_results.append(result)
                all_results.append(result)

                if result.success:
                    logger.info(f"  [{value_index + 1}] {sweep_label} completed")
                else:
                    logger.error(
                        f"  [{value_index + 1}] {sweep_label} failed: {result.error}"
                    )

                # Sweep cooldown between values within a trial
                if self.sweep.should_continue(sweep_results):
                    cooldown = self.sweep.get_cooldown_seconds()
                    if cooldown > 0:
                        logger.info(f"  Applying sweep cooldown: {cooldown}s")
                        time.sleep(cooldown)

            trial_results.append(sweep_results[-1])  # Track trial completion

            # Trial cooldown between trials
            if self.confidence.should_continue(trial_results):
                cooldown = self.confidence.get_cooldown_seconds()
                if cooldown > 0:
                    logger.info(f"Applying trial cooldown: {cooldown}s")
                    time.sleep(cooldown)

        self._log_completion(all_results)
        return all_results

    def _execute_sweep_then_trials(
        self,
        config: UserConfig,
        run_fn: Callable[[UserConfig, str, Path], RunResult],
        base_dir: Path,
    ) -> list[RunResult]:
        """Independent mode: for value in values: for trial in trials.

        Path: base_dir/concurrency_10/profile_runs/trial_0001/
        """
        all_results: list[RunResult] = []
        sweep_results: list[RunResult] = []

        n_values = len(self.sweep.parameter_values)
        n_trials = self.confidence.num_trials

        logger.info(
            f"Starting independent mode: {n_values} sweep values × {n_trials} trials"
        )

        while self.sweep.should_continue(sweep_results):
            value_index = len(sweep_results)
            value_config = self.sweep.get_next_config(config, sweep_results)
            sweep_dir = self.sweep.get_run_path(base_dir, value_index)
            sweep_label = self.sweep.get_run_label(value_index)

            logger.info(f"[Value {value_index + 1}/{n_values}] Starting {sweep_label}")

            trial_results: list[RunResult] = []
            while self.confidence.should_continue(trial_results):
                trial_index = len(trial_results)
                run_config = self.confidence.get_next_config(
                    value_config, trial_results
                )
                trial_label = _sanitize_label(f"trial_{trial_index + 1:04d}")
                run_path = Path(sweep_dir) / "profile_runs" / trial_label
                label = f"{sweep_label}_{trial_label}"

                logger.info(
                    f"  [{trial_index + 1}/{n_trials}] Executing {trial_label}..."
                )

                result = run_fn(run_config, label, run_path)
                result.metadata.update(
                    {
                        "trial_index": trial_index,
                        "value_index": value_index,
                        self.sweep.parameter_name: getattr(
                            run_config.loadgen, self.sweep.parameter_name
                        ),
                        "sweep_mode": self.mode.value,
                    }
                )
                trial_results.append(result)
                all_results.append(result)

                if result.success:
                    logger.info(f"  [{trial_index + 1}] {trial_label} completed")
                else:
                    logger.error(
                        f"  [{trial_index + 1}] {trial_label} failed: {result.error}"
                    )

                # Trial cooldown between trials within a value
                if self.confidence.should_continue(trial_results):
                    cooldown = self.confidence.get_cooldown_seconds()
                    if cooldown > 0:
                        logger.info(f"  Applying trial cooldown: {cooldown}s")
                        time.sleep(cooldown)

            sweep_results.append(trial_results[-1])  # Track sweep completion

            # Sweep cooldown between values
            if self.sweep.should_continue(sweep_results):
                cooldown = self.sweep.get_cooldown_seconds()
                if cooldown > 0:
                    logger.info(f"Applying sweep cooldown: {cooldown}s")
                    time.sleep(cooldown)

        self._log_completion(all_results)
        return all_results

    def _log_completion(self, results: list[RunResult]) -> None:
        """Log completion summary and failed values."""
        successful = sum(1 for r in results if r.success)
        mode_name = "Repeated" if self.mode == SweepMode.REPEATED else "Independent"
        logger.info(
            f"{mode_name} mode complete: {successful}/{len(results)} runs successful"
        )

        failed = self.collect_failed_values(results)
        if failed:
            logger.warning(
                f"Some sweep values failed: {[fv['value'] for fv in failed]}"
            )
            for fv in failed:
                logger.warning(f"  {fv['parameter_name']}={fv['value']}: {fv['error']}")

    def aggregate(
        self, results: list[RunResult], config: UserConfig
    ) -> AggregateResult | None:
        """Compute per-value confidence aggregates + cross-value sweep aggregate.

        Steps:
        1. Group results by parameter value
        2. Per-value confidence aggregation
        3. Sweep aggregation over confidence means

        Args:
            results: List of all run results
            config: User configuration

        Returns:
            AggregateResult with sweep statistics and per-value aggregates in metadata
        """
        from dataclasses import asdict, is_dataclass

        from aiperf.orchestrator.aggregation.base import AggregateResult
        from aiperf.orchestrator.aggregation.confidence import ConfidenceAggregation
        from aiperf.orchestrator.aggregation.sweep import (
            ParameterCombination,
            SweepAnalyzer,
        )

        confidence_level = config.loadgen.confidence_level

        results_by_value: dict[Any, list[RunResult]] = defaultdict(list)
        for result in results:
            if self.sweep.parameter_name in result.metadata:
                value = result.metadata[self.sweep.parameter_name]
                results_by_value[value].append(result)

        if not results_by_value:
            logger.warning("No results with sweep metadata found")
            return None

        aggregation = ConfidenceAggregation(confidence_level=confidence_level)
        per_value_aggregates: dict[Any, AggregateResult] = {}

        for value in sorted(results_by_value.keys()):
            value_results = results_by_value[value]
            successful = [r for r in value_results if r.success]
            if len(successful) < 2:
                logger.warning(
                    f"Skipping aggregate for {self.sweep.parameter_name}={value}: "
                    f"only {len(successful)} successful run(s), need at least 2"
                )
                continue

            logger.info(
                f"  Aggregating {self.sweep.parameter_name}={value} "
                f"({len(successful)}/{len(value_results)} successful)"
            )
            agg = aggregation.aggregate(value_results)
            agg.metadata[self.sweep.parameter_name] = value
            agg.metadata["sweep_mode"] = self.mode.value
            per_value_aggregates[value] = agg

        if not per_value_aggregates:
            logger.warning("No values had enough successful runs for aggregation")
            return None

        per_combination_stats: dict[Any, dict] = {}
        for value, agg in per_value_aggregates.items():
            coord = ParameterCombination({self.sweep.parameter_name: value})
            metrics_dict = {}
            for metric_name, metric_value in agg.metrics.items():
                if is_dataclass(metric_value):
                    metrics_dict[metric_name] = asdict(metric_value)
                elif isinstance(metric_value, dict):
                    metrics_dict[metric_name] = metric_value
                else:
                    metrics_dict[metric_name] = (
                        dict(metric_value)
                        if hasattr(metric_value, "__iter__")
                        else metric_value
                    )
            per_combination_stats[coord] = metrics_dict

        sweep_parameters = [
            {
                "name": self.sweep.parameter_name,
                "values": sorted(self.sweep.parameter_values),
            }
        ]
        sweep_dict = SweepAnalyzer.compute(per_combination_stats, sweep_parameters)

        total_runs = len(results)
        successful_runs = len([r for r in results if r.success])
        failed_run_details = [
            {"label": r.label, "error": str(r.error) if r.error else "Unknown error"}
            for r in results
            if not r.success
        ]

        sweep_dict["metadata"]["sweep_mode"] = self.mode.value
        sweep_dict["metadata"]["confidence_level"] = confidence_level
        sweep_dict["metadata"]["aggregation_type"] = "sweep"

        if results_by_value:
            first_value = sorted(results_by_value.keys())[0]
            sweep_dict["metadata"]["num_trials_per_value"] = len(
                results_by_value[first_value]
            )

        sweep_result = AggregateResult(
            aggregation_type="sweep",
            num_runs=total_runs,
            num_successful_runs=successful_runs,
            failed_runs=failed_run_details,
            metadata=sweep_dict["metadata"],
            metrics=sweep_dict["per_combination_metrics"],
        )
        sweep_result.metadata["best_configurations"] = sweep_dict["best_configurations"]
        sweep_result.metadata["pareto_optimal"] = sweep_dict["pareto_optimal"]
        sweep_result.metadata["per_value_aggregates"] = per_value_aggregates

        return sweep_result

    def export_aggregates(self, aggregate: AggregateResult, base_dir: Path) -> None:
        """Export per-value confidence aggregates and sweep-level aggregate.

        Args:
            aggregate: Sweep aggregate result (with per_value_aggregates in metadata)
            base_dir: Base artifact directory
        """
        from aiperf.orchestrator.export_helpers import export_confidence, export_sweep

        per_value_aggregates = aggregate.metadata.get("per_value_aggregates", {})

        for value, conf_aggregate in per_value_aggregates.items():
            if self.mode == SweepMode.REPEATED:
                agg_dir = (
                    Path(base_dir)
                    / "aggregate"
                    / f"{self.sweep.parameter_name}_{value}"
                )
            else:
                agg_dir = (
                    Path(base_dir)
                    / f"{self.sweep.parameter_name}_{value}"
                    / "aggregate"
                )
            export_confidence(conf_aggregate, agg_dir)

        sweep_dir = self.get_aggregate_path(base_dir)
        export_sweep(aggregate, sweep_dir)

        best_configs = aggregate.metadata.get("best_configurations", {})
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

        pareto_optimal = aggregate.metadata.get("pareto_optimal", [])
        if pareto_optimal:
            logger.info(f"  Pareto optimal points: {pareto_optimal}")

    def collect_failed_values(self, results: list[RunResult]) -> list[dict[str, Any]]:
        """Collect failed sweep values using the sweep strategy's parameter name."""
        failed = []
        seen: set[tuple[str, Any]] = set()
        for result in results:
            if not result.success and self.sweep.parameter_name in result.metadata:
                value = result.metadata[self.sweep.parameter_name]
                key = (self.sweep.parameter_name, value)
                if key not in seen:
                    seen.add(key)
                    failed.append(
                        {
                            "value": value,
                            "parameter_name": self.sweep.parameter_name,
                            "error": result.error or "Unknown error",
                        }
                    )
        return failed
