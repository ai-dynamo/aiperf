# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution strategies for multi-run orchestration."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.enums import SweepMode  # re-exported for back-compat
from aiperf.config.config import BenchmarkConfig
from aiperf.orchestrator.models import RunResult

if TYPE_CHECKING:
    from aiperf.orchestrator.convergence.base import ConvergenceCriterion

logger = logging.getLogger(__name__)

__all__ = [
    "AdaptiveStrategy",
    "ExecutionStrategy",
    "FixedTrialsStrategy",
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


class ExecutionStrategy(ABC):
    """Base class for execution strategies.

    Strategies decide:
    1. What config to run next (based on results so far)
    2. Whether to continue or stop
    3. How to label runs for artifact organization
    4. Where to store artifacts (path structure)
    5. Cooldown duration between runs
    """

    def validate_config(self, config: BenchmarkConfig) -> None:  # noqa: B027
        """Validate that config is suitable for this strategy.

        Override this method to add strategy-specific validation.
        Called by orchestrator before starting execution.

        Args:
            config: Benchmark configuration to validate
        """
        # Default implementation: no validation required
        # Subclasses can override to add strategy-specific validation
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
        self, base_config: BenchmarkConfig, results: list[RunResult]
    ) -> BenchmarkConfig:
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

        Concrete strategies always emit a flat ``profile_runs/run_NNNN/``
        layout under ``base_dir``. Per-variation hierarchy (e.g.
        ``concurrency_10/...``) is handled upstream by
        ``MultiRunOrchestrator._resolve_artifact_dir``, which passes the
        per-variation ``base_dir`` in.

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

        Concrete strategies always return ``base_dir/aggregate``. Per-variation
        aggregate dirs are computed upstream of the strategy by
        ``_per_variation_aggregate_dir`` in the sweep helpers.

        Args:
            base_dir: Base artifact directory (Path)

        Returns:
            Path where aggregate artifacts should be stored
        """
        pass


class FixedTrialsStrategy(ExecutionStrategy):
    """Strategy for fixed number of trials with identical config.

    Used for confidence reporting: run same benchmark N times to quantify variance.

    Attributes:
        num_trials: Number of trials to run
        cooldown_seconds: Sleep duration between trials
    """

    def __init__(
        self,
        num_trials: int,
        *,
        cooldown_seconds: float = 0.0,
        disable_warmup_after_first: bool = True,
    ) -> None:
        """Initialize FixedTrialsStrategy.

        Args:
            num_trials: Number of trials to run (must be between 1 and 10)
            cooldown_seconds: Sleep duration between trials (must be >= 0)
            disable_warmup_after_first: Disable warmup for runs after the first.
                If True (default), only the first run includes warmup, subsequent
                runs measure steady-state performance. If False, all runs include
                warmup (useful for long cooldown periods or cold-start scenarios).

        Raises:
            ValueError: If cooldown_seconds < 0

        Note:
            num_trials validation is handled by Pydantic at the config level.
        """
        if cooldown_seconds < 0:
            raise ValueError(
                f"Invalid cooldown_seconds: {cooldown_seconds}. Must be non-negative."
            )

        self.num_trials = num_trials
        self.cooldown_seconds = cooldown_seconds
        self.disable_warmup_after_first = disable_warmup_after_first

    def validate_config(self, config: BenchmarkConfig) -> None:
        """Validate that config is suitable for this strategy.

        For FixedTrialsStrategy, seed handling moved to the envelope/plan
        layer (`AIPerfConfig.random_seed` -> `BenchmarkPlan.variation_seeds`).
        Strategies no longer mutate per-config seeds; the orchestrator
        threads `plan.variation_seeds[i]` into each `BenchmarkRun`.

        Args:
            config: Benchmark configuration to validate
        """
        # No-op: kept for subclass override surface.
        return

    def should_continue(self, results: list[RunResult]) -> bool:
        """Continue until we've run num_trials."""
        return len(results) < self.num_trials

    def get_next_config(
        self, base_config: BenchmarkConfig, results: list[RunResult]
    ) -> BenchmarkConfig:
        """Return config for next trial.

        For subsequent trials: optionally disable warmup based on strategy settings.
        Seed handling moved to BenchmarkPlan.variation_seeds (envelope-level);
        the orchestrator threads it into BenchmarkRun.random_seed at construction.

        Warmup behavior is controlled by disable_warmup_after_first:
        - True (default): Only first run has warmup, subsequent runs measure steady-state
        - False: All runs include warmup (useful for long cooldowns or cold-start testing)
        """
        config = base_config

        # Subsequent trials: optionally disable warmup
        if len(results) > 0 and self.disable_warmup_after_first:
            config = self._disable_warmup(config)

        return config

    def get_run_label(self, run_index: int) -> str:
        """Generate zero-padded label: run_0001, run_0002, etc.

        Args:
            run_index: Zero-based index of run

        Returns:
            Sanitized label safe for filesystem paths
        """
        label = f"run_{run_index + 1:04d}"
        # Sanitize label to prevent path traversal
        return _sanitize_label(label)

    def get_cooldown_seconds(self) -> float:
        """Return configured cooldown duration."""
        return self.cooldown_seconds

    def get_run_path(self, base_dir: Path, run_index: int) -> Path:
        """Build path for a run's artifacts.

        For fixed trials, uses flat structure: base_dir/profile_runs/run_NNNN/

        Directory Structure Example:
        When base_dir is the auto-generated artifact directory:
        artifacts/llama-3-8b-openai-chat-concurrency_10/profile_runs/run_0001/
        artifacts/llama-3-8b-openai-chat-concurrency_10/profile_runs/run_0002/
        artifacts/llama-3-8b-openai-chat-concurrency_10/aggregate/

        Args:
            base_dir: Base artifact directory (Path)
            run_index: Zero-based run index

        Returns:
            Path where this run's artifacts should be stored
        """
        base_dir = Path(base_dir)
        label = self.get_run_label(run_index)
        return base_dir / "profile_runs" / label

    def get_aggregate_path(self, base_dir: Path) -> Path:
        """Build path for aggregate artifacts.

        For fixed trials, uses single aggregate directory: base_dir/aggregate/

        Directory Structure Example:
        artifacts/llama-3-8b-openai-chat-concurrency_10/aggregate/

        This directory contains the aggregated results across all runs:
        - profile_export_aiperf_aggregate.json
        - profile_export_aiperf_aggregate.csv

        Args:
            base_dir: Base artifact directory (Path)

        Returns:
            Path where aggregate artifacts should be stored
        """
        base_dir = Path(base_dir)
        return base_dir / "aggregate"

    def _disable_warmup(self, config: BenchmarkConfig) -> BenchmarkConfig:
        """Create config copy with warmup disabled.

        This is called for subsequent trials when disable_warmup_after_first=True.
        Disabling warmup after the first trial provides more accurate aggregate
        statistics by measuring steady-state performance without warmup overhead.

        However, users may want warmup on all runs (disable_warmup_after_first=False)
        for scenarios like:
        - Long cooldown periods where system returns to cold state
        - Testing cold-start performance explicitly
        - Ensuring consistent conditions across all runs

        Removes phases that have exclude_from_results=True (warmup phases).

        Args:
            config: Original configuration

        Returns:
            Configuration with warmup phases removed
        """
        config = config.model_copy(deep=True)
        config.phases = [p for p in config.phases if not p.exclude_from_results]
        return config


class AdaptiveStrategy(ExecutionStrategy):
    """Strategy that stops early when a convergence criterion is satisfied.

    Composes with any ConvergenceCriterion to decide when metrics have
    stabilized, bounded by configurable min/max run counts. Delegates
    run labeling, path structure, seed handling, and warmup disabling to
    a FixedTrialsStrategy instance for artifact compatibility.
    """

    def __init__(
        self,
        criterion: ConvergenceCriterion,
        *,
        min_runs: int = 3,
        max_runs: int = 10,
        cooldown_seconds: float = 0.0,
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
            disable_warmup_after_first=disable_warmup_after_first,
        )

    @property
    def cooldown_seconds(self) -> float:
        return self._delegate.cooldown_seconds

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
        self, base_config: BenchmarkConfig, results: list[RunResult]
    ) -> BenchmarkConfig:
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
