# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy + convergence construction for cli_runner.

The functions here translate a fully-validated :class:`BenchmarkPlan` into
the runtime objects that drive multi-run execution:

* :func:`build_strategy` - per-cell execution strategy (fixed-trials or
  adaptive convergence) used by ``MultiRunOrchestrator`` to decide when a
  variation's trial loop has run enough trials.
* :func:`_build_convergence_criterion` - the criterion the adaptive
  strategy consults each trial (plugin-dispatched).

The outer-loop adaptive search planner is built by
:func:`aiperf.orchestrator.search_planner.build_search_planner`, which owns
that dispatch outright; nothing here wraps or re-exports it.

:func:`validate_convergence_config` rejects plan configurations the
multi-run path can't honor before any setup work begins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkPlan
    from aiperf.orchestrator.convergence.base import ConvergenceCriterion
    from aiperf.orchestrator.strategies import ExecutionStrategy


def validate_convergence_config(plan: BenchmarkPlan) -> None:
    """Raise ValueError for invalid adaptive/convergence plan configurations."""
    from aiperf.common.enums import ExportLevel
    from aiperf.plugin.enums import ConvergenceCriterionType

    if not plan.use_adaptive:
        return
    if plan.trials <= 1:
        raise ValueError(
            "--convergence-metric requires --num-profile-runs > 1. "
            "Set --num-profile-runs to at least 2 to enable adaptive convergence."
        )
    convergence = plan.multi_run.convergence
    assert convergence is not None  # use_adaptive guards this
    if (
        convergence.mode == ConvergenceCriterionType.DISTRIBUTION
        and plan.export_level == ExportLevel.SUMMARY
    ):
        raise ValueError(
            "--convergence-mode distribution requires per-request JSONL data, "
            "but --export-level is set to 'summary'. "
            "Use --export-level records or --export-level raw."
        )


def build_strategy(plan: BenchmarkPlan, logger: AIPerfLogger) -> ExecutionStrategy:
    """Construct the per-trial execution strategy (adaptive or fixed).

    Called once per config by both ``cli_runner`` (single-trial,
    non-sweep path) and ``MultiRunOrchestrator`` (per-variation). When
    ``plan.is_sweep`` is True (multiple variations), the orchestrator
    invokes this N times for N variations so each cell gets a fresh
    strategy with no convergence state leakage. The returned strategy
    governs only the inner trial loop within a single variation; the
    orchestrator's outer variation loop is owned by
    ``MultiRunOrchestrator``.
    """
    from aiperf.orchestrator.strategies import FixedTrialsStrategy

    if not plan.use_adaptive:
        return FixedTrialsStrategy(
            num_trials=plan.trials,
            cooldown_seconds=plan.cooldown_seconds,
            disable_warmup_after_first=plan.disable_warmup_after_first,
        )

    from aiperf.orchestrator.strategies import AdaptiveStrategy

    criterion = _build_convergence_criterion(plan)

    convergence = plan.multi_run.convergence
    assert convergence is not None  # guaranteed by plan.use_adaptive
    if convergence.min_runs < 3:
        logger.warning(
            f"convergence.min_runs={convergence.min_runs} is below the recommended minimum of 3. "
            "Convergence checks will have reduced statistical power."
        )

    return AdaptiveStrategy(
        criterion=criterion,
        min_runs=convergence.min_runs,
        max_runs=plan.trials,
        cooldown_seconds=plan.cooldown_seconds,
        disable_warmup_after_first=plan.disable_warmup_after_first,
    )


def _build_convergence_criterion(plan: BenchmarkPlan) -> ConvergenceCriterion:
    """Pick the convergence criterion matching ``plan.multi_run.convergence.mode``.

    Dispatches via the plugin registry so third-party criteria (registered in
    `plugins.yaml` under the `convergence_criterion` category) are reachable
    through the same code path as the built-ins. Each criterion class owns the
    mapping from BenchmarkPlan fields to its constructor via `from_plan`.
    """
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    convergence = plan.multi_run.convergence
    assert convergence is not None  # callers must check use_adaptive
    criterion_cls = plugins.get_class(
        PluginType.CONVERGENCE_CRITERION, str(convergence.mode)
    )
    return criterion_cls.from_plan(plan)
