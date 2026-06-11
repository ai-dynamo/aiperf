# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy + convergence + search-planner construction for cli_runner.

The functions here translate a fully-validated :class:`BenchmarkPlan` into
the three runtime objects that drive multi-run execution:

* :func:`build_strategy` - per-cell execution strategy (fixed-trials or
  adaptive convergence) used by ``MultiRunOrchestrator`` to decide when a
  variation's trial loop has run enough trials.
* :func:`_build_convergence_criterion` - the criterion the adaptive
  strategy consults each trial (plugin-dispatched).
* :func:`_build_search_planner` - the outer-loop planner for adaptive
  search sweeps (plugin-dispatched, returns ``None`` outside adaptive search).

:func:`validate_convergence_config` rejects plan configurations the
multi-run path can't honor before any setup work begins.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkPlan
    from aiperf.orchestrator.convergence.base import ConvergenceCriterion
    from aiperf.orchestrator.search_planner.base import SearchPlanner
    from aiperf.orchestrator.strategies import ExecutionStrategy

logger = logging.getLogger(__name__)


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


def _build_search_planner(plan: BenchmarkPlan) -> SearchPlanner | None:
    """Build the outer-loop SearchPlanner for adaptive search.

    Returns None when ``plan.is_adaptive_search`` is False. Dispatches via the
    plugin registry so third-party planners (registered in plugins.yaml under
    the `search_planner` category) are reachable through the same code path
    as the built-in `bayesian` planner.

    When 2+ SLO tiers are configured (via ``--search-sla-tier``), the
    ``MultiTierPlanner`` is instantiated instead of the single-tier planner.
    Single-tier behavior (no ``--search-sla-tier``) remains unchanged.

    The planner class is responsible for raising a clear ImportError if an
    explicitly requested optional sampler is unavailable.
    """
    from aiperf.config.sweep import AdaptiveSearchSweep

    if not isinstance(plan.sweep, AdaptiveSearchSweep):
        return None

    cfg = plan.sweep

    # Multi-tier override: when 2+ tiers are configured, activate the
    # MultiTierPlanner regardless of the underlying planner selection.
    # Single-tier behavior is preserved: existing planners run unmodified.
    if len(cfg.sla_tiers) >= 2:
        from aiperf.orchestrator.search_planner.multi_tier_planner import (
            MultiTierPlanner,
        )
        from aiperf.plugin.enums import SearchPlannerType

        # The search style's *algorithm* is not used by multi-tier (it runs its
        # own bracket/bisection); warn so the user isn't surprised. The style's
        # precision and warmup settings ARE still applied.
        if cfg.planner != SearchPlannerType.SMOOTH_ISOTONIC:
            logger.warning(
                "The search algorithm for --search-style %s is not used when "
                "--search-sla-tier is active; multi-tier uses its own "
                "bracket/bisection method. The style's precision and warmup "
                "settings still apply.",
                cfg.planner,
            )
        return MultiTierPlanner(plan.configs[0], cfg, cfg.sla_tiers)

    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    planner_cls = plugins.get_class(PluginType.SEARCH_PLANNER, str(cfg.planner))
    return planner_cls(plan.configs[0], cfg)
