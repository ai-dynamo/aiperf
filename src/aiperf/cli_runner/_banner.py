# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-run banner logging for cli_runner.

These functions emit the human-readable header block that runs before the
multi-run benchmark begins. Post-run aggregate summary lives in
:mod:`aiperf.cli_runner._aggregate`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkPlan
    from aiperf.orchestrator.search_planner.base import SearchPlanner


def log_multi_run_banner(
    plan: BenchmarkPlan, total_runs: int, logger: AIPerfLogger
) -> None:
    """Emit the banner describing a multi-run benchmark's configuration."""
    from aiperf.plugin.enums import ConvergenceCriterionType

    logger.info("=" * 80)
    logger.info("Starting Multi-Run Benchmark")
    logger.info(f"  Configurations: {len(plan.configs)}")
    logger.info(f"  Trials per config: {plan.trials}")
    logger.info(f"  Total runs: {total_runs}")
    logger.info(f"  Confidence level: {plan.confidence_level:.0%}")
    logger.info(f"  Cooldown between runs: {plan.cooldown_seconds}s")
    if plan.use_adaptive:
        convergence = plan.multi_run.convergence
        assert convergence is not None  # use_adaptive guards this
        logger.info(f"  Convergence mode: {convergence.mode}")
        logger.info(f"  Convergence metric: {convergence.metric}")
        logger.info(f"  Convergence threshold: {convergence.threshold}")
        if convergence.mode == ConvergenceCriterionType.DISTRIBUTION:
            logger.info(
                "  Note: distribution mode converges when KS p-value > threshold "
                "(higher threshold = stricter, opposite of ci_width/cv)"
            )
    logger.info("=" * 80)


def _log_search_planner_active(
    plan: BenchmarkPlan,
    search_planner: SearchPlanner | None,
    logger: AIPerfLogger,
) -> None:
    """Log the adaptive-search banner when a planner was built."""
    if search_planner is None:
        return
    sweep = plan.sweep
    assert sweep is not None  # _build_search_planner returned non-None
    logger.info(
        f"Adaptive search active: planner={sweep.planner}, "
        f"max_iterations={sweep.max_iterations}, "
        f"search-space={[d.path for d in sweep.search_space]}, "
        f"objectives=[{','.join(f'{o.metric}:{o.stat}:{o.direction}' for o in sweep.objectives)}]"
    )
