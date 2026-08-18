# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared post-execution aggregation dispatch for benchmark plans."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.config import BenchmarkPlan
    from aiperf.orchestrator.models import RunResult
    from aiperf.orchestrator.strategies import ExecutionStrategy


async def aggregate_plan_results(
    results: list[RunResult],
    plan: BenchmarkPlan,
    *,
    strategy: ExecutionStrategy,
    base_dir: Path,
    logger: AIPerfLogger,
) -> Path:
    """Export aggregates using the plan's canonical single-point or sweep path.

    Grid sweeps and adaptive searches require both per-variation confidence
    aggregates and the cross-variation sweep aggregate. A single configuration
    instead uses the ordinary confidence aggregate across its repeated trials.

    Returns:
        Directory containing the aggregate artifacts a caller should publish.
        When a sweep has no successful cells, this falls back to the strategy's
        aggregate directory so callers can still write their own manifest.
    """
    if plan.is_sweep or plan.is_adaptive_search:
        from aiperf.cli_runner._sweep_aggregate import (
            aggregate_per_variation_and_export,
            aggregate_sweep_and_export,
        )

        _, sweep_dir = await asyncio.gather(
            aggregate_per_variation_and_export(results, plan, base_dir, logger),
            aggregate_sweep_and_export(results, plan, base_dir, logger),
        )
        if sweep_dir is not None:
            return sweep_dir
    else:
        from aiperf.cli_runner._aggregate import aggregate_and_export

        await aggregate_and_export(
            results,
            plan,
            strategy=strategy,
            base_dir=base_dir,
            logger=logger,
        )

    return strategy.get_aggregate_path(base_dir)
