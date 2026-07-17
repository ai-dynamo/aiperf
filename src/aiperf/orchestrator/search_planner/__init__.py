# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive outer-loop planners (e.g. Bayesian Optimization) for AIPerf.

A BenchmarkPlan can carry an optional AdaptiveSearchSweep (defined in
aiperf.config.sweep). When ``plan.sweep`` is an AdaptiveSearchSweep, the
orchestrator iterates by asking a planner for the next BenchmarkConfig to
evaluate rather than walking a pre-enumerated variation list.
"""

from aiperf.orchestrator.search_planner.base import SearchIteration, SearchPlanner
from aiperf.orchestrator.search_planner.monotonic import MonotonicSLASearchPlanner

__all__ = [
    "BayesianSearchPlanner",
    "MonotonicSLASearchPlanner",
    "MultiTierPlanner",
    "OptunaSearchPlanner",
    "SearchIteration",
    "SearchPlanner",
    "SmoothIsotonicSLAPlanner",
    "evaluate_tiers_on_grid",
]


def __getattr__(name: str) -> object:
    """Lazy import so optional extras (optuna / scipy) only load on use."""
    if name == "BayesianSearchPlanner":
        from aiperf.orchestrator.search_planner.bayesian import (
            BayesianSearchPlanner,
        )

        return BayesianSearchPlanner
    if name == "OptunaSearchPlanner":
        from aiperf.orchestrator.search_planner.optuna_planner import (
            OptunaSearchPlanner,
        )

        return OptunaSearchPlanner
    if name == "SmoothIsotonicSLAPlanner":
        from aiperf.orchestrator.search_planner.smooth_isotonic import (
            SmoothIsotonicSLAPlanner,
        )

        return SmoothIsotonicSLAPlanner
    if name == "MultiTierPlanner":
        from aiperf.orchestrator.search_planner.multi_tier_planner import (
            MultiTierPlanner,
        )

        return MultiTierPlanner
    if name == "evaluate_tiers_on_grid":
        from aiperf.orchestrator.search_planner.multi_tier_grid import (
            evaluate_tiers_on_grid,
        )

        return evaluate_tiers_on_grid
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
