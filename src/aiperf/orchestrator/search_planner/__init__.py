# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive outer-loop planners (e.g. Bayesian Optimization) for AIPerf.

A BenchmarkPlan can carry an optional AdaptiveSearchConfig (defined in
aiperf.config.adaptive_search). When present, the orchestrator iterates by
asking a planner for the next BenchmarkConfig to evaluate rather than
walking a pre-enumerated variation list.
"""

from aiperf.orchestrator.search_planner.base import SearchIteration, SearchPlanner

__all__ = ["SearchIteration", "SearchPlanner"]
