# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-run orchestration for AIPerf benchmarks.

This module provides infrastructure for executing multiple benchmark runs
with different strategies (confidence reporting, parameter sweeps, etc.).
"""

from aiperf.orchestrator.aggregation import (
    AggregateResult,
    AggregationStrategy,
    ConfidenceAggregation,
    ConfidenceMetric,
)
from aiperf.orchestrator.models import (
    RunConfig,
    RunResult,
)
from aiperf.orchestrator.orchestrator import (
    MultiRunOrchestrator,
)
from aiperf.orchestrator.strategies import (
    ExecutionStrategy,
    FixedTrialsStrategy,
)

__all__ = [
    "AggregateResult",
    "AggregationStrategy",
    "ConfidenceAggregation",
    "ConfidenceMetric",
    "ExecutionStrategy",
    "FixedTrialsStrategy",
    "MultiRunOrchestrator",
    "RunConfig",
    "RunResult",
]
