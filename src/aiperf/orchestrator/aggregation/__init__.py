# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregation strategies for multi-run results."""

from aiperf.orchestrator.aggregation.base import (
    AggregateResult,
    AggregationStrategy,
)
from aiperf.orchestrator.aggregation.confidence import (
    ConfidenceAggregation,
    ConfidenceMetric,
)
from aiperf.orchestrator.aggregation.detailed import DetailedAggregation
from aiperf.orchestrator.aggregation.sweep import (
    DEFAULT_PARETO_OBJECTIVES,
    Objective,
    OptimizationDirection,
    ParameterCombination,
    SweepAnalyzer,
    identify_pareto_optimal,
)

__all__ = [
    "AggregateResult",
    "AggregationStrategy",
    "ConfidenceAggregation",
    "ConfidenceMetric",
    "DetailedAggregation",
    "DEFAULT_PARETO_OBJECTIVES",
    "Objective",
    "OptimizationDirection",
    "ParameterCombination",
    "SweepAnalyzer",
    "identify_pareto_optimal",
]
