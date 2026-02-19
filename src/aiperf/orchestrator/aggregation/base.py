# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base classes for aggregation strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from aiperf.orchestrator.models import RunResult


@dataclass
class AggregateResult:
    """Results from aggregating multiple runs.

    Extensible: Different strategies can add strategy-specific fields.

    Attributes:
        aggregation_type: Type of aggregation (e.g., "confidence", "sweep")
        num_runs: Total number of runs
        num_successful_runs: Number of successful runs
        failed_runs: List of failed runs with error details
        metrics: Strategy-specific aggregated metrics
        metadata: Strategy-specific metadata
    """

    aggregation_type: str
    num_runs: int
    num_successful_runs: int
    failed_runs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class AggregationStrategy(ABC):
    """Base class for multi-run aggregation strategies.

    Design: Strategy pattern allows different aggregation logic
    without modifying orchestration.
    """

    @abstractmethod
    def aggregate(self, results: list[RunResult]) -> AggregateResult:
        """Aggregate results from multiple runs.

        Args:
            results: List of RunResult from orchestrator

        Returns:
            AggregateResult with strategy-specific statistics
        """
        pass

    @abstractmethod
    def get_aggregation_type(self) -> str:
        """Return type identifier for this strategy."""
        pass
