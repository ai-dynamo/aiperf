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
    """

    aggregation_type: str
    """Type of aggregation (e.g., "confidence", "sweep")."""

    num_runs: int
    """Total number of runs."""

    num_successful_runs: int
    """Number of runs that completed successfully."""

    failed_runs: list[dict[str, Any]] = field(default_factory=list)
    """Failed runs with error details."""

    metrics: dict[str, Any] = field(default_factory=dict)
    """Strategy-specific aggregated metrics."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Strategy-specific metadata."""


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
