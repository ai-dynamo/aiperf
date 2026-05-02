# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SearchPlanner ABC and SearchIteration dataclass.

Schema for the BO config itself lives in aiperf.config.adaptive_search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.sweep import SweepVariation
    from aiperf.orchestrator.models import RunResult


__all__ = ["SearchIteration", "SearchPlanner"]


@dataclass
class SearchIteration:
    """One entry in the BO trajectory log.

    Written to search_history.json incrementally after each iteration. `results`
    is the per-trial RunResult list at this BO point (length == plan.trials
    for FixedTrialsStrategy).
    """

    iteration_idx: int
    variation_values: dict[str, Any]
    objective_value: float | None = None
    results: list[Any] = field(default_factory=list)


class SearchPlanner(ABC):
    """Abstract base for adaptive outer-loop planners.

    Implementations: BayesianSearchPlanner (skopt-backed). Future: GridPlanner
    (for testing), OptunaSearchPlanner (TPE), RandomSearchPlanner (baseline).
    """

    @abstractmethod
    def ask(self) -> tuple[BenchmarkConfig, SweepVariation] | None:
        """Return (cfg, variation) for the next iteration, or None when done.

        The cfg is a deep-copied BenchmarkConfig with the proposed values
        substituted at their dotted paths. The SweepVariation has
        `index = iteration_idx`, `label = "search_iter_NNNN"`, and
        `values = {path: proposed_value, ...}` so downstream
        `aggregate_sweep_and_export` groups results naturally.
        """

    @abstractmethod
    def tell(self, variation: SweepVariation, results: list[RunResult]) -> None:
        """Tell the planner what happened at the most recent point."""

    @abstractmethod
    def is_converged(self) -> bool:
        """True when max_iterations exhausted or plateau detected."""

    @abstractmethod
    def history(self) -> list[SearchIteration]:
        """All iterations recorded so far, in submission order."""
