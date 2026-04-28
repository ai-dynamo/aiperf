# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep aggregation for parameter sweeping."""

from enum import Enum
from typing import Any, NamedTuple


class OptimizationDirection(Enum):
    """Direction of optimization for a metric.

    Note: This is defined here for the parameter sweeping feature. Ideally, this
    would be a property of BaseMetric itself (e.g., BaseMetric.optimization_direction).
    See "Future Extensions" section for discussion of adding this to the metrics system.
    """

    MAXIMIZE = "maximize"  # Higher is better (e.g., throughput)
    MINIMIZE = "minimize"  # Lower is better (e.g., latency)


class Objective(NamedTuple):
    """Definition of an optimization objective.

    Args:
        metric_key: The metric tag (e.g., "request_throughput_avg", "time_to_first_token_p99")
        direction: Whether to maximize or minimize this metric

    Note: In the future, direction could be derived from BaseMetric.optimization_direction
    if that property is added to the metrics system.
    """

    metric_key: str
    direction: OptimizationDirection


class ParameterCombination(NamedTuple):
    """A specific combination of parameter values.

    Args:
        parameters: Dictionary mapping parameter names to their values
            Example: {"concurrency": 2, "request_rate": 10}
    """

    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return self.parameters.copy()

    def __str__(self) -> str:
        """String representation for logging/display."""
        parts = [f"{k}={v}" for k, v in sorted(self.parameters.items())]
        return ", ".join(parts)

    def __hash__(self) -> int:
        """Make hashable for use in sets/dicts."""
        return hash(tuple(sorted(self.parameters.items())))


# Default objectives for common use case.
# time_to_first_token_p99 is the canonical flattened key produced by both
# confidence aggregation and sweep-only aggregation (metric tag + stat key).
DEFAULT_PARETO_OBJECTIVES = [
    Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
    Objective("time_to_first_token_p99", OptimizationDirection.MINIMIZE),
]

# Non-streaming endpoints may not emit TTFT. request_latency_p99 is a
# separate metric (total round-trip latency), used as a fallback when
# TTFT is absent.
_LATENCY_CANDIDATES = ["time_to_first_token_p99", "request_latency_p99"]


def identify_pareto_optimal(
    per_combination_stats: dict[ParameterCombination, dict],
    objectives: list[Objective] | None = None,
) -> list[ParameterCombination]:
    """Identify Pareto optimal configurations across N objectives.

    A configuration is Pareto optimal if no other configuration is strictly better
    on ALL objectives simultaneously.

    Args:
        per_combination_stats: Statistics for each parameter combination
        objectives: List of objectives to optimize. If None, uses DEFAULT_PARETO_OBJECTIVES
            (throughput vs latency).

            Example: [
                Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
                Objective("time_to_first_token_p99", OptimizationDirection.MINIMIZE),
            ]

    Returns:
        List of parameter combinations that are Pareto optimal

    Example:
        # 2D Pareto frontier (throughput vs latency) - uses defaults
        pareto = identify_pareto_optimal(stats)

        # 3D Pareto frontier (throughput, latency, cost)
        objectives = [
            Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
            Objective("time_to_first_token_p99", OptimizationDirection.MINIMIZE),
            Objective("cost_per_request", OptimizationDirection.MINIMIZE),
        ]
        pareto = identify_pareto_optimal(stats, objectives)
    """
    if objectives is None:
        objectives = DEFAULT_PARETO_OBJECTIVES

    pareto_optimal = []

    for combo1, stats1 in per_combination_stats.items():
        # Extract objective values for this configuration
        values1 = [stats1[obj.metric_key]["mean"] for obj in objectives]

        is_dominated = False
        for combo2, stats2 in per_combination_stats.items():
            if combo1 == combo2:
                continue

            # Extract objective values for comparison configuration
            values2 = [stats2[obj.metric_key]["mean"] for obj in objectives]

            # Check if combo2 dominates combo1
            # Domination means: better or equal on all objectives, AND strictly better on at least one
            better_or_equal_count = 0
            strictly_better_count = 0

            for i, obj in enumerate(objectives):
                if obj.direction == OptimizationDirection.MAXIMIZE:
                    if values2[i] > values1[i]:
                        strictly_better_count += 1
                        better_or_equal_count += 1
                    elif values2[i] == values1[i]:
                        better_or_equal_count += 1
                    # else: values2[i] < values1[i], worse on this objective
                else:  # MINIMIZE
                    if values2[i] < values1[i]:
                        strictly_better_count += 1
                        better_or_equal_count += 1
                    elif values2[i] == values1[i]:
                        better_or_equal_count += 1
                    # else: values2[i] > values1[i], worse on this objective

            # combo2 dominates combo1 if it's better or equal on all AND strictly better on at least one
            if better_or_equal_count == len(objectives) and strictly_better_count > 0:
                is_dominated = True
                break

        if not is_dominated:
            pareto_optimal.append(combo1)

    return sorted(pareto_optimal, key=lambda c: tuple(sorted(c.parameters.items())))


class SweepAnalyzer:
    """Compute sweep-level statistics and analysis."""

    @staticmethod
    def _resolve_latency_key(
        per_combination_stats: dict["ParameterCombination", dict],
    ) -> str | None:
        """Find the best latency metric key present in all combinations.

        Prefers time_to_first_token_p99 (streaming). Falls back to
        request_latency_p99 (non-streaming).
        """
        for key in _LATENCY_CANDIDATES:
            if all(key in stats for stats in per_combination_stats.values()):
                return key
        return None

    @staticmethod
    def compute(
        per_combination_stats: dict[ParameterCombination, dict],
        sweep_parameters: list[dict[str, Any]],
    ) -> dict:
        """Compute sweep-level aggregate statistics.

        Args:
            per_combination_stats: Statistics for each parameter combination
            sweep_parameters: List of parameter definitions, each with:
                - name: Parameter name (e.g., "concurrency")
                - values: List of values for this parameter

        Returns:
            Dictionary with:
                - metadata: Sweep parameters and configuration
                - per_combination_metrics: Metrics for each parameter combination
                - best_configurations: Best combinations for key metrics
                - pareto_optimal: List of Pareto optimal combinations

        Example:
            >>> combo1 = ParameterCombination({"concurrency": 2, "request_rate": 10})
            >>> combo2 = ParameterCombination({"concurrency": 4, "request_rate": 10})
            >>> per_combination_stats = {
            ...     combo1: {"request_throughput_avg": {"mean": 100, "std": 5, ...}},
            ...     combo2: {"request_throughput_avg": {"mean": 180, "std": 8, ...}},
            ... }
            >>> sweep_params = [
            ...     {"name": "concurrency", "values": [2, 4]},
            ...     {"name": "request_rate", "values": [10]},
            ... ]
            >>> result = SweepAnalyzer.compute(per_combination_stats, sweep_params)
            >>> result["metadata"]["num_combinations"]
            2
        """
        # Calculate total number of combinations
        num_combinations = 1
        for param in sweep_parameters:
            num_combinations *= len(param["values"])

        # Build metadata section
        metadata = {
            "sweep_parameters": sweep_parameters,
            "num_combinations": num_combinations,
        }

        # Per-combination metrics section (convert to list format)
        per_combination_metrics = [
            {"parameters": combo.to_dict(), "metrics": stats}
            for combo, stats in per_combination_stats.items()
        ]

        # Identify best configurations
        best_configurations = {}
        if per_combination_stats:
            # Best throughput: highest request_throughput_avg
            if all(
                "request_throughput_avg" in stats
                for stats in per_combination_stats.values()
            ):
                best_throughput_combo, best_throughput_stats = max(
                    per_combination_stats.items(),
                    key=lambda item: item[1]["request_throughput_avg"]["mean"],
                )
                best_configurations["best_throughput"] = {
                    "parameters": best_throughput_combo.to_dict(),
                    "metric": best_throughput_stats["request_throughput_avg"]["mean"],
                    "unit": best_throughput_stats["request_throughput_avg"].get(
                        "unit", "requests/sec"
                    ),
                }

            # Best latency: check TTFT aliases then request_latency_p99
            latency_metric = SweepAnalyzer._resolve_latency_key(per_combination_stats)

            if latency_metric:
                best_latency_combo, best_latency_stats = min(
                    per_combination_stats.items(),
                    key=lambda item: item[1][latency_metric]["mean"],
                )
                best_configurations["best_latency_p99"] = {
                    "parameters": best_latency_combo.to_dict(),
                    "metric": best_latency_stats[latency_metric]["mean"],
                    "unit": best_latency_stats[latency_metric].get("unit", "ms"),
                }

        # Identify Pareto optimal configurations
        pareto_optimal = []
        if per_combination_stats:
            latency_key = SweepAnalyzer._resolve_latency_key(per_combination_stats)
            has_throughput = all(
                "request_throughput_avg" in stats
                for stats in per_combination_stats.values()
            )
            if has_throughput and latency_key:
                objectives = [
                    Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
                    Objective(latency_key, OptimizationDirection.MINIMIZE),
                ]
                pareto_combos = identify_pareto_optimal(
                    per_combination_stats, objectives
                )
                pareto_optimal = [combo.to_dict() for combo in pareto_combos]

        return {
            "metadata": metadata,
            "per_combination_metrics": per_combination_metrics,
            "best_configurations": best_configurations,
            "pareto_optimal": pareto_optimal,
        }
