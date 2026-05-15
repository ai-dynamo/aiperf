# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep aggregation: pure-computation Pareto / sweep-analysis helpers.

Operates on per-combination metric dicts of shape::

    {ParameterCombination: {metric_tag: {"mean": float, "p99": float, ...}}}

No I/O, no Pydantic, no BenchmarkConfig dependencies. The CSV/JSON sweep
exporters consume the dict produced by :meth:`SweepAnalyzer.compute`.

In-process vs cluster sweeps
----------------------------

AIPerf has TWO sweep paths and they target different scales:

1. **In-process sweep** (this module + ``MultiRunOrchestrator`` +
   ``aggregate_sweep_and_export``). Triggered by ``--concurrency 10,20,30``
   on a single ``aiperf profile`` invocation. ``build_benchmark_plan``
   materializes one ``BenchmarkConfig`` per variation from the config-v2
   ``BenchmarkPlan`` envelope.
   The orchestrator then executes variations x trials sequentially in the
   same process. Best for: single-machine sweeps, quick iteration, CI/dev.

2. **Cluster sweep** (``AIPerfSweep`` CRD + ``operator/handlers/sweep/``).
   The k8s operator owns the cluster-wide cardinality contract: one
   ``AIPerfJob`` (and one controller pod) per variation. Each child pod
   sees a single-config plan. Best for: large sweeps, parallelism across
   nodes, durability through controller restarts.

The two paths are mutually exclusive at runtime: when
``AIPERF_OPERATOR_MANAGED=1`` is set in the controller pod, the
in-process gate (``cli_runner._reject_in_process_sweep_under_operator``)
hard-fails on ``plan.is_sweep`` to keep both layers from sweeping at once.
Use the CRD path whenever you need horizontal scale or restart durability;
otherwise the in-process path is simpler.
"""

from typing import Any, NamedTuple

from aiperf.common.enums import OptimizationDirection


class ParetoObjective(NamedTuple):
    """Definition of a single optimization objective for Pareto analysis.

    Args:
        metric_key: Flattened metric tag, e.g. ``"request_throughput_avg"``
            or ``"time_to_first_token_p99"`` (metric tag + stat key, the same
            shape produced by both confidence and sweep-only aggregation).
        direction: Whether higher or lower values are preferred.
    """

    metric_key: str
    direction: OptimizationDirection


class ParameterCombination(NamedTuple):
    """A specific combination of swept parameter values.

    Args:
        parameters: Mapping of parameter name -> concrete value, e.g.
            ``{"concurrency": 10, "request_rate": 20}``.

    Example:
        >>> combo = ParameterCombination({"concurrency": 10, "request_rate": 20})
        >>> combo.to_dict()
        {'concurrency': 10, 'request_rate': 20}
        >>> str(combo)
        'concurrency=10, request_rate=20'
    """

    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the parameters dict.

        Example:
            >>> combo = ParameterCombination({"concurrency": 10})
            >>> d = combo.to_dict()
            >>> d["concurrency"] = 999
            >>> combo.parameters["concurrency"]
            10
        """
        return self.parameters.copy()

    def __str__(self) -> str:
        """Render as ``"k1=v1, k2=v2"`` with keys sorted alphabetically."""
        parts = [f"{k}={v}" for k, v in sorted(self.parameters.items())]
        return ", ".join(parts)

    def __hash__(self) -> int:
        """Hash by sorted (key, value) pairs so dict ordering doesn't matter."""
        return hash(tuple(sorted(self.parameters.items())))


# Default objectives for the common throughput-vs-latency Pareto frontier.
# `time_to_first_token_p99` is the canonical flattened key produced by both
# confidence aggregation and sweep-only aggregation (metric tag + stat key).
DEFAULT_PARETO_OBJECTIVES: list[ParetoObjective] = [
    ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
    ParetoObjective("time_to_first_token_p99", OptimizationDirection.MINIMIZE),
]

# Non-streaming endpoints may not emit TTFT. `request_latency_p99` is the
# fallback (total round-trip latency) when TTFT is absent across the sweep.
_LATENCY_CANDIDATES: list[str] = [
    "time_to_first_token_p99",
    "request_latency_p99",
]


def _dominates(
    values_a: list[float],
    values_b: list[float],
    objectives: list[ParetoObjective],
) -> bool:
    """Return True iff configuration A strictly Pareto-dominates B.

    Domination means A is better-or-equal on every objective AND strictly
    better on at least one. Direction (maximize/minimize) is taken from the
    matching :class:`ParetoObjective`.

    Args:
        values_a: Per-objective metric means for candidate dominator.
        values_b: Per-objective metric means for candidate dominated.
        objectives: Objectives in the same order as ``values_a`` / ``values_b``.

    Example:
        >>> objs = [
        ...     ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
        ...     ParetoObjective("time_to_first_token_p99", OptimizationDirection.MINIMIZE),
        ... ]
        >>> _dominates([200.0, 50.0], [100.0, 80.0], objs)
        True
        >>> _dominates([200.0, 80.0], [100.0, 50.0], objs)
        False
    """
    better_or_equal = 0
    strictly_better = 0
    for i, obj in enumerate(objectives):
        a, b = values_a[i], values_b[i]
        if obj.direction == OptimizationDirection.MAXIMIZE:
            if a > b:
                strictly_better += 1
                better_or_equal += 1
            elif a == b:
                better_or_equal += 1
        else:  # MINIMIZE
            if a < b:
                strictly_better += 1
                better_or_equal += 1
            elif a == b:
                better_or_equal += 1
    return better_or_equal == len(objectives) and strictly_better > 0


def identify_pareto_optimal(
    per_combination_stats: dict[ParameterCombination, dict],
    objectives: list[ParetoObjective] | None = None,
) -> list[ParameterCombination]:
    """Identify Pareto-optimal combinations across N objectives.

    A combination is Pareto-optimal if no other combination strictly
    dominates it (better-or-equal on all objectives, strictly better on at
    least one). Outer loop here; pairwise check is in :func:`_dominates`.

    Args:
        per_combination_stats: Per-combination metric dict.
        objectives: Objectives to optimize; defaults to
            :data:`DEFAULT_PARETO_OBJECTIVES` (throughput vs TTFT-p99).

    Returns:
        Pareto-optimal combinations, sorted by ``(key, value)`` pairs for
        deterministic output.

    Example:
        >>> c_low = ParameterCombination({"concurrency": 10})
        >>> c_mid = ParameterCombination({"concurrency": 20})
        >>> c_high = ParameterCombination({"concurrency": 30})
        >>> stats = {
        ...     c_low:  {"request_throughput_avg": {"mean": 100.0},
        ...              "time_to_first_token_p99": {"mean": 50.0}},
        ...     c_mid:  {"request_throughput_avg": {"mean": 150.0},
        ...              "time_to_first_token_p99": {"mean": 80.0}},
        ...     c_high: {"request_throughput_avg": {"mean": 140.0},
        ...              "time_to_first_token_p99": {"mean": 90.0}},
        ... }
        >>> pareto = identify_pareto_optimal(stats)
        >>> sorted(c.parameters["concurrency"] for c in pareto)
        [10, 20]
    """
    if objectives is None:
        objectives = DEFAULT_PARETO_OBJECTIVES

    pareto_optimal: list[ParameterCombination] = []
    for combo1, stats1 in per_combination_stats.items():
        values1 = [stats1[obj.metric_key]["mean"] for obj in objectives]
        is_dominated = False
        for combo2, stats2 in per_combination_stats.items():
            if combo1 == combo2:
                continue
            values2 = [stats2[obj.metric_key]["mean"] for obj in objectives]
            if _dominates(values2, values1, objectives):
                is_dominated = True
                break
        if not is_dominated:
            pareto_optimal.append(combo1)

    return sorted(pareto_optimal, key=lambda c: tuple(sorted(c.parameters.items())))


class SweepAnalyzer:
    """Compute sweep-level statistics and analysis.

    Public surface is :meth:`compute`; the underscore-prefixed helpers
    factor the orchestration.
    """

    @staticmethod
    def _resolve_latency_key(
        per_combination_stats: dict[ParameterCombination, dict],
    ) -> str | None:
        """Pick the best latency metric key present in every combination.

        Prefers ``time_to_first_token_p99`` (streaming); falls back to
        ``request_latency_p99`` (non-streaming).

        Example:
            >>> combo = ParameterCombination({"concurrency": 10})
            >>> SweepAnalyzer._resolve_latency_key({
            ...     combo: {"request_latency_p99": {"mean": 75.0}}
            ... })
            'request_latency_p99'
        """
        for key in _LATENCY_CANDIDATES:
            if all(key in stats for stats in per_combination_stats.values()):
                return key
        return None

    @staticmethod
    def _build_metadata(
        sweep_parameters: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the ``metadata`` block: parameters and combination count.

        Example:
            >>> SweepAnalyzer._build_metadata([{"name": "concurrency", "values": [10, 20, 30]}])
            {'sweep_parameters': [{'name': 'concurrency', 'values': [10, 20, 30]}], 'num_combinations': 3}
        """
        num_combinations = 1
        for param in sweep_parameters:
            num_combinations *= len(param["values"])
        return {
            "sweep_parameters": sweep_parameters,
            "num_combinations": num_combinations,
        }

    @staticmethod
    def _build_per_combination_block(
        per_combination_stats: dict[ParameterCombination, dict],
    ) -> list[dict[str, Any]]:
        """Structure the ``per_combination_metrics`` list (input order).

        Example:
            >>> combo = ParameterCombination({"concurrency": 10})
            >>> SweepAnalyzer._build_per_combination_block({
            ...     combo: {"request_throughput_avg": {"mean": 100.0}}
            ... })
            [{'parameters': {'concurrency': 10}, 'metrics': {'request_throughput_avg': {'mean': 100.0}}}]
        """
        return [
            {"parameters": combo.to_dict(), "metrics": stats}
            for combo, stats in per_combination_stats.items()
        ]

    @staticmethod
    def _compute_best_configurations(
        per_combination_stats: dict[ParameterCombination, dict],
    ) -> dict[str, Any]:
        """Pick per-objective single best combination (throughput, latency).

        Returns the ``best_configurations`` dict subset of the schema.
        Empty stats produce ``{}``. A missing throughput / latency metric
        in any combination drops only that key; the other still emits.

        Example:
            >>> c1 = ParameterCombination({"concurrency": 10})
            >>> c2 = ParameterCombination({"concurrency": 20})
            >>> stats = {
            ...     c1: {"request_throughput_avg": {"mean": 100.0, "unit": "requests/sec"}},
            ...     c2: {"request_throughput_avg": {"mean": 180.0, "unit": "requests/sec"}},
            ... }
            >>> best = SweepAnalyzer._compute_best_configurations(stats)
            >>> best["best_throughput"]["parameters"]
            {'concurrency': 20}
        """
        best: dict[str, Any] = {}
        if not per_combination_stats:
            return best

        if all(
            "request_throughput_avg" in stats
            for stats in per_combination_stats.values()
        ):
            combo, stats = max(
                per_combination_stats.items(),
                key=lambda item: item[1]["request_throughput_avg"]["mean"],
            )
            best["best_throughput"] = {
                "parameters": combo.to_dict(),
                "metric": stats["request_throughput_avg"]["mean"],
                "unit": stats["request_throughput_avg"].get("unit", "requests/sec"),
            }

        latency_metric = SweepAnalyzer._resolve_latency_key(per_combination_stats)
        if latency_metric:
            combo, stats = min(
                per_combination_stats.items(),
                key=lambda item: item[1][latency_metric]["mean"],
            )
            best["best_latency_p99"] = {
                "parameters": combo.to_dict(),
                "metric": stats[latency_metric]["mean"],
                "unit": stats[latency_metric].get("unit", "ms"),
            }
        return best

    @staticmethod
    def _compute_pareto(
        per_combination_stats: dict[ParameterCombination, dict],
    ) -> list[dict[str, Any]]:
        """Compute the throughput-vs-best-latency Pareto frontier.

        Uses ``request_throughput_avg`` (maximize) and the resolved latency
        key (minimize). Returns a list of ``parameters`` dicts; empty if
        either metric is missing in any combination.

        Example:
            >>> c1 = ParameterCombination({"concurrency": 10})
            >>> stats = {
            ...     c1: {"request_throughput_avg": {"mean": 100.0},
            ...          "time_to_first_token_p99": {"mean": 50.0}},
            ... }
            >>> SweepAnalyzer._compute_pareto(stats)
            [{'concurrency': 10}]
        """
        if not per_combination_stats:
            return []

        latency_key = SweepAnalyzer._resolve_latency_key(per_combination_stats)
        has_throughput = all(
            "request_throughput_avg" in stats
            for stats in per_combination_stats.values()
        )
        if not (has_throughput and latency_key):
            return []

        objectives = [
            ParetoObjective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
            ParetoObjective(latency_key, OptimizationDirection.MINIMIZE),
        ]
        pareto_combos = identify_pareto_optimal(per_combination_stats, objectives)
        return [combo.to_dict() for combo in pareto_combos]

    @staticmethod
    def compute(
        per_combination_stats: dict[ParameterCombination, dict],
        sweep_parameters: list[dict[str, Any]],
        sla_filters: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Compute sweep-level aggregate statistics.

        Thin orchestrator over :meth:`_build_metadata`,
        :meth:`_build_per_combination_block`,
        :meth:`_compute_best_configurations`, and :meth:`_compute_pareto`.

        When ``sla_filters`` is non-empty, ``best_configurations`` is filtered
        to feasible configurations first (falling back to the full set when
        zero are feasible — the user still sees something), and
        ``pareto_optimal`` is filtered to feasible-only with NO fallback (a
        Pareto frontier of infeasible points isn't meaningful). The applied
        constraints land on ``metadata.sla_constraints`` for downstream
        renderers.

        Args:
            per_combination_stats: Statistics for each parameter combination.
            sweep_parameters: List of parameter definitions, each with
                ``name`` (str) and ``values`` (list of allowed values).
            sla_filters: Optional list of ``SLAFilter`` objects (or dicts with
                ``metric_tag`` / ``stat`` / ``op`` / ``threshold``) to filter
                ``best_configurations`` and ``pareto_optimal`` against. Empty
                list / ``None`` disables filtering.

        Returns:
            Dict with keys ``metadata``, ``per_combination_metrics``,
            ``best_configurations``, ``pareto_optimal``.

        Example:
            >>> c1 = ParameterCombination({"concurrency": 10})
            >>> c2 = ParameterCombination({"concurrency": 20})
            >>> stats = {
            ...     c1: {"request_throughput_avg": {"mean": 100.0},
            ...          "time_to_first_token_p99": {"mean": 50.0}},
            ...     c2: {"request_throughput_avg": {"mean": 180.0},
            ...          "time_to_first_token_p99": {"mean": 80.0}},
            ... }
            >>> result = SweepAnalyzer.compute(
            ...     stats,
            ...     [{"name": "concurrency", "values": [10, 20]}],
            ... )
            >>> result["metadata"]["num_combinations"]
            2
        """
        metadata = SweepAnalyzer._build_metadata(sweep_parameters)
        per_combination = SweepAnalyzer._build_per_combination_block(
            per_combination_stats
        )

        feasible_stats: dict[ParameterCombination, dict] | None = None
        if sla_filters:
            from aiperf.orchestrator.aggregation.sweep_sla_filter import (
                filter_feasible,
                sla_filter_to_dict,
            )

            feasible_stats = filter_feasible(per_combination_stats, sla_filters)
            metadata["sla_constraints"] = {
                "active_filters": [sla_filter_to_dict(f) for f in sla_filters],
                "feasible_count": len(feasible_stats),
                "infeasible_count": len(per_combination_stats) - len(feasible_stats),
            }

        best_source = feasible_stats if feasible_stats else per_combination_stats
        best = SweepAnalyzer._compute_best_configurations(best_source)
        pareto = SweepAnalyzer._compute_pareto(
            feasible_stats if sla_filters else per_combination_stats
        )

        return {
            "metadata": metadata,
            "per_combination_metrics": per_combination,
            "best_configurations": best,
            "pareto_optimal": pareto,
        }
