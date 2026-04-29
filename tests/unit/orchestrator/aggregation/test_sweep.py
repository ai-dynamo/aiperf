# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for sweep aggregation helpers.

A small subset (5 cases) of main's PR #699 test suite — Pareto 2-D,
Pareto 3-D, ``ParameterCombination.to_dict`` round-trip, ``compute`` over
a 1-D sweep, and ``compute`` over empty stats. Phase 4.7 will land
broader integration coverage.
"""

from aiperf.orchestrator.aggregation import (
    DEFAULT_PARETO_OBJECTIVES,
    Objective,
    OptimizationDirection,
    ParameterCombination,
    SweepAnalyzer,
    identify_pareto_optimal,
)


def _stat(mean: float, p99: float | None = None) -> dict[str, float]:
    """Build a metric stats dict with realistic keys."""
    out: dict[str, float] = {"mean": mean}
    if p99 is not None:
        out["p99"] = p99
    return out


def test_parameter_combination_to_dict_round_trip_returns_independent_copy() -> None:
    combo = ParameterCombination({"concurrency": 10, "request_rate": 20})

    result = combo.to_dict()

    assert result == {"concurrency": 10, "request_rate": 20}
    result["concurrency"] = 999
    assert combo.parameters["concurrency"] == 10


def test_identify_pareto_optimal_2d_drops_dominated_combo() -> None:
    # Realistic throughput-vs-TTFT frontier across concurrency = 10/20/30.
    # c20 dominates c30: higher throughput AND lower latency.
    c10 = ParameterCombination({"concurrency": 10})
    c20 = ParameterCombination({"concurrency": 20})
    c30 = ParameterCombination({"concurrency": 30})
    stats = {
        c10: {
            "request_throughput_avg": _stat(100.0),
            "time_to_first_token_p99": _stat(50.0),
        },
        c20: {
            "request_throughput_avg": _stat(180.0),
            "time_to_first_token_p99": _stat(75.0),
        },
        c30: {
            "request_throughput_avg": _stat(170.0),
            "time_to_first_token_p99": _stat(90.0),
        },
    }

    pareto = identify_pareto_optimal(stats)

    assert {c.parameters["concurrency"] for c in pareto} == {10, 20}
    assert c30 not in pareto


def test_identify_pareto_optimal_3d_drops_dominated_combo() -> None:
    # 3-D objective space: throughput (max), TTFT (min), TTFO (min).
    # c_mid is dominated by c_best on all three axes.
    c_low = ParameterCombination({"concurrency": 5})
    c_mid = ParameterCombination({"concurrency": 10})
    c_best = ParameterCombination({"concurrency": 20})
    stats = {
        c_low: {
            "request_throughput_avg": _stat(80.0),
            "time_to_first_token_p99": _stat(40.0),
            "time_to_first_output_p99": _stat(45.0),
        },
        c_mid: {
            "request_throughput_avg": _stat(120.0),
            "time_to_first_token_p99": _stat(60.0),
            "time_to_first_output_p99": _stat(70.0),
        },
        c_best: {
            "request_throughput_avg": _stat(200.0),
            "time_to_first_token_p99": _stat(50.0),
            "time_to_first_output_p99": _stat(55.0),
        },
    }
    objectives = [
        Objective("request_throughput_avg", OptimizationDirection.MAXIMIZE),
        Objective("time_to_first_token_p99", OptimizationDirection.MINIMIZE),
        Objective("time_to_first_output_p99", OptimizationDirection.MINIMIZE),
    ]

    pareto = identify_pareto_optimal(stats, objectives)

    assert c_mid not in pareto
    assert c_low in pareto and c_best in pareto


def test_sweep_analyzer_compute_1d_concurrency_sweep_returns_full_schema() -> None:
    c10 = ParameterCombination({"concurrency": 10})
    c20 = ParameterCombination({"concurrency": 20})
    c30 = ParameterCombination({"concurrency": 30})
    stats = {
        c10: {
            "request_throughput_avg": _stat(100.0),
            "time_to_first_token_p99": _stat(50.0),
        },
        c20: {
            "request_throughput_avg": _stat(180.0),
            "time_to_first_token_p99": _stat(80.0),
        },
        c30: {
            "request_throughput_avg": _stat(170.0),
            "time_to_first_token_p99": _stat(95.0),
        },
    }
    sweep_parameters = [{"name": "concurrency", "values": [10, 20, 30]}]

    result = SweepAnalyzer.compute(stats, sweep_parameters)

    assert result["metadata"]["num_combinations"] == 3
    assert result["metadata"]["sweep_parameters"] == sweep_parameters

    per_combo = result["per_combination_metrics"]
    assert len(per_combo) == 3
    assert [entry["parameters"]["concurrency"] for entry in per_combo] == [10, 20, 30]

    best = result["best_configurations"]
    assert best["best_throughput"]["parameters"] == {"concurrency": 20}
    assert best["best_throughput"]["metric"] == 180.0
    assert best["best_latency_p99"]["parameters"] == {"concurrency": 10}

    pareto = result["pareto_optimal"]
    assert {tuple(p.items()) for p in pareto} == {
        (("concurrency", 10),),
        (("concurrency", 20),),
    }


def test_sweep_analyzer_compute_empty_stats_returns_empty_blocks() -> None:
    result = SweepAnalyzer.compute({}, [{"name": "concurrency", "values": [10]}])

    assert result["metadata"]["num_combinations"] == 1
    assert result["per_combination_metrics"] == []
    assert result["best_configurations"] == {}
    assert result["pareto_optimal"] == []


def test_default_pareto_objectives_are_throughput_max_then_ttft_min() -> None:
    assert len(DEFAULT_PARETO_OBJECTIVES) == 2
    assert DEFAULT_PARETO_OBJECTIVES[0] == Objective(
        "request_throughput_avg", OptimizationDirection.MAXIMIZE
    )
    assert DEFAULT_PARETO_OBJECTIVES[1] == Objective(
        "time_to_first_token_p99", OptimizationDirection.MINIMIZE
    )
