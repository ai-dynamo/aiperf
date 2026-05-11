# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SLA-filter integration in SweepAnalyzer.compute()."""

from __future__ import annotations

from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.orchestrator.aggregation.sweep import (
    ParameterCombination,
    SweepAnalyzer,
)


def _stats_for(throughput: float, ttft_p99: float) -> dict:
    return {
        "request_throughput_avg": {"mean": throughput, "unit": "requests/sec"},
        "time_to_first_token_p99": {"mean": ttft_p99, "unit": "ms"},
    }


def _build_stats(rows: list[tuple[int, float, float]]) -> dict:
    return {
        ParameterCombination({"concurrency": c}): _stats_for(t, ttft)
        for c, t, ttft in rows
    }


def test_compute_without_sla_filters_unchanged():
    stats = _build_stats([(10, 100.0, 50.0), (20, 180.0, 80.0)])
    out = SweepAnalyzer.compute(stats, [{"name": "concurrency", "values": [10, 20]}])
    assert "sla_constraints" not in out["metadata"]
    assert out["best_configurations"]["best_throughput"]["parameters"] == {
        "concurrency": 20
    }


def test_compute_filters_best_configurations_to_feasible_only():
    stats = _build_stats([(10, 100.0, 40.0), (20, 180.0, 250.0)])
    sla_filter = SLAFilter(
        metric_tag="time_to_first_token", stat="p99", op="lt", threshold=200.0
    )
    out = SweepAnalyzer.compute(
        stats,
        [{"name": "concurrency", "values": [10, 20]}],
        sla_filters=[sla_filter],
    )
    # 20 is infeasible (250 ms > 200 ms); best feasible throughput is concurrency=10.
    assert out["best_configurations"]["best_throughput"]["parameters"] == {
        "concurrency": 10
    }
    assert out["metadata"]["sla_constraints"]["feasible_count"] == 1
    assert out["metadata"]["sla_constraints"]["infeasible_count"] == 1


def test_compute_falls_back_to_full_set_when_zero_feasible():
    stats = _build_stats([(10, 100.0, 250.0), (20, 180.0, 300.0)])
    sla_filter = SLAFilter(
        metric_tag="time_to_first_token", stat="p99", op="lt", threshold=200.0
    )
    out = SweepAnalyzer.compute(
        stats,
        [{"name": "concurrency", "values": [10, 20]}],
        sla_filters=[sla_filter],
    )
    # All infeasible -> best_configurations falls back to global best.
    assert out["best_configurations"]["best_throughput"]["parameters"] == {
        "concurrency": 20
    }
    assert out["metadata"]["sla_constraints"]["feasible_count"] == 0


def test_compute_pareto_restricted_to_feasible_with_no_fallback():
    stats = _build_stats(
        [
            (10, 100.0, 50.0),
            (20, 180.0, 250.0),
            (30, 120.0, 60.0),
        ]
    )
    sla_filter = SLAFilter(
        metric_tag="time_to_first_token", stat="p99", op="lt", threshold=200.0
    )
    out = SweepAnalyzer.compute(
        stats,
        [{"name": "concurrency", "values": [10, 20, 30]}],
        sla_filters=[sla_filter],
    )
    # Concurrency=20 is infeasible -> dropped from pareto. The remaining
    # feasible set is {(10, 100, 50), (30, 120, 60)}; (30, 120, 60) dominates
    # nothing since (10, 100, 50) has lower latency. Both are pareto-optimal.
    pareto_concurrencies = sorted(p["concurrency"] for p in out["pareto_optimal"])
    assert 20 not in pareto_concurrencies
    assert pareto_concurrencies == [10, 30]


def test_compute_metadata_includes_filter_dump():
    stats = _build_stats([(10, 100.0, 50.0)])
    sla_filter = SLAFilter(
        metric_tag="time_to_first_token", stat="p99", op="lt", threshold=200.0
    )
    out = SweepAnalyzer.compute(
        stats,
        [{"name": "concurrency", "values": [10]}],
        sla_filters=[sla_filter],
    )
    constraints = out["metadata"]["sla_constraints"]
    assert constraints["active_filters"][0]["metric_tag"] == "time_to_first_token"
    assert constraints["active_filters"][0]["op"] == "lt"
    assert constraints["active_filters"][0]["threshold"] == 200.0


def test_compute_accepts_filter_as_dict():
    """Grid path round-trips SLAFilter through model_dump; dicts must work too."""
    stats = _build_stats([(10, 100.0, 50.0), (20, 180.0, 250.0)])
    out = SweepAnalyzer.compute(
        stats,
        [{"name": "concurrency", "values": [10, 20]}],
        sla_filters=[
            {
                "metric_tag": "time_to_first_token",
                "stat": "p99",
                "op": "lt",
                "threshold": 200.0,
            }
        ],
    )
    assert out["metadata"]["sla_constraints"]["feasible_count"] == 1


def test_compute_reads_single_trial_tag_only_key_layout():
    """Single-trial sweeps store ``"<metric_tag>": {"avg": v, "p99": v, ...}``
    instead of the multi-trial flat ``"<metric_tag>_<stat>": {"mean": v}`` shape.

    Locks in dual-key support in `passes_filter` so single-trial sweeps don't
    silently mark every combination infeasible due to key-shape mismatch.
    """
    # Single-trial layout: stat lives as a direct attribute on the metric dict.
    single_trial_stats = {
        ParameterCombination({"concurrency": 10}): {
            "request_throughput": {"avg": 100.0, "p50": 99.0, "p99": 110.0},
            "time_to_first_token": {"avg": 35.0, "p50": 34.0, "p99": 50.0},
        },
        ParameterCombination({"concurrency": 20}): {
            "request_throughput": {"avg": 180.0, "p50": 175.0, "p99": 200.0},
            "time_to_first_token": {"avg": 220.0, "p50": 210.0, "p99": 250.0},
        },
    }
    out = SweepAnalyzer.compute(
        single_trial_stats,
        [{"name": "concurrency", "values": [10, 20]}],
        sla_filters=[
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p99",
                op="lt",
                threshold=200.0,
            )
        ],
    )
    constraints = out["metadata"]["sla_constraints"]
    assert constraints["feasible_count"] == 1, (
        "single-trial layout should produce one feasible combo (concurrency=10 has p99=50<200)"
    )
    assert constraints["infeasible_count"] == 1
