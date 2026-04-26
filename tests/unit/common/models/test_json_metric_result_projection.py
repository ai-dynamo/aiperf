# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for JsonMetricResult.project_summary_dict.

The orchestrator's RunResult.summary_metrics is dict[str, JsonMetricResult].
JsonMetricResult uses extra="forbid" and only knows percentile fields, so
naive dict[str, JsonMetricResult] coercion fails on:

- ``status.summary`` (operator path) — mixes per-tag stat dicts with
  bolted-on top-level scalars (``total_requests``, ``error_rate``).
- ``profile_export_aiperf.json`` — per-tag dicts include ``count``,
  ``header``, ``sum`` extras that JsonMetricResult rejects.

project_summary_dict is the projection helper that drops scalars, drops
extras, and returns a clean dict[str, JsonMetricResult].
"""

from __future__ import annotations

import pytest

from aiperf.common.models.export_models import JsonMetricResult


def test_project_handles_top_level_scalars():
    """Bolted-on scalar entries (total_requests, error_rate) drop out."""
    summary = {
        "total_requests": 200,
        "error_rate": 0.0,
        "request_latency": {"unit": "ms", "avg": 12.3, "p99": 45.6},
    }
    out = JsonMetricResult.project_summary_dict(summary)
    assert set(out.keys()) == {"request_latency"}
    assert isinstance(out["request_latency"], JsonMetricResult)
    assert out["request_latency"].avg == 12.3
    assert out["request_latency"].p99 == 45.6


def test_project_drops_per_tag_extras():
    """count/header/sum are MetricResult-only fields and get projected away."""
    summary = {
        "output_token_throughput": {
            "unit": "tokens/sec",
            "avg": 1438.88,
            "p50": 1400.0,
            "p99": 1600.0,
            "count": 1,
            "header": "Total Token Throughput",
            "sum": 1438.88,
        },
    }
    out = JsonMetricResult.project_summary_dict(summary)
    assert isinstance(out["output_token_throughput"], JsonMetricResult)
    assert out["output_token_throughput"].avg == 1438.88
    assert out["output_token_throughput"].p99 == 1600.0


def test_project_drops_dicts_without_unit():
    """A dict without `unit` is not a metric — drop it."""
    summary = {
        "weird_meta": {"foo": "bar", "baz": 1},
        "request_count": {"unit": "count", "avg": 200},
    }
    out = JsonMetricResult.project_summary_dict(summary)
    assert set(out.keys()) == {"request_count"}


def test_project_returns_empty_for_none_or_empty():
    assert JsonMetricResult.project_summary_dict(None) == {}
    assert JsonMetricResult.project_summary_dict({}) == {}


def test_project_realistic_status_summary():
    """End-to-end shape that mirrors what MetricsSummary.to_status_dict writes."""
    # Mirrors the dict that broke smoke-sweep on DGX 2026-04-26.
    summary = {
        "request_throughput": {
            "unit": "req/sec",
            "avg": 50.0,
            "p50": 49.5,
            "p99": 51.2,
            "count": 200,
            "header": "Request Throughput",
            "sum": 10000,
        },
        "time_to_first_token": {
            "unit": "ms",
            "avg": 100.0,
            "p50": 90.0,
            "p99": 200.0,
            "count": 200,
            "header": "Time to First Token",
            "sum": 20000,
        },
        "total_requests": 200,
        "error_rate": 0.0,
    }
    out = JsonMetricResult.project_summary_dict(summary)
    assert set(out.keys()) == {"request_throughput", "time_to_first_token"}
    assert all(isinstance(v, JsonMetricResult) for v in out.values())
    assert out["time_to_first_token"].p99 == 200.0


@pytest.mark.parametrize(
    "non_dict_value",
    [200, 0.5, "string", [1, 2, 3], None, True],
)
def test_project_drops_non_dict_values(non_dict_value):
    """Anything that's not a dict-with-unit gets filtered."""
    summary = {
        "scalar_field": non_dict_value,
        "good_metric": {"unit": "ms", "avg": 1.0},
    }
    out = JsonMetricResult.project_summary_dict(summary)
    assert set(out.keys()) == {"good_metric"}


def test_project_used_by_runresult_validation():
    """End-to-end: the projection output validates as RunResult.summary_metrics."""
    from aiperf.orchestrator.models import RunResult

    summary = {
        "request_latency": {"unit": "ms", "avg": 1.0, "p99": 2.0, "count": 100},
        "total_requests": 200,
    }
    projected = JsonMetricResult.project_summary_dict(summary)
    # Construct the RunResult — Pydantic validation must accept the projection.
    result = RunResult(
        label="test/run_0001",
        success=True,
        summary_metrics=projected,
        artifacts_path=None,
    )
    assert "request_latency" in result.summary_metrics
    assert "total_requests" not in result.summary_metrics
