# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf._cli_runner_sweep_helpers.aggregate_sweep_and_export."""

from __future__ import annotations

import json
import logging

import pytest

from aiperf._cli_runner_sweep_helpers import aggregate_sweep_and_export
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.benchmark import BenchmarkConfig, BenchmarkPlan
from aiperf.orchestrator.models import RunResult

_MINIMAL_CONFIG_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {
            "name": "default",
            "type": "concurrency",
            "requests": 100,
            "concurrency": 1,
        }
    ],
    "random_seed": 42,
}


def _make_plan(confidence_level: float = 0.95) -> BenchmarkPlan:
    cfg = BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS)
    return BenchmarkPlan(configs=[cfg], confidence_level=confidence_level)


def _result(
    label: str, concurrency: int, throughput: float, ttft_p99: float, *, success=True
) -> RunResult:
    """Build a RunResult tagged with a single-param sweep variation."""
    return RunResult(
        label=label,
        success=success,
        summary_metrics={
            "request_throughput": JsonMetricResult(
                unit="requests/sec", avg=throughput, min=throughput, max=throughput
            ),
            "time_to_first_token": JsonMetricResult(
                unit="ms", avg=ttft_p99 - 5, p99=ttft_p99, min=ttft_p99, max=ttft_p99
            ),
        },
        variation_label=f"concurrency={concurrency}",
        variation_values={"concurrency": concurrency},
        trial_index=0,
    )


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test.sweep_helpers")


@pytest.mark.asyncio
async def test_aggregate_sweep_and_export_two_variations_one_trial(tmp_path, logger):
    """Two variations × 1 trial: writes JSON+CSV with 2 per-combination rows."""
    plan = _make_plan()
    results = [
        _result("c10", concurrency=10, throughput=100.0, ttft_p99=50.0),
        _result("c20", concurrency=20, throughput=180.0, ttft_p99=80.0),
    ]

    out_dir = await aggregate_sweep_and_export(results, plan, tmp_path, logger)

    assert out_dir == tmp_path / "sweep_aggregate"
    json_path = out_dir / "profile_export_aiperf_sweep.json"
    csv_path = out_dir / "profile_export_aiperf_sweep.csv"
    assert json_path.exists()
    assert csv_path.exists()

    data = json.loads(json_path.read_text())
    assert len(data["per_combination_metrics"]) == 2
    # Single-trial collapse: std == 0 across cells
    for entry in data["per_combination_metrics"]:
        for metric in entry["metrics"].values():
            assert metric["std"] == 0.0


@pytest.mark.asyncio
async def test_aggregate_sweep_and_export_two_variations_three_trials(tmp_path, logger):
    """Two variations × 3 trials: ConfidenceAggregation runs inside each cell."""
    plan = _make_plan()
    results = []
    # concurrency=10: throughput jitters around 100
    for i, tput in enumerate([100.0, 105.0, 95.0]):
        r = _result(f"c10_t{i}", concurrency=10, throughput=tput, ttft_p99=50.0 + i)
        r.trial_index = i
        results.append(r)
    # concurrency=20: throughput jitters around 180
    for i, tput in enumerate([180.0, 175.0, 185.0]):
        r = _result(f"c20_t{i}", concurrency=20, throughput=tput, ttft_p99=80.0 + i)
        r.trial_index = i
        results.append(r)

    out_dir = await aggregate_sweep_and_export(results, plan, tmp_path, logger)
    assert out_dir is not None

    data = json.loads((out_dir / "profile_export_aiperf_sweep.json").read_text())
    assert len(data["per_combination_metrics"]) == 2

    # Multi-trial path: at least one metric has non-zero std.
    saw_nonzero_std = False
    for entry in data["per_combination_metrics"]:
        for stats in entry["metrics"].values():
            if stats.get("std", 0.0) > 0.0:
                saw_nonzero_std = True
    assert saw_nonzero_std, "expected aggregation across trials to produce non-zero std"


@pytest.mark.asyncio
async def test_aggregate_sweep_and_export_empty_results_no_crash(tmp_path, logger):
    """Empty results list: helper logs and returns None without writing files."""
    plan = _make_plan()

    out = await aggregate_sweep_and_export([], plan, tmp_path, logger)

    assert out is None
    assert not (tmp_path / "sweep_aggregate").exists()
