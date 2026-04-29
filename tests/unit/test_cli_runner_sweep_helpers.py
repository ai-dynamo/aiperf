# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf._cli_runner_sweep_helpers.aggregate_sweep_and_export."""

from __future__ import annotations

import json
import logging

import pytest

from aiperf._cli_runner_sweep_helpers import (
    _per_variation_aggregate_dir,
    aggregate_per_variation_and_export,
    aggregate_sweep_and_export,
)
from aiperf.common.enums import SweepMode
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


def _make_plan_mode(mode: SweepMode) -> BenchmarkPlan:
    """Plan with explicit parameter_sweep_mode for per-variation path tests."""
    cfg = BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS)
    return BenchmarkPlan(configs=[cfg], parameter_sweep_mode=mode)


def test_per_variation_aggregate_dir_independent_mode():
    """Independent mode -> ``<base>/<variation_label>/aggregate/``."""
    from pathlib import Path

    out = _per_variation_aggregate_dir(
        Path("/tmp/x"),
        "phases.profiling.concurrency=10",
        SweepMode.INDEPENDENT,
    )
    assert out == Path("/tmp/x/phases.profiling.concurrency=10/aggregate")


def test_per_variation_aggregate_dir_repeated_mode():
    """Repeated mode -> ``<base>/aggregate/<variation_label>/``."""
    from pathlib import Path

    out = _per_variation_aggregate_dir(
        Path("/tmp/x"),
        "phases.profiling.concurrency=10",
        SweepMode.REPEATED,
    )
    assert out == Path("/tmp/x/aggregate/phases.profiling.concurrency=10")


@pytest.mark.asyncio
async def test_aggregate_per_variation_writes_aggregate_per_cell_independent(
    tmp_path, logger
):
    """Independent mode: 2 variations × 3 trials -> 2 aggregate dirs under cells."""
    plan = _make_plan_mode(SweepMode.INDEPENDENT)
    results = []
    for i, tput in enumerate([100.0, 105.0, 95.0]):
        r = _result(f"c10_t{i}", concurrency=10, throughput=tput, ttft_p99=50.0 + i)
        r.variation_label = "phases.profiling.concurrency=10"
        r.variation_values = {"phases.profiling.concurrency": 10}
        r.trial_index = i
        results.append(r)
    for i, tput in enumerate([180.0, 175.0, 185.0]):
        r = _result(f"c20_t{i}", concurrency=20, throughput=tput, ttft_p99=80.0 + i)
        r.variation_label = "phases.profiling.concurrency=20"
        r.variation_values = {"phases.profiling.concurrency": 20}
        r.trial_index = i
        results.append(r)

    written = await aggregate_per_variation_and_export(results, plan, tmp_path, logger)
    assert len(written) == 2

    for concurrency in (10, 20):
        agg_dir = tmp_path / f"phases.profiling.concurrency={concurrency}" / "aggregate"
        agg_json = agg_dir / "profile_export_aiperf_aggregate.json"
        agg_csv = agg_dir / "profile_export_aiperf_aggregate.csv"
        assert agg_json.exists(), f"missing per-variation aggregate JSON: {agg_json}"
        assert agg_csv.exists(), f"missing per-variation aggregate CSV: {agg_csv}"

        data = json.loads(agg_json.read_text())
        # AggregateConfidenceJsonExporter flattens our AggregateResult
        # metadata into a single ``metadata`` block, with the run counts
        # bumped up under ``num_profile_runs`` / ``num_successful_runs``.
        meta = data["metadata"]
        assert meta["aggregation_type"] == "confidence"
        assert meta["num_profile_runs"] == 3
        assert meta["num_successful_runs"] == 3
        assert meta["variation_label"] == (
            f"phases.profiling.concurrency={concurrency}"
        )
        assert str(meta["sweep_mode"]).lower() == "independent"


@pytest.mark.asyncio
async def test_aggregate_per_variation_writes_aggregate_per_cell_repeated(
    tmp_path, logger
):
    """Repeated mode: per-variation dirs land under ``<base>/aggregate/<label>/``."""
    plan = _make_plan_mode(SweepMode.REPEATED)
    results = []
    for i, tput in enumerate([100.0, 105.0]):
        r = _result(f"c10_t{i}", concurrency=10, throughput=tput, ttft_p99=50.0 + i)
        r.variation_label = "phases.profiling.concurrency=10"
        r.variation_values = {"phases.profiling.concurrency": 10}
        r.trial_index = i
        results.append(r)
    for i, tput in enumerate([180.0, 175.0]):
        r = _result(f"c20_t{i}", concurrency=20, throughput=tput, ttft_p99=80.0 + i)
        r.variation_label = "phases.profiling.concurrency=20"
        r.variation_values = {"phases.profiling.concurrency": 20}
        r.trial_index = i
        results.append(r)

    written = await aggregate_per_variation_and_export(results, plan, tmp_path, logger)
    assert len(written) == 2

    for concurrency in (10, 20):
        agg_dir = tmp_path / "aggregate" / f"phases.profiling.concurrency={concurrency}"
        assert (agg_dir / "profile_export_aiperf_aggregate.json").exists()
        # Independent-mode layout must not be written.
        wrong = tmp_path / f"phases.profiling.concurrency={concurrency}" / "aggregate"
        assert not (wrong / "profile_export_aiperf_aggregate.json").exists()


@pytest.mark.asyncio
async def test_aggregate_per_variation_skips_below_minimum_runs(tmp_path, logger):
    """Single-trial cells (1 successful run) are skipped without crashing."""
    plan = _make_plan_mode(SweepMode.INDEPENDENT)
    r = _result("c10", concurrency=10, throughput=100.0, ttft_p99=50.0)
    r.variation_label = "phases.profiling.concurrency=10"
    r.variation_values = {"phases.profiling.concurrency": 10}

    written = await aggregate_per_variation_and_export([r], plan, tmp_path, logger)
    assert written == []
    # No aggregate dir should be created.
    assert not (
        tmp_path / "phases.profiling.concurrency=10" / "aggregate"
    ).exists()


@pytest.mark.asyncio
async def test_aggregate_per_variation_handles_partial_failures_per_cell(
    tmp_path, logger
):
    """One variation fully fails; the OTHER variation still produces an aggregate."""
    plan = _make_plan_mode(SweepMode.INDEPENDENT)
    results = []
    # concurrency=10: 2 successful trials.
    for i, tput in enumerate([100.0, 105.0]):
        r = _result(f"c10_t{i}", concurrency=10, throughput=tput, ttft_p99=50.0 + i)
        r.variation_label = "phases.profiling.concurrency=10"
        r.variation_values = {"phases.profiling.concurrency": 10}
        r.trial_index = i
        results.append(r)
    # concurrency=20: both failed -> no aggregate.
    for i in range(2):
        r = _result(
            f"c20_t{i}", concurrency=20, throughput=0.0, ttft_p99=0.0, success=False
        )
        r.error = "synthetic failure"
        r.variation_label = "phases.profiling.concurrency=20"
        r.variation_values = {"phases.profiling.concurrency": 20}
        r.trial_index = i
        results.append(r)

    written = await aggregate_per_variation_and_export(results, plan, tmp_path, logger)
    assert len(written) == 1

    success_dir = tmp_path / "phases.profiling.concurrency=10" / "aggregate"
    fail_dir = tmp_path / "phases.profiling.concurrency=20" / "aggregate"
    assert (success_dir / "profile_export_aiperf_aggregate.json").exists()
    assert not fail_dir.exists()
