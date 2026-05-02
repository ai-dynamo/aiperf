# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the post-process hook in aggregate_sweep_and_export."""

from __future__ import annotations

import logging
from pathlib import Path

import orjson
import pytest

from aiperf._cli_runner_sweep_helpers import aggregate_sweep_and_export
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config import BenchmarkConfig, BenchmarkPlan
from aiperf.config.sweep import SweepVariation
from aiperf.orchestrator.models import RunResult
from aiperf.search_recipes._base import PostProcessSpec

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


def _config(concurrency: int) -> BenchmarkConfig:
    kwargs = dict(_MINIMAL_CONFIG_KWARGS)
    kwargs["phases"] = [
        {
            "name": "default",
            "type": "concurrency",
            "requests": 100,
            "concurrency": concurrency,
        },
    ]
    return BenchmarkConfig(**kwargs)


def _make_plan(
    *, configs: list[BenchmarkConfig], post_process: PostProcessSpec | None = None
) -> BenchmarkPlan:
    return BenchmarkPlan(
        configs=configs,
        variations=[
            SweepVariation(index=i, label=f"v{i}", values={})
            for i in range(len(configs))
        ],
        trials=1,
        post_process=post_process,
    )


def _result_for(
    label: str, concurrency: int, request_latency_p99: float, throughput_avg: float
) -> RunResult:
    return RunResult(
        label=label,
        success=True,
        summary_metrics={
            "request_latency": JsonMetricResult(
                unit="ms", avg=request_latency_p99, p99=request_latency_p99
            ),
            "request_throughput": JsonMetricResult(
                unit="requests/sec", avg=throughput_avg
            ),
        },
        variation_label=label,
        variation_values={"phases.profiling.concurrency": concurrency},
        trial_index=0,
    )


def _logger() -> logging.Logger:
    return logging.getLogger("test")


@pytest.mark.asyncio
async def test_post_process_hook_writes_artifact(tmp_path: Path):
    spec = PostProcessSpec(
        handler="degradation_knee_detect",
        params={
            "threshold_pct": 0.20,
            "metric_tag": "request_latency",
            "stat": "p99",
            "swept_param": "phases.profiling.concurrency",
        },
        output_filename="degradation_knee.json",
    )
    plan = _make_plan(
        configs=[_config(c) for c in (1, 50, 200)],
        post_process=spec,
    )
    results = [
        _result_for("v0", 1, 10.0, 100.0),
        _result_for("v1", 50, 11.0, 200.0),
        _result_for("v2", 200, 14.0, 250.0),
    ]
    out_dir = await aggregate_sweep_and_export(results, plan, tmp_path, _logger())
    assert out_dir is not None
    artifact = out_dir / "degradation_knee.json"
    assert artifact.exists()
    payload = orjson.loads(artifact.read_bytes())
    assert payload["baseline_concurrency"] == 1
    assert payload["knee_concurrency"] == 200


@pytest.mark.asyncio
async def test_post_process_hook_quarantines_handler_failure(tmp_path: Path):
    # Bad params on purpose -- handler raises KeyError; standard artifacts must still land.
    spec = PostProcessSpec(
        handler="degradation_knee_detect",
        params={"this_is_wrong": True},
        output_filename="degradation_knee.json",
    )
    plan = _make_plan(
        configs=[_config(c) for c in (1, 50)],
        post_process=spec,
    )
    results = [
        _result_for("v0", 1, 10.0, 100.0),
        _result_for("v1", 50, 11.0, 200.0),
    ]
    out_dir = await aggregate_sweep_and_export(results, plan, tmp_path, _logger())
    assert out_dir is not None
    # Standard artifacts present.
    assert (out_dir / "profile_export_aiperf_sweep.json").exists()
    assert (out_dir / "profile_export_aiperf_sweep.csv").exists()
    # Sidecar errors file written.
    errors_path = out_dir / "post_process_errors.json"
    assert errors_path.exists()
    payload = orjson.loads(errors_path.read_bytes())
    assert payload["handler"] == "degradation_knee_detect"
    assert "error" in payload


@pytest.mark.asyncio
async def test_post_process_hook_skipped_when_no_spec(tmp_path: Path):
    plan = _make_plan(configs=[_config(c) for c in (1, 50)])
    results = [
        _result_for("v0", 1, 10.0, 100.0),
        _result_for("v1", 50, 11.0, 200.0),
    ]
    out_dir = await aggregate_sweep_and_export(results, plan, tmp_path, _logger())
    assert out_dir is not None
    assert not (out_dir / "post_process_errors.json").exists()
    # No degradation_knee.json since there is no spec.
    assert not (out_dir / "degradation_knee.json").exists()


@pytest.mark.asyncio
async def test_post_process_hook_quarantines_unregistered_handler(tmp_path: Path):
    """Handler name that isn't in the plugin registry should land in errors,
    not crash the sweep. Standard artifacts must still be written.

    Locks in the BLE001 quarantine in run_post_process_hook so a regression
    (e.g. someone removing the try/except for being "too broad") surfaces here.
    """
    spec = PostProcessSpec(
        handler="not_a_real_handler",
        params={},
        output_filename="never_written.json",
    )
    plan = _make_plan(
        configs=[_config(c) for c in (1, 50)],
        post_process=spec,
    )
    results = [
        _result_for("v0", 1, 10.0, 100.0),
        _result_for("v1", 50, 11.0, 200.0),
    ]
    out_dir = await aggregate_sweep_and_export(results, plan, tmp_path, _logger())
    assert out_dir is not None
    # Standard artifacts survived.
    assert (out_dir / "profile_export_aiperf_sweep.json").exists()
    assert (out_dir / "profile_export_aiperf_sweep.csv").exists()
    # Output file the bogus handler claimed was never written.
    assert not (out_dir / "never_written.json").exists()
    # Sidecar errors file names the missing handler.
    errors_path = out_dir / "post_process_errors.json"
    assert errors_path.exists()
    payload = orjson.loads(errors_path.read_bytes())
    assert payload["handler"] == "not_a_real_handler"
    assert "error" in payload
