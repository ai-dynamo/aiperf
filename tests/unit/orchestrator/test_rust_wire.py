# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.orchestrator.rust_wire import RustWireError, build_run_request


def _run(tmp_path: Path, *, dataset: dict | None = None, phases: list | None = None):
    envelope = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["mock-model"],
                "endpoint": {
                    "urls": ["http://127.0.0.1:8000/v1/chat/completions"],
                    "streaming": True,
                },
                "dataset": dataset
                or {
                    "type": "synthetic",
                    "entries": 8,
                    "prompts": {
                        "isl": {
                            "peaks": [
                                {"value": 8, "weight": 1},
                                {"mean": 16, "stddev": 2, "weight": 3},
                            ]
                        },
                        "osl": 2,
                    },
                    "turns": 2,
                },
                "phases": phases
                or [
                    {
                        "name": "profiling",
                        "type": "gamma",
                        "requests": 8,
                        "rate": 20,
                        "smoothness": 2,
                        "concurrency": 4,
                    }
                ],
                "artifacts": {"dir": str(tmp_path)},
            }
        }
    )
    return BenchmarkRun(
        benchmark_id="wire-test",
        cfg=envelope.benchmark,
        artifact_dir=tmp_path,
        label="cell",
        random_seed=9,
    )


def test_projection_is_explicit_and_canonicalizes_nested_distributions(
    tmp_path,
) -> None:
    request = build_run_request(_run(tmp_path))

    assert request["protocol_version"] == 1
    run = request["run"]
    assert run["benchmark_id"] == "wire-test"
    assert run["random_seed"] == 9
    assert run["phases"] == [
        {
            "type": "gamma",
            "name": "profiling",
            "exclude_from_results": False,
            "seamless": False,
            "requests": 8,
            "rate": 20.0,
            "concurrency": 4,
            "smoothness": 2.0,
        }
    ]
    assert run["dataset"]["prompts"]["isl"]["peaks"] == [
        {"distribution": {"value": 8.0}, "weight": 1.0},
        {
            "distribution": {"mean": 16.0, "stddev": 2.0},
            "weight": 3.0,
        },
    ]
    assert "adaptive_scale" not in run["phases"][0]
    assert run["metrics"] == {"slos": {}}
    assert run["artifacts"] == {
        "records_path": "profile_export.jsonl",
        "trace": False,
    }


def test_projects_slos_timeslices_and_custom_record_path(tmp_path) -> None:
    run = _run(tmp_path)
    run.cfg.slos = {"request_latency": 500.0, "time_to_first_token": 100.0}
    run.cfg.artifacts.slice_duration = 2.5
    run.cfg.artifacts.prefix = "search-samples"
    run.cfg.artifacts.trace = True

    projected = build_run_request(run)["run"]

    assert projected["metrics"] == {
        "slice_duration_seconds": 2.5,
        "slos": {"request_latency": 500.0, "time_to_first_token": 100.0},
    }
    assert projected["artifacts"] == {
        "records_path": "search-samples.jsonl",
        "trace": True,
    }


def test_projects_user_centric_and_fixed_schedule_variants(tmp_path) -> None:
    user = build_run_request(
        _run(
            tmp_path,
            phases=[
                {
                    "name": "profiling",
                    "type": "user_centric",
                    "sessions": 4,
                    "rate": 10,
                    "users": 2,
                    "concurrency": 3,
                }
            ],
        )
    )["run"]["phases"][0]
    assert user["type"] == "user_centric"
    assert user["users"] == 2
    assert user["rate"] == 10.0

    fixed = build_run_request(
        _run(
            tmp_path,
            phases=[
                {
                    "name": "profiling",
                    "type": "fixed_schedule",
                    "auto_offset": False,
                    "start_offset": 100,
                    "end_offset": 500,
                }
            ],
        )
    )["run"]["phases"][0]
    assert fixed == {
        "type": "fixed_schedule",
        "name": "profiling",
        "exclude_from_results": False,
        "seamless": False,
        "auto_offset": False,
        "start_offset": 100,
        "end_offset": 500,
    }


def test_rejects_dataset_shapes_not_yet_in_protocol(tmp_path) -> None:
    run = _run(
        tmp_path,
        dataset={
            "type": "file",
            "records": [{"text": "hello"}],
            "format": "single_turn",
        },
    )
    with pytest.raises(RustWireError, match="dataset type file"):
        build_run_request(run)
