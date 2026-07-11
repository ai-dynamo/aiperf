# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.orchestrator.rust_wire import build_run_request


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


def test_projects_inline_file_dataset_through_native_registry(tmp_path) -> None:
    run = _run(
        tmp_path,
        dataset={
            "type": "file",
            "records": [{"text": "hello", "output_length": 4}],
            "format": "single_turn",
            "entries": 1,
            "sampling": "random",
            "osl": 7,
        },
    )

    request = build_run_request(run)["run"]

    assert request["dataset"] == {
        "type": "file",
        "format": "single_turn",
        "sampling": "random",
        "options": {},
        "entries": 1,
        "osl": {"value": 7.0},
        "records": [{"text": "hello", "output_length": 4}],
    }
    assert request["tokenizer"] == {"name": "builtin"}


def test_projects_complete_synthetic_dataset_shape(tmp_path) -> None:
    run = _run(
        tmp_path,
        dataset={
            "type": "synthetic",
            "entries": 3,
            "random_seed": 41,
            "sampling": "shuffle",
            "prompts": {
                "isl": 12,
                "osl": 5,
                "block_size": 16,
                "batch_size": 2,
                "sequence_distribution": [
                    {"isl": 12, "osl": 5, "probability": 40},
                    {
                        "isl": {"mean": 24, "stddev": 2},
                        "osl": {"mean": 7, "stddev": 1},
                        "probability": 60,
                    },
                ],
            },
            "prefix_prompts": {
                "shared_system_length": 4,
                "user_context_length": 3,
            },
            "turns": 2,
            "turn_delay": 7,
            "turn_delay_ratio": 0.5,
            "images": {
                "batch_size": 1,
                "width": 8,
                "height": 6,
                "format": "png",
                "source": "noise",
                "source_sampling": "random-with-replacement",
            },
            "audio": {
                "batch_size": 1,
                "length": 0.02,
                "format": "wav",
                "sample_rates": [16.0],
                "depths": [16],
                "channels": 1,
            },
            "video": {
                "batch_size": 1,
                "duration": 0.25,
                "fps": 4,
                "width": 8,
                "height": 6,
                "format": "webm",
                "codec": "libvpx-vp9",
                "synth_type": "grid_clock",
                "audio": {
                    "sample_rate": 44.1,
                    "channels": 1,
                    "codec": "libvorbis",
                    "depth": 16,
                },
            },
            "rankings": {
                "passages": 3,
                "passage_tokens": 9,
                "query_tokens": 4,
            },
        },
    )

    dataset = build_run_request(run)["run"]["dataset"]

    assert dataset["random_seed"] == 41
    assert dataset["sampling"] == "shuffle"
    assert dataset["prompts"]["sequence_distribution"][1] == {
        "isl": {"mean": 24.0, "stddev": 2.0},
        "osl": {"mean": 7.0, "stddev": 1.0},
        "probability": 60.0,
    }
    assert dataset["prefix_prompts"] == {
        "shared_system_length": 4,
        "user_context_length": 3,
    }
    assert dataset["images"]["source_sampling"] == "random-with-replacement"
    assert dataset["audio"]["sample_rates"] == [16.0]
    assert dataset["video"]["audio"] == {
        "sample_rate": 44.1,
        "channels": 1,
        "depth": 16,
        "codec": "libvorbis",
    }
    assert dataset["rankings"]["passages"] == {"value": 3.0}
