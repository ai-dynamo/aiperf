# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.config.accuracy import AccuracyConfig
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
    assert run["gpu_telemetry"] == {
        "collection_interval_ns": 333_000_000,
        "request_timeout_ns": 10_000_000_000,
        "records_path": "gpu_telemetry_export.jsonl",
        "sources": [
            {"type": "dcgm", "url": "http://localhost:9400/metrics"},
            {"type": "dcgm", "url": "http://localhost:9401/metrics"},
        ],
    }


def test_projects_gpu_telemetry_defaults_custom_urls_and_disable(tmp_path) -> None:
    run = _run(tmp_path)
    run.cfg.gpu_telemetry.urls = [
        "http://localhost:9400/",
        "http://gpu-node:9400",
    ]

    telemetry = build_run_request(run)["run"]["gpu_telemetry"]

    assert telemetry["sources"] == [
        {"type": "dcgm", "url": "http://localhost:9400/metrics"},
        {"type": "dcgm", "url": "http://localhost:9401/metrics"},
        {"type": "dcgm", "url": "http://gpu-node:9400/metrics"},
    ]

    run.cfg.gpu_telemetry.enabled = False
    assert "gpu_telemetry" not in build_run_request(run)["run"]


def test_projects_custom_dcgm_through_the_canonical_python_source(tmp_path) -> None:
    from aiperf.config.resolution.resolvers import GpuMetricsResolver

    metrics_file = tmp_path / "custom-metrics.csv"
    metrics_file.write_text(
        "DCGM_FI_DEV_SM_CLOCK,gauge,SM Clock Frequency (in MHz)\n"
    )
    run = _run(tmp_path)
    run.cfg.gpu_telemetry.metrics_file = metrics_file
    run.cfg.gpu_telemetry.urls = ["http://gpu-node:9400"]
    GpuMetricsResolver().resolve(run)

    telemetry = build_run_request(run)["run"]["gpu_telemetry"]

    assert telemetry["custom_metrics"] == [
        {
            "header": "SM Clock Frequency",
            "name": "sm_clock",
            "unit": "megahertz",
        }
    ]
    assert all(source["type"] == "python" for source in telemetry["sources"])
    assert telemetry["sources"][-1] == {
        "type": "python",
        "collector": "dcgm",
        "url": "http://gpu-node:9400/metrics",
        "metrics_file": str(metrics_file.resolve()),
        "python_executable": str(Path(sys.executable).absolute()),
        "worker_module": "aiperf.gpu_telemetry.worker",
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


def test_projects_outputs_json_into_the_native_results_layer(tmp_path) -> None:
    run = _run(tmp_path)
    run.cfg.artifacts.export_outputs_json = True

    projected = build_run_request(run)["run"]

    assert projected["artifacts"]["outputs_path"] == "outputs.json"


def test_projects_raw_records_as_a_distinct_native_artifact(tmp_path) -> None:
    run = _run(tmp_path)
    run.cfg.artifacts.raw = True

    projected = build_run_request(run)["run"]

    assert projected["artifacts"]["records_path"] == "profile_export.jsonl"
    assert projected["artifacts"]["raw_path"] == "profile_export_raw.jsonl"


def test_projects_complete_adaptive_scale_policy(tmp_path) -> None:
    phase = build_run_request(
        _run(
            tmp_path,
            phases=[
                {
                    "name": "profiling",
                    "type": "constant",
                    "duration": 30,
                    "rate": 20,
                    "concurrency": 8,
                    "adaptive_scale": {
                        "enabled": True,
                        "control": {"variable": "request_rate", "min": 2},
                        "assessment_period": 1,
                        "min_completed_requests": 3,
                        "sustain_duration": 2,
                        "strategy": {
                            "type": "ramp_until_fail",
                            "step_policy": "fixed_percent_step",
                            "step_percent": 50,
                        },
                    },
                    "sla": {
                        "request_latency": {"p95": {"le": 1000}},
                        "success_rate": {"avg": {"ge": 0.99}},
                    },
                }
            ],
        )
    )["run"]["phases"][0]

    assert phase["adaptive_scale"] == {
        "control_variable": "request_rate",
        "minimum": 2.0,
        "maximum": 20.0,
        "assessment_period_seconds": 1.0,
        "sustain_duration_seconds": 2.0,
        "min_completed_requests": 3,
        "strategy_type": "ramp_until_fail",
        "step_policy": "fixed_percent_step",
        "base_step": 10,
        "max_step_multiplier": 4,
        "step_percent": 50.0,
        "sla_filters": [
            {
                "metric_tag": "request_latency",
                "stat": "p95",
                "op": "le",
                "threshold": 1000.0,
            },
            {
                "metric_tag": "success_rate",
                "stat": "avg",
                "op": "ge",
                "threshold": 0.99,
            },
        ],
    }


def test_rejects_adaptive_sla_without_adaptive_controller(tmp_path) -> None:
    run = _run(
        tmp_path,
        phases=[
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": 1,
                "concurrency": 1,
                "sla": [
                    {
                        "metric_tag": "request_latency",
                        "stat": "p95",
                        "op": "le",
                        "threshold": 1000,
                    }
                ],
            }
        ],
    )

    with pytest.raises(RustWireError, match="without enabling adaptive_scale"):
        build_run_request(run)


def test_projects_accuracy_to_the_canonical_python_worker(tmp_path) -> None:
    run = _run(tmp_path)
    run.cfg.accuracy = AccuracyConfig(
        benchmark="mmlu",
        tasks=["abstract_algebra"],
        n_shots=3,
        enable_cot=False,
        grader="exact_match",
        system_prompt="Answer with one letter.",
        verbose=True,
    )

    accuracy = build_run_request(run)["run"]["accuracy"]

    assert accuracy == {
        "benchmark": "mmlu",
        "tasks": ["abstract_algebra"],
        "n_shots": 3,
        "enable_cot": False,
        "grader": "exact_match",
        "system_prompt": "Answer with one letter.",
        "python_executable": str(Path(sys.executable).absolute()),
        "worker_module": "aiperf.accuracy.worker",
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


def test_projects_chat_template_token_accounting(tmp_path) -> None:
    run = _run(tmp_path)
    run.cfg.tokenizer.apply_chat_template = True

    request = build_run_request(run)["run"]

    assert request["tokenizer"] == {
        "name": "builtin",
        "apply_chat_template": True,
    }


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


def test_resolves_public_dataset_plugins_to_native_sources(tmp_path) -> None:
    sharegpt = build_run_request(
        _run(
            tmp_path,
            dataset={
                "type": "public",
                "dataset": "sharegpt",
                "entries": 2,
                "random_seed": 17,
                "sampling": "shuffle",
            },
        )
    )["run"]["dataset"]
    assert sharegpt["type"] == "public"
    assert sharegpt["name"] == "sharegpt"
    assert sharegpt["format"] == "sharegpt"
    assert sharegpt["source"]["type"] == "url"
    assert sharegpt["source"]["url"].endswith(
        "ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    assert sharegpt["entries"] == 2
    assert sharegpt["random_seed"] == 17
    assert sharegpt["sampling"] == "shuffle"

    gsm8k = build_run_request(
        _run(
            tmp_path,
            dataset={
                "type": "public",
                "dataset": "spec_al_gsm8k",
                "entries": 3,
                "hf_subset": "main",
            },
        )
    )["run"]["dataset"]
    assert gsm8k["format"] == "hf_instruction_response"
    assert gsm8k["source"] == {
        "type": "hugging_face",
        "dataset": "openai/gsm8k",
        "subset": "main",
        "split": "test",
    }
    assert gsm8k["options"] == {
        "prompt_column": "question",
        "max_conversations": 3,
    }


def test_projects_exgentic_filters_fixed_schedule_and_pinned_revision(tmp_path) -> None:
    request = build_run_request(
        _run(
            tmp_path,
            dataset={
                "type": "public",
                "dataset": "exgentic_v2",
                "entries": 4,
                "filters": {
                    "harness": "claude_code",
                    "benchmark": "swebench",
                },
            },
            phases=[
                {
                    "name": "profiling",
                    "type": "fixed_schedule",
                    "requests": 4,
                }
            ],
        )
    )["run"]["dataset"]

    assert request["format"] == "exgentic_v2"
    assert request["source"] == {
        "type": "hugging_face",
        "dataset": "Exgentic/agent-llm-traces-v2",
        "subset": "default",
        "split": "train",
        "revision": "4b8ad4ab198438e5a170f9171c19c6a2cf7c1814",
    }
    assert request["options"] == {
        "harness": "claude_code",
        "benchmark": "swebench",
        "fixed_schedule": True,
        "max_conversations": 4,
    }


def test_projects_trace_synthesis_as_typed_native_policy(tmp_path) -> None:
    request = build_run_request(
        _run(
            tmp_path,
            dataset={
                "type": "file",
                "format": "mooncake_trace",
                "records": [
                    {
                        "session_id": "a",
                        "timestamp": 100,
                        "input_length": 10,
                        "output_length": 2,
                        "hash_ids": [1, 2],
                    },
                    {
                        "session_id": "b",
                        "timestamp": 200,
                        "input_length": 10,
                        "output_length": 3,
                        "hash_ids": [1, 3],
                    },
                ],
                "synthesis": {
                    "speedup_ratio": 2,
                    "prefix_len_multiplier": 2,
                    "prefix_root_multiplier": 1,
                    "prompt_len_multiplier": 1.5,
                    "output_len_multiplier": 1.5,
                    "max_isl": 64,
                    "max_osl": 8,
                },
            },
        )
    )["run"]["dataset"]

    assert request["format"] == "mooncake_trace"
    assert request["options"] == {"block_size": 512}
    assert request["synthesis"] == {
        "speedup_ratio": 2.0,
        "prefix_len_multiplier": 2.0,
        "prefix_root_multiplier": 1,
        "prompt_len_multiplier": 1.5,
        "output_len_multiplier": 1.5,
        "max_isl": 64,
        "max_osl": 8,
    }
