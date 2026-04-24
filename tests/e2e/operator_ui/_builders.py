# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case builders for the operator web UI e2e suite.

Most tests use the committed golden tree via ``seeded_results_dir``;
these builders produce synthetic trees for the three edge cases the
golden data doesn't represent.

The summary JSON schema here mirrors
``tests/fixtures/operator_ui/generate_golden.py`` — top-level metric
keys each hold ``{unit, avg, p50, p90, p99, min, max}`` — matching what
``aiperf.operator.results_db.ResultsDB`` actually queries.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import orjson


def _metric_entry(avg: float, unit: str) -> dict:
    """Shape DuckDB destructures into ``t.<metric>.{avg,p50,p99,min,max,unit}``."""
    return {
        "unit": unit,
        "avg": avg,
        "p50": avg * 0.95,
        "p90": avg * 1.10,
        "p99": avg * 1.30,
        "min": avg * 0.70,
        "max": avg * 1.50,
    }


def clear_results_dir(target: Path) -> None:
    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def build_empty(target: Path) -> None:
    clear_results_dir(target)


def build_single_job(target: Path, *, job_id: str, namespace: str) -> None:
    clear_results_dir(target)
    d = target / namespace / job_id
    d.mkdir(parents=True)
    summary = {
        "schema_version": "1.0.0",
        "aiperf_version": "0.0.0-e2e",
        "benchmark_id": f"{namespace}/{job_id}",
        "request_throughput": _metric_entry(1.0, "requests/sec"),
        "request_latency": _metric_entry(100.0, "ms"),
        "time_to_first_token": _metric_entry(50.0, "ms"),
        "inter_token_latency": _metric_entry(10.0, "ms"),
        "output_token_throughput": _metric_entry(128.0, "tokens/sec"),
        "start_time": "2026-04-22T10:00:00Z",
        "end_time": "2026-04-22T10:01:00Z",
        "input_config": {
            "models": {
                "strategy": "round_robin",
                "items": [{"name": "llama3-8b"}],
            },
            "endpoint": {
                "urls": ["http://llama3.svc:8000/v1"],
                "type": "chat",
                "streaming": True,
            },
            "runtime": {"concurrency": 1},
        },
        "status": "Succeeded",
    }
    (d / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(summary, option=orjson.OPT_INDENT_2)
    )
    (d / "conditions.json").write_bytes(
        orjson.dumps(
            [
                {"type": "Ready", "status": "True", "reason": "BenchmarkComplete"},
                {"type": "Succeeded", "status": "True"},
            ],
            option=orjson.OPT_INDENT_2,
        )
    )
    (d / ".aiperf_results_ready.json").write_bytes(
        orjson.dumps({"ready": True, "version": 1})
    )


def build_all_failed(target: Path, *, n: int = 3) -> None:
    clear_results_dir(target)
    for i in range(n):
        namespace = "aiperf-bench"
        job_id = f"failed-{i}"
        d = target / namespace / job_id
        d.mkdir(parents=True)
        summary = {
            "schema_version": "1.0.0",
            "aiperf_version": "0.0.0-e2e",
            "benchmark_id": f"{namespace}/{job_id}",
            "request_throughput": _metric_entry(0.0, "requests/sec"),
            "request_latency": _metric_entry(0.0, "ms"),
            "time_to_first_token": _metric_entry(0.0, "ms"),
            "inter_token_latency": _metric_entry(0.0, "ms"),
            "output_token_throughput": _metric_entry(0.0, "tokens/sec"),
            "start_time": "2026-04-22T12:00:00Z",
            "end_time": "2026-04-22T12:00:30Z",
            "input_config": {
                "models": {
                    "strategy": "round_robin",
                    "items": [{"name": "llama3-8b"}],
                },
                "endpoint": {
                    "urls": ["http://llama3.svc:8000/v1"],
                    "type": "chat",
                    "streaming": True,
                },
                "runtime": {"concurrency": 1},
            },
            "status": "Failed",
        }
        (d / "profile_export_aiperf.json").write_bytes(
            orjson.dumps(summary, option=orjson.OPT_INDENT_2)
        )
        (d / "conditions.json").write_bytes(
            orjson.dumps(
                [
                    {"type": "Ready", "status": "True", "reason": "BenchmarkFailed"},
                    {"type": "Succeeded", "status": "False"},
                ],
                option=orjson.OPT_INDENT_2,
            )
        )
        (d / ".aiperf_results_ready.json").write_bytes(
            orjson.dumps({"ready": True, "version": 1})
        )
