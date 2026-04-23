# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the committed golden fixture tree under tests/fixtures/operator_ui/.

Run once, commit the output. Re-run to refresh.

Usage:
    uv run python tests/fixtures/operator_ui/generate_golden.py

The JSON schema matches what ``aiperf.operator.results_db.ResultsDB`` actually
queries (verified against ``src/aiperf/operator/results_db.py``):

    - top-level metric keys (``request_throughput``, ``request_latency``,
      ``time_to_first_token``, ``inter_token_latency``,
      ``output_token_throughput``) each hold an object with
      ``{avg, p50, p90, p99, min, max, unit}``.
    - top-level ``start_time`` / ``end_time`` as ISO strings.
    - ``input_config.models.items[0].name`` and
      ``input_config.endpoint.urls[0]`` populated for leaderboard/history
      model + endpoint columns (DuckDB uses 1-based indexing: ``items[1]``).

No parquet is generated: nothing under ``src/aiperf/operator`` currently
queries ``profile_export_aiperf.parquet``. If a future feature grows a
parquet reader, extend this script.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import orjson

FIXTURES = Path(__file__).parent
RESULTS = FIXTURES / "results"
K8S = FIXTURES / "k8s"


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


def _write_job(
    namespace: str,
    job_id: str,
    *,
    model: str,
    endpoint_url: str,
    concurrency: int,
    request_throughput: float,
    ttft_ms: float,
    itl_ms: float,
    latency_ms: float,
    start_time: str,
    end_time: str,
    status: str = "Succeeded",
) -> None:
    d = RESULTS / namespace / job_id
    d.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "1.0.0",
        "aiperf_version": "0.0.0-e2e",
        "benchmark_id": f"{namespace}/{job_id}",
        "request_throughput": _metric_entry(request_throughput, "requests/sec"),
        "request_latency": _metric_entry(latency_ms, "ms"),
        "time_to_first_token": _metric_entry(ttft_ms, "ms"),
        "inter_token_latency": _metric_entry(itl_ms, "ms"),
        "output_token_throughput": _metric_entry(
            request_throughput * 128, "tokens/sec"
        ),
        "start_time": start_time,
        "end_time": end_time,
        "input_config": {
            "models": {
                "strategy": "round_robin",
                "items": [{"name": model}],
            },
            "endpoint": {
                "urls": [endpoint_url],
                "type": "chat",
                "streaming": True,
            },
            "runtime": {"concurrency": concurrency},
        },
        "status": status,
    }
    (d / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(summary, option=orjson.OPT_INDENT_2)
    )
    (d / "conditions.json").write_bytes(
        orjson.dumps(
            [
                {"type": "Ready", "status": "True", "reason": "BenchmarkComplete"},
                {"type": "Succeeded", "status": str(status == "Succeeded")},
            ],
            option=orjson.OPT_INDENT_2,
        )
    )
    (d / ".aiperf_results_ready.json").write_bytes(
        orjson.dumps({"ready": True, "version": 1})
    )


def _write_k8s_fixtures() -> None:
    K8S.mkdir(parents=True, exist_ok=True)
    jobs = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJobList",
        "items": [
            {
                "apiVersion": "aiperf.nvidia.com/v1alpha1",
                "kind": "AIPerfJob",
                "metadata": {
                    "name": name,
                    "namespace": ns,
                    "uid": f"uid-{name}",
                    "creationTimestamp": "2026-04-22T12:00:00Z",
                },
                "spec": {"model": "llama3"},
                "status": {
                    "phase": phase,
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            }
            for name, ns, phase in [
                ("aiperf-llama3-c128", "aiperf-bench", "Succeeded"),
                ("aiperf-llama3-c256", "aiperf-bench", "Succeeded"),
                ("mistral-7b-run1", "ml-lab", "Succeeded"),
                ("failed-run", "ml-lab", "Failed"),
                ("live-run", "aiperf-bench", "Running"),
            ]
        ],
    }
    (K8S / "jobs.json").write_bytes(orjson.dumps(jobs, option=orjson.OPT_INDENT_2))
    pods = {
        "items": [
            {
                "metadata": {
                    "name": "live-run-controller-0",
                    "namespace": "aiperf-bench",
                    "labels": {"aiperf.nvidia.com/job-id": "live-run"},
                },
                "status": {
                    "phase": "Running",
                    "containerStatuses": [{"ready": True, "restartCount": 0}],
                },
            },
        ]
    }
    (K8S / "pods.json").write_bytes(orjson.dumps(pods, option=orjson.OPT_INDENT_2))
    (K8S / "version.json").write_bytes(
        orjson.dumps(
            {"gitVersion": "v1.29.0", "platform": "linux/amd64"},
            option=orjson.OPT_INDENT_2,
        )
    )


def main() -> None:
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    _write_job(
        "aiperf-bench",
        "aiperf-llama3-c128",
        model="llama3-8b",
        endpoint_url="http://llama3.svc:8000/v1",
        concurrency=128,
        request_throughput=42.1,
        ttft_ms=150.0,
        itl_ms=25.0,
        latency_ms=300.0,
        start_time="2026-04-22T10:00:00Z",
        end_time="2026-04-22T10:05:00Z",
    )
    _write_job(
        "aiperf-bench",
        "aiperf-llama3-c256",
        model="llama3-8b",
        endpoint_url="http://llama3.svc:8000/v1",
        concurrency=256,
        request_throughput=78.4,
        ttft_ms=220.0,
        itl_ms=32.0,
        latency_ms=410.0,
        start_time="2026-04-22T10:10:00Z",
        end_time="2026-04-22T10:15:00Z",
    )
    _write_job(
        "ml-lab",
        "mistral-7b-run1",
        model="mistral-7b",
        endpoint_url="http://mistral.svc:8000/v1",
        concurrency=64,
        request_throughput=28.9,
        ttft_ms=180.0,
        itl_ms=28.0,
        latency_ms=340.0,
        start_time="2026-04-22T11:00:00Z",
        end_time="2026-04-22T11:04:00Z",
    )
    _write_job(
        "ml-lab",
        "failed-run",
        model="mistral-7b",
        endpoint_url="http://mistral.svc:8000/v1",
        concurrency=16,
        request_throughput=0.0,
        ttft_ms=0.0,
        itl_ms=0.0,
        latency_ms=0.0,
        start_time="2026-04-22T12:00:00Z",
        end_time="2026-04-22T12:00:30Z",
        status="Failed",
    )
    _write_k8s_fixtures()
    print(f"Wrote golden tree under {FIXTURES}")


if __name__ == "__main__":
    main()
