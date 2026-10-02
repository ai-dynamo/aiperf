# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario builders for dashboard-v2 Playwright tests."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from aiperf.config import AIPerfConfig


def dashboard_cfg(
    models: Sequence[str] | None = None,
    phases: Sequence[dict[str, Any]] | None = None,
    slos: dict[str, float] | None = None,
    endpoint_urls: Sequence[str] | None = None,
) -> AIPerfConfig:
    """Build a representative dashboard config with secret redaction coverage."""
    benchmark: dict[str, Any] = {
        "models": list(models or ["llama3-8b", "llama3-70b"]),
        "endpoint": {
            "urls": list(endpoint_urls or ["http://srv:8000/v1/chat/completions"]),
            "type": "chat",
            "streaming": True,
            "api_key": "SHOULD_NOT_LEAK",
        },
        "datasets": [
            {
                "name": "default",
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
            }
        ],
        "phases": list(
            phases
            or [
                {
                    "name": "warmup",
                    "type": "concurrency",
                    "requests": 50,
                    "concurrency": 4,
                },
                {
                    "name": "profiling",
                    "type": "poisson",
                    "rate": 20,
                    "duration": 300,
                    "concurrency": 32,
                },
            ]
        ),
        "runtime": {"api_port": 8080},
    }
    if slos is not None:
        benchmark["slos"] = slos
    return AIPerfConfig(benchmark=benchmark)


def metric_result(
    tag: str,
    header: str,
    unit: str,
    *,
    current: float | None = None,
    avg: float | None = None,
    p99: float | None = None,
    max: float | None = None,
    p50: float | None = None,
) -> dict[str, Any]:
    """Build the JSON shape emitted for a live ``MetricResult``."""
    return {
        "tag": tag,
        "header": header,
        "unit": unit,
        "count": 60,
        "current": current,
        "sum": None,
        "avg": avg,
        "p1": None,
        "p5": None,
        "p10": None,
        "p25": None,
        "p50": p50 if p50 is not None else avg,
        "p75": None,
        "p90": p99,
        "p95": None,
        "p99": p99,
        "min": None,
        "max": max,
        "std": None,
    }


def realtime_metrics_payload(*metrics: dict[str, Any]) -> dict[str, Any]:
    """Build a ``realtime_metrics`` WebSocket payload."""
    return {"type": "realtime_metrics", "metrics": list(metrics)}


def phase_start_payload(
    phase: str,
    *,
    total_expected_requests: int = 100,
    requests_completed: int = 0,
    start_ns: int | None = None,
) -> dict[str, Any]:
    """Build a ``credit_phase_start`` WebSocket payload."""
    stats: dict[str, Any] = {
        "start_ns": start_ns if start_ns is not None else time.time_ns(),
        "total_expected_requests": total_expected_requests,
    }
    if requests_completed:
        stats["requests_completed"] = requests_completed
    return {"type": "credit_phase_start", "phase": phase, "stats": stats}


def server_metrics_response() -> dict[str, Any]:
    """Build the default ``/api/server-metrics`` response body."""
    return {"endpoint_summaries": []}
