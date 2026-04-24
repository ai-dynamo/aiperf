# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for router tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.models import MetricResult
from aiperf.config import AIPerfConfig, BenchmarkRun


def make_latency_metric(
    avg: float = 100.0,
    min: float = 50.0,
    max: float = 200.0,
    p50: float = 95.0,
    p95: float = 180.0,
    p99: float = 195.0,
) -> MetricResult:
    """Create a typical latency metric for testing."""
    return MetricResult(
        tag="latency",
        header="Latency",
        unit="ms",
        avg=avg,
        min=min,
        max=max,
        p50=p50,
        p95=p95,
        p99=p99,
    )


@pytest.fixture
def router_config() -> BenchmarkRun:
    """BenchmarkRun for router testing."""
    config = AIPerfConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
            }
        },
        phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
    )
    return BenchmarkRun(
        benchmark_id="test",
        cfg=config,
        artifact_dir=Path("/tmp/test"),
    )
