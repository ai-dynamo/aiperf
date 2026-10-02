# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for router tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.common.models import MetricResult
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.config.flags.cli_config import CLIConfig
from tests.unit.conftest import make_run_from_cli


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
        benchmark={
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
                    "kind": "profiling",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        }
    )
    return BenchmarkRun(
        benchmark_id="test",
        cfg=config.benchmark,
        artifact_dir=Path("/tmp/test"),
    )


@pytest.fixture
def router_service_config() -> CLIConfig:
    """CLIConfig for router testing."""
    return CLIConfig(api_port=9999, api_host="127.0.0.1")


@pytest.fixture
def router_cfg() -> CLIConfig:
    """CLIConfig for router testing."""
    return CLIConfig(model_names=["test-model"])


@pytest.fixture
def router_benchmark_run(
    router_cfg: CLIConfig, router_service_config: CLIConfig
) -> BenchmarkRun:
    """BenchmarkRun for router testing."""
    run = make_run_from_cli(router_cfg)
    run.benchmark_id = "test-bench"
    return run
