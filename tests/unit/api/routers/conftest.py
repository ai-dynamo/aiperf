# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for router tests."""

from __future__ import annotations

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.models import MetricResult


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
def router_service_config() -> ServiceConfig:
    """ServiceConfig for router testing."""
    return ServiceConfig(api_port=9999, api_host="127.0.0.1")


@pytest.fixture
def router_user_config() -> UserConfig:
    """UserConfig for router testing."""
    return UserConfig(
        benchmark_id="test-bench",
        endpoint=EndpointConfig(model_names=["test-model"]),
    )
