# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for API tests."""

import pytest
from starlette.testclient import TestClient

from aiperf.api.api_service import FastAPIService
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.resolution.plan import BenchmarkRun
from tests.unit.conftest import make_run_from_cli


@pytest.fixture
def api_service_config() -> CLIConfig:
    """Create a CLIConfig for API service testing."""
    return CLIConfig(api_port=9999, api_host="127.0.0.1")


@pytest.fixture
def api_cfg() -> CLIConfig:
    """Create a CLIConfig for API service testing."""
    return CLIConfig(model_names=["test-model"])


@pytest.fixture
def api_benchmark_run(
    api_cfg: CLIConfig, api_service_config: CLIConfig
) -> BenchmarkRun:
    """BenchmarkRun for API service testing, with api_host/api_port set."""
    run = make_run_from_cli(api_cfg)
    run.benchmark_id = "test-bench"
    run.cfg.runtime.api_host = "127.0.0.1"
    run.cfg.runtime.api_port = 9999
    return run


@pytest.fixture
def mock_fastapi_service(mock_zmq, api_benchmark_run: BenchmarkRun) -> FastAPIService:
    """Create a FastAPIService instance for testing without starting the server."""
    return FastAPIService(
        run=api_benchmark_run,
        service_id="api-test-1",
    )


@pytest.fixture
def api_test_client(mock_fastapi_service: FastAPIService) -> TestClient:
    """Create a synchronous TestClient for HTTP testing."""
    return TestClient(mock_fastapi_service.app)
