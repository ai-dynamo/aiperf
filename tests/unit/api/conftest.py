# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for API tests."""

import pytest
from starlette.testclient import TestClient

from aiperf.api.api_service import FastAPIService
from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig


@pytest.fixture
def api_service_config() -> ServiceConfig:
    """Create a ServiceConfig for API service testing."""
    return ServiceConfig(api_port=9999, api_host="127.0.0.1")


@pytest.fixture
def api_user_config() -> UserConfig:
    """Create a UserConfig for API service testing."""
    return UserConfig(
        benchmark_id="test-bench",
        endpoint=EndpointConfig(model_names=["test-model"]),
    )


@pytest.fixture
def mock_fastapi_service(
    mock_zmq, api_service_config: ServiceConfig, api_user_config: UserConfig
) -> FastAPIService:
    """Create a FastAPIService instance for testing without starting the server."""
    return FastAPIService(
        service_config=api_service_config,
        user_config=api_user_config,
        service_id="api-test-1",
    )


@pytest.fixture
def api_test_client(mock_fastapi_service: FastAPIService) -> TestClient:
    """Create a synchronous TestClient for HTTP testing."""
    return TestClient(mock_fastapi_service.app)
