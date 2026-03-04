# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the core API router (config, healthz, readyz)."""

import pytest
from pytest import param
from starlette.testclient import TestClient

from aiperf.api.api_service import FastAPIService
from aiperf.common.enums import LifecycleState


class TestConfigEndpoint:
    """Test the /api/config endpoint."""

    def test_config_returns_json(self, api_test_client: TestClient) -> None:
        """Test config endpoint returns JSON config."""
        response = api_test_client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "benchmark_id" in data


class TestHealthzEndpoint:
    """Test Kubernetes liveness probe /healthz."""

    @pytest.mark.parametrize(
        "state,expected_code,expected_text",
        [
            param(LifecycleState.RUNNING, 200, "ok", id="running-healthy"),
            param(LifecycleState.INITIALIZING, 200, "ok", id="initializing-healthy"),
            param(LifecycleState.STARTING, 200, "ok", id="starting-healthy"),
            param(LifecycleState.STOPPING, 200, "ok", id="stopping-healthy"),
            param(LifecycleState.STOPPED, 200, "ok", id="stopped-healthy"),
            param(LifecycleState.FAILED, 503, "unhealthy", id="failed-unhealthy"),
        ],
    )  # fmt: skip
    def test_healthz_by_state(
        self,
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
        state: LifecycleState,
        expected_code: int,
        expected_text: str,
    ) -> None:
        mock_fastapi_service._state = state
        response = api_test_client.get("/healthz")
        assert response.status_code == expected_code
        assert response.text == expected_text


class TestReadyzEndpoint:
    """Test Kubernetes readiness probe /readyz."""

    @pytest.mark.parametrize(
        "state,expected_code,expected_text",
        [
            param(LifecycleState.RUNNING, 200, "ok", id="running-ready"),
            param(LifecycleState.CREATED, 503, "not ready", id="created-not-ready"),
            param(LifecycleState.INITIALIZING, 503, "not ready", id="initializing-not-ready"),
            param(LifecycleState.INITIALIZED, 503, "not ready", id="initialized-not-ready"),
            param(LifecycleState.STARTING, 503, "not ready", id="starting-not-ready"),
            param(LifecycleState.STOPPING, 503, "not ready", id="stopping-not-ready"),
            param(LifecycleState.STOPPED, 503, "not ready", id="stopped-not-ready"),
            param(LifecycleState.FAILED, 503, "not ready", id="failed-not-ready"),
        ],
    )  # fmt: skip
    def test_readyz_by_state(
        self,
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
        state: LifecycleState,
        expected_code: int,
        expected_text: str,
    ) -> None:
        mock_fastapi_service._state = state
        response = api_test_client.get("/readyz")
        assert response.status_code == expected_code
        assert response.text == expected_text
