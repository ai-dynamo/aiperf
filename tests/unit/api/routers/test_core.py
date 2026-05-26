# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the core API router (config, run, healthz, readyz)."""

import sys

import pytest
from pytest import param
from starlette.testclient import TestClient

from aiperf.api.api_service import FastAPIService
from aiperf.common.enums import LifecycleState
from aiperf.config.flags.cli_config import CLIConfig
from tests.unit.conftest import make_run_from_cli


class TestConfigEndpoint:
    """Test the /api/config endpoint."""

    def test_config_returns_json(self, api_test_client: TestClient) -> None:
        """Test config endpoint returns JSON config."""
        response = api_test_client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "endpoint" in data
        assert "artifacts" in data

    def test_config_does_not_expose_run_identity(
        self, api_test_client: TestClient
    ) -> None:
        """/api/config is purely BenchmarkConfig. Run-identity fields belong on
        /api/run; this guards against re-introducing the pre-refactor grab-bag."""
        data = api_test_client.get("/api/config").json()
        assert "cli_command" not in data
        assert "benchmark_id" not in data
        assert "sweep_id" not in data


class TestRunEndpoint:
    """Test the /api/run endpoint."""

    def test_run_returns_identity_block(self, api_test_client: TestClient) -> None:
        """/api/run returns the RunInfo-shaped identity block, with cli_command
        populated from the test process's sys.argv via build_cli_command."""
        response = api_test_client.get("/api/run")
        assert response.status_code == 200
        data = response.json()
        assert data["benchmark_id"] == "test-bench"
        assert isinstance(data["cli_command"], str)
        assert data["cli_command"].startswith("aiperf")

    def test_run_does_not_expose_config_or_secrets(
        self, api_test_client: TestClient
    ) -> None:
        """/api/run exposes run-identity only — never the BenchmarkConfig or
        secrets that live inside it."""
        data = api_test_client.get("/api/run").json()
        assert "cfg" not in data
        assert "endpoint" not in data
        assert "api_key" not in data
        assert "artifact_dir" not in data
        assert "variables" not in data

    def test_run_does_not_leak_api_key_in_cli_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_zmq,
        api_cfg: CLIConfig,
    ) -> None:
        """End-to-end: ``--api-key <secret>`` in sys.argv must not leak through
        ``/api/run.cli_command``. Mirrors the QA ``test_api_key_redaction``
        contract by constructing a fresh ``BenchmarkRun`` under a sys.argv
        containing a secret, then asserting the secret never appears in the
        API response body."""
        secret = "sk-secret-DO-NOT-LEAK-9876543210"
        monkeypatch.setattr(
            sys,
            "argv",
            ["aiperf", "profile", "--model", "test-model", "--api-key", secret],
        )
        run = make_run_from_cli(api_cfg)
        run.benchmark_id = "test-bench"
        run.cfg.runtime.api_host = "127.0.0.1"
        run.cfg.runtime.api_port = 9999

        service = FastAPIService(run=run, service_id="api-redaction-test")
        client = TestClient(service.app)

        response = client.get("/api/run")
        assert response.status_code == 200
        assert secret not in response.text

        cli_command = response.json()["cli_command"]
        assert "--api-key" in cli_command
        assert "<redacted>" in cli_command

    def test_run_omits_unset_optional_fields(self, api_test_client: TestClient) -> None:
        """The endpoint promises shape parity with ``run_info`` in
        ``profile_export_aiperf.json``, which uses ``exclude_none=True``.
        Optional ``RunInfo`` fields that the fixture doesn't populate
        (``sweep_id``, ``random_seed``, ``variation_*``) must be omitted from
        the response — not serialized as ``null``."""
        data = api_test_client.get("/api/run").json()
        for field in (
            "sweep_id",
            "random_seed",
            "variation_label",
            "variation_index",
            "variation_values",
        ):
            assert field not in data, (
                f"{field} should be omitted when unset, not serialized as null"
            )

    def test_run_returns_503_when_no_active_run(
        self,
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
    ) -> None:
        """The router has an explicit 503 branch for missing run context. The
        type system says ``svc.run`` is non-Optional, so this branch is
        effectively defensive — the test locks the contract for any future
        change that makes ``run`` optional or nullable."""
        mock_fastapi_service.run = None
        response = api_test_client.get("/api/run")
        assert response.status_code == 503
        assert response.json() == {"detail": "No active benchmark run."}


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
