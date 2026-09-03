# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for service CLI command."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import orjson
import pytest

from aiperf.cli_commands.service import app, service
from aiperf.common.environment import Environment
from aiperf.config import BenchmarkConfig, BenchmarkRun

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_bootstrap() -> Generator[MagicMock, None, None]:
    """Mock bootstrap_and_run_service."""
    # Patched at source; works because service() uses lazy imports inside the function body.
    with patch("aiperf.common.bootstrap.bootstrap_and_run_service") as mock:
        yield mock


@pytest.fixture
def service_type() -> MagicMock:
    """Create a mock ServiceType."""
    return MagicMock()


def _write_benchmark_run(
    path: Path,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    urls: list[str] | None = None,
) -> Path:
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {
                "urls": urls or ["http://server/v1"],
                "type": "chat",
                "api_key": api_key,
                "headers": headers or {},
            },
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 1,
                    "concurrency": 1,
                }
            ],
        }
    )
    run = BenchmarkRun(benchmark_id="job", cfg=cfg, artifact_dir=path.parent)
    path.write_bytes(orjson.dumps(run.model_dump(mode="json", exclude_none=True)))
    return path


@pytest.fixture
def benchmark_run_file(tmp_path: Path) -> Path:
    """Write a valid serialized run for service bootstrap tests."""
    return _write_benchmark_run(tmp_path / "run.json")


@pytest.fixture(autouse=True)
def _reset_health_settings() -> Generator[None, None, None]:
    """Reset Environment.SERVICE health settings after each test."""
    original_enabled = Environment.SERVICE.HEALTH_ENABLED
    original_host = Environment.SERVICE.HEALTH_HOST
    original_port = Environment.SERVICE.HEALTH_PORT
    yield
    Environment.SERVICE.HEALTH_ENABLED = original_enabled
    Environment.SERVICE.HEALTH_HOST = original_host
    Environment.SERVICE.HEALTH_PORT = original_port


class TestServiceCommand:
    """Tests for service() CLI function."""

    def test_requires_serialized_benchmark_run(self) -> None:
        """The service command has no per-process CLI config fallback."""
        signature = inspect.signature(service)

        assert "cli_config" not in signature.parameters
        assert (
            signature.parameters["benchmark_run_file"].default
            is inspect.Parameter.empty
        )

    def test_kubernetes_service_args_parse(self) -> None:
        """The operator's serialized-run invocation remains accepted by Cyclopts."""
        command, _bound, _ignored = app.parse_args(
            [
                "--type",
                "worker",
                "--benchmark-run",
                "/etc/aiperf/run_config.json",
            ]
        )

        assert command is service

    def test_forwards_all_arguments(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that service_id is forwarded to bootstrap."""
        service(
            service_type=service_type,
            benchmark_run_file=benchmark_run_file,
            service_id="worker-1",
        )

        mock_bootstrap.assert_called_once()
        call_kwargs = mock_bootstrap.call_args.kwargs
        assert call_kwargs["service_type"] is service_type
        assert call_kwargs["service_id"] == "worker-1"
        assert "run" in call_kwargs

    def test_default_optional_arguments(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that optional arguments default to None."""
        service(service_type=service_type, benchmark_run_file=benchmark_run_file)

        call_kwargs = mock_bootstrap.call_args.kwargs
        assert call_kwargs["service_id"] is None

    def test_health_port_sets_environment(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that health_port sets Environment.SERVICE health settings."""
        service(
            service_type=service_type,
            benchmark_run_file=benchmark_run_file,
            health_port=9090,
        )

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_PORT == 9090

    def test_health_host_sets_environment(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that health_host sets Environment.SERVICE health settings."""
        service(
            service_type=service_type,
            benchmark_run_file=benchmark_run_file,
            health_host="0.0.0.0",
        )

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_HOST == "0.0.0.0"

    def test_health_host_and_port_set_environment(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that both health_host and health_port set Environment.SERVICE health settings."""
        service(
            service_type=service_type,
            benchmark_run_file=benchmark_run_file,
            health_host="0.0.0.0",
            health_port=8081,
        )

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_HOST == "0.0.0.0"
        assert Environment.SERVICE.HEALTH_PORT == 8081

    def test_none_health_args_do_not_modify_environment(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that None health args leave Environment.SERVICE unchanged."""
        original_enabled = Environment.SERVICE.HEALTH_ENABLED
        original_host = Environment.SERVICE.HEALTH_HOST
        original_port = Environment.SERVICE.HEALTH_PORT

        service(
            service_type=service_type,
            benchmark_run_file=benchmark_run_file,
            health_host=None,
            health_port=None,
        )

        assert original_enabled == Environment.SERVICE.HEALTH_ENABLED
        assert original_host == Environment.SERVICE.HEALTH_HOST
        assert original_port == Environment.SERVICE.HEALTH_PORT

    def test_health_args_not_passed_to_bootstrap(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
        benchmark_run_file: Path,
    ) -> None:
        """Test that health args are not forwarded to bootstrap_and_run_service."""
        service(
            service_type=service_type,
            benchmark_run_file=benchmark_run_file,
            health_host="0.0.0.0",
            health_port=8080,
        )

        call_kwargs = mock_bootstrap.call_args.kwargs
        assert "health_host" not in call_kwargs
        assert "health_port" not in call_kwargs

    def test_benchmark_run_rehydrates_all_secret_backed_endpoint_fields(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        run_file = _write_benchmark_run(
            tmp_path / "run.json",
            api_key="serialized-api-key",
            headers={"Authorization": "Bearer serialized-header"},
            urls=["http://user:password@server/v1"],
        )
        monkeypatch.setenv("OPENAI_API_KEY", "secret-api-key")
        monkeypatch.setenv(
            "AIPERF_INJECTED_HEADERS",
            '{"Authorization":"Bearer secret-header"}',
        )
        monkeypatch.setenv(
            "AIPERF_INJECTED_ENDPOINT_URLS",
            '["http://real-user:real-password@server/v1"]',
        )

        service(service_type=service_type, benchmark_run_file=run_file)

        endpoint = mock_bootstrap.call_args.kwargs["run"].cfg.endpoint
        assert endpoint.api_key == "secret-api-key"
        assert endpoint.headers["Authorization"] == "Bearer secret-header"
        assert endpoint.urls == ["http://real-user:real-password@server/v1"]
        assert "OPENAI_API_KEY" not in __import__("os").environ
        assert "AIPERF_INJECTED_HEADERS" not in __import__("os").environ
        assert "AIPERF_INJECTED_ENDPOINT_URLS" not in __import__("os").environ

    def test_private_api_key_injection_precedes_openai_alias(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        run_file = _write_benchmark_run(tmp_path / "run.json", api_key="serialized")
        monkeypatch.setenv("AIPERF_INJECTED_API_KEY", "private-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "compatibility-secret")

        service(service_type=service_type, benchmark_run_file=run_file)

        endpoint = mock_bootstrap.call_args.kwargs["run"].cfg.endpoint
        assert endpoint.api_key == "private-secret"

    def test_benchmark_run_rejects_unresolved_redacted_credentials(
        self,
        tmp_path: Path,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        run_file = _write_benchmark_run(tmp_path / "run.json", api_key="serialized")

        with pytest.raises(SystemExit):
            service(service_type=service_type, benchmark_run_file=run_file)

        mock_bootstrap.assert_not_called()

    def test_unreadable_benchmark_run_fails_before_bootstrap(
        self,
        tmp_path: Path,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        """Missing or unsafe run files cannot fall back to local CLI config."""
        with pytest.raises(SystemExit):
            service(
                service_type=service_type,
                benchmark_run_file=tmp_path / "missing.json",
            )

        mock_bootstrap.assert_not_called()
