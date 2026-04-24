# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for service CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from aiperf.cli_commands.service import service
from aiperf.common.environment import Environment

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_bootstrap() -> Generator[MagicMock, None, None]:
    """Mock bootstrap_and_run_service."""
    with patch("aiperf.common.bootstrap.bootstrap_and_run_service") as mock:
        yield mock


@pytest.fixture
def mock_load_config() -> Generator[MagicMock, None, None]:
    """Mock load_config."""
    with patch("aiperf.config.loader.load_config") as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def service_type() -> MagicMock:
    """Create a mock ServiceType."""
    return MagicMock()


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

    def test_forwards_benchmark_run_and_service_id(
        self,
        mock_bootstrap: MagicMock,
        mock_load_config: MagicMock,
        service_type: MagicMock,
    ) -> None:
        """Test that benchmark_run_file and service_id are forwarded to bootstrap."""
        run_file = Path("/path/to/run.json")
        mock_run = MagicMock()

        with (
            patch.object(Path, "read_bytes", return_value=b'{"cfg": {}}'),
            patch("orjson.loads", return_value={"cfg": {}}) as mock_loads,
            patch(
                "aiperf.config.benchmark.BenchmarkRun.model_validate",
                return_value=mock_run,
            ) as mock_validate,
        ):
            service(
                service_type=service_type,
                benchmark_run_file=run_file,
                service_id="worker-1",
            )

        mock_loads.assert_called_once()
        mock_validate.assert_called_once_with({"cfg": {}})
        mock_bootstrap.assert_called_once_with(
            service_type=service_type,
            run=mock_run,
            config=None,
            service_id="worker-1",
            api_port=None,
        )

    def test_default_optional_arguments(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        """Test that optional arguments default to None."""
        service(service_type=service_type)

        call_kwargs = mock_bootstrap.call_args.kwargs
        assert call_kwargs["service_id"] is None
        assert call_kwargs["config"] is None

    def test_none_benchmark_run_file_passes_none_to_bootstrap(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        """Test that None benchmark_run_file passes None run to bootstrap."""
        service(service_type=service_type, benchmark_run_file=None)

        mock_bootstrap.assert_called_once_with(
            service_type=service_type,
            run=None,
            config=None,
            service_id=None,
            api_port=None,
        )

    def test_health_port_sets_environment(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        """Test that health_port sets Environment.SERVICE health settings."""
        service(service_type=service_type, health_port=9090)

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_PORT == 9090

    def test_health_host_sets_environment(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        """Test that health_host sets Environment.SERVICE health settings."""
        service(service_type=service_type, health_host="0.0.0.0")

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_HOST == "0.0.0.0"

    def test_health_host_and_port_set_environment(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        """Test that both health_host and health_port set Environment.SERVICE."""
        service(service_type=service_type, health_host="0.0.0.0", health_port=8081)

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_HOST == "0.0.0.0"
        assert Environment.SERVICE.HEALTH_PORT == 8081

    def test_none_health_args_do_not_modify_environment(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        """Test that None health args leave Environment.SERVICE unchanged."""
        original_enabled = Environment.SERVICE.HEALTH_ENABLED
        original_host = Environment.SERVICE.HEALTH_HOST
        original_port = Environment.SERVICE.HEALTH_PORT

        service(service_type=service_type, health_host=None, health_port=None)

        assert original_enabled == Environment.SERVICE.HEALTH_ENABLED
        assert original_host == Environment.SERVICE.HEALTH_HOST
        assert original_port == Environment.SERVICE.HEALTH_PORT

    def test_health_args_not_passed_to_bootstrap(
        self,
        mock_bootstrap: MagicMock,
        service_type: MagicMock,
    ) -> None:
        """Test that health args are not forwarded to bootstrap_and_run_service."""
        service(service_type=service_type, health_host="0.0.0.0", health_port=8080)

        call_kwargs = mock_bootstrap.call_args.kwargs
        assert "health_host" not in call_kwargs
        assert "health_port" not in call_kwargs
