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
    # Patched at source; works because service() uses lazy imports inside the function body.
    with patch("aiperf.common.bootstrap.bootstrap_and_run_service") as mock:
        yield mock


@pytest.fixture
def mock_loaders() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """Mock both config loaders."""
    with (
        # Patched at source; works because service() uses lazy imports inside the function body.
        patch("aiperf.common.config.loader.load_user_config") as mock_user,
        patch("aiperf.common.config.loader.load_service_config") as mock_service,
    ):
        mock_user.return_value = MagicMock()
        mock_service.return_value = MagicMock()
        yield mock_user, mock_service


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

    def test_forwards_all_arguments(
        self,
        mock_bootstrap: MagicMock,
        mock_loaders: tuple[MagicMock, MagicMock],
        service_type: MagicMock,
    ) -> None:
        """Test that config files and service_id are forwarded to bootstrap."""
        mock_load_user, mock_load_service = mock_loaders
        user_file = Path("/path/to/user.yaml")
        service_file = Path("/path/to/service.yaml")

        service(
            service_type=service_type,
            user_config_file=user_file,
            service_config_file=service_file,
            service_id="worker-1",
        )

        mock_load_user.assert_called_once_with(user_file)
        mock_load_service.assert_called_once_with(service_file)
        mock_bootstrap.assert_called_once_with(
            service_type=service_type,
            service_config=mock_load_service.return_value,
            user_config=mock_load_user.return_value,
            service_id="worker-1",
        )

    def test_default_optional_arguments(
        self,
        mock_bootstrap: MagicMock,
        mock_loaders: tuple[MagicMock, MagicMock],
        service_type: MagicMock,
    ) -> None:
        """Test that optional arguments default to None."""
        service(service_type=service_type)

        call_kwargs = mock_bootstrap.call_args.kwargs
        assert call_kwargs["service_id"] is None

    def test_none_config_files_passed_to_loaders(
        self,
        mock_bootstrap: MagicMock,
        mock_loaders: tuple[MagicMock, MagicMock],
        service_type: MagicMock,
    ) -> None:
        """Test that None config file paths are passed to loaders for env var fallback."""
        mock_load_user, mock_load_service = mock_loaders

        service(
            service_type=service_type, user_config_file=None, service_config_file=None
        )

        mock_load_user.assert_called_once_with(None)
        mock_load_service.assert_called_once_with(None)

    def test_health_port_sets_environment(
        self,
        mock_bootstrap: MagicMock,
        mock_loaders: tuple[MagicMock, MagicMock],
        service_type: MagicMock,
    ) -> None:
        """Test that health_port sets Environment.SERVICE health settings."""
        service(service_type=service_type, health_port=9090)

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_PORT == 9090

    def test_health_host_sets_environment(
        self,
        mock_bootstrap: MagicMock,
        mock_loaders: tuple[MagicMock, MagicMock],
        service_type: MagicMock,
    ) -> None:
        """Test that health_host sets Environment.SERVICE health settings."""
        service(service_type=service_type, health_host="0.0.0.0")

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_HOST == "0.0.0.0"

    def test_health_host_and_port_set_environment(
        self,
        mock_bootstrap: MagicMock,
        mock_loaders: tuple[MagicMock, MagicMock],
        service_type: MagicMock,
    ) -> None:
        """Test that both health_host and health_port set Environment.SERVICE health settings."""
        service(service_type=service_type, health_host="0.0.0.0", health_port=8081)

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_HOST == "0.0.0.0"
        assert Environment.SERVICE.HEALTH_PORT == 8081

    def test_none_health_args_do_not_modify_environment(
        self,
        mock_bootstrap: MagicMock,
        mock_loaders: tuple[MagicMock, MagicMock],
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
        mock_loaders: tuple[MagicMock, MagicMock],
        service_type: MagicMock,
    ) -> None:
        """Test that health args are not forwarded to bootstrap_and_run_service."""
        service(service_type=service_type, health_host="0.0.0.0", health_port=8080)

        call_kwargs = mock_bootstrap.call_args.kwargs
        assert "health_host" not in call_kwargs
        assert "health_port" not in call_kwargs
