# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for service CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from aiperf.cli_commands.service import service
from aiperf.common.environment import Environment
from aiperf.config.flags import CLIConfig

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_bootstrap() -> Generator[MagicMock, None, None]:
    """Mock bootstrap_and_run_service."""
    # Patched at source; works because service() uses lazy imports inside the function body.
    with patch("aiperf.common.bootstrap.bootstrap_and_run_service") as mock:
        yield mock


@pytest.fixture
def mock_loaders() -> Generator[MagicMock, None, None]:
    """Mock the v1->v2 conversion + plan-building path."""
    with (
        # Patched at source; works because service() uses lazy imports inside the function body.
        patch("aiperf.config.flags.resolver.resolve_config"),
        patch("aiperf.config.loader.build_benchmark_plan") as mock_plan,
        patch("aiperf.cli_runner._make_benchmark_run"),
    ):
        # build_benchmark_plan returns an object with .configs[0], .variations[0],
        # and .variation_seeds; service.py uses resolve_run_seed which
        # reads variation.index, plan.variation_seeds, plan.random_seed, and
        # plan.multi_run.vary_seed_per_trial.
        from aiperf.config.sweep import SweepVariation

        mock_multi_run = MagicMock()
        mock_multi_run.vary_seed_per_trial = False
        mock_plan.return_value = MagicMock(
            configs=[MagicMock()],
            variations=[SweepVariation(index=0, label="base", values={})],
            variation_seeds=[42],
            random_seed=42,
            multi_run=mock_multi_run,
        )
        yield mock_plan


@pytest.fixture
def service_type() -> MagicMock:
    """Create a mock ServiceType."""
    return MagicMock()


@pytest.fixture
def cli_config() -> CLIConfig:
    """Construct a default CLIConfig (cyclopts would normally produce this)."""
    return CLIConfig()


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
        mock_loaders: MagicMock,
        service_type: MagicMock,
        cli_config: CLIConfig,
    ) -> None:
        """Test that service_id is forwarded to bootstrap."""
        service(
            service_type=service_type,
            cli_config=cli_config,
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
        mock_loaders: MagicMock,
        service_type: MagicMock,
        cli_config: CLIConfig,
    ) -> None:
        """Test that optional arguments default to None."""
        service(service_type=service_type, cli_config=cli_config)

        call_kwargs = mock_bootstrap.call_args.kwargs
        assert call_kwargs["service_id"] is None

    def test_health_port_sets_environment(
        self,
        mock_bootstrap: MagicMock,
        mock_loaders: MagicMock,
        service_type: MagicMock,
        cli_config: CLIConfig,
    ) -> None:
        """Test that health_port sets Environment.SERVICE health settings."""
        service(service_type=service_type, cli_config=cli_config, health_port=9090)

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_PORT == 9090

    def test_health_host_sets_environment(
        self,
        mock_bootstrap: MagicMock,
        mock_loaders: MagicMock,
        service_type: MagicMock,
        cli_config: CLIConfig,
    ) -> None:
        """Test that health_host sets Environment.SERVICE health settings."""
        service(service_type=service_type, cli_config=cli_config, health_host="0.0.0.0")

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_HOST == "0.0.0.0"

    def test_health_host_and_port_set_environment(
        self,
        mock_bootstrap: MagicMock,
        mock_loaders: MagicMock,
        service_type: MagicMock,
        cli_config: CLIConfig,
    ) -> None:
        """Test that both health_host and health_port set Environment.SERVICE health settings."""
        service(
            service_type=service_type,
            cli_config=cli_config,
            health_host="0.0.0.0",
            health_port=8081,
        )

        assert Environment.SERVICE.HEALTH_ENABLED is True
        assert Environment.SERVICE.HEALTH_HOST == "0.0.0.0"
        assert Environment.SERVICE.HEALTH_PORT == 8081

    def test_none_health_args_do_not_modify_environment(
        self,
        mock_bootstrap: MagicMock,
        mock_loaders: MagicMock,
        service_type: MagicMock,
        cli_config: CLIConfig,
    ) -> None:
        """Test that None health args leave Environment.SERVICE unchanged."""
        original_enabled = Environment.SERVICE.HEALTH_ENABLED
        original_host = Environment.SERVICE.HEALTH_HOST
        original_port = Environment.SERVICE.HEALTH_PORT

        service(
            service_type=service_type,
            cli_config=cli_config,
            health_host=None,
            health_port=None,
        )

        assert original_enabled == Environment.SERVICE.HEALTH_ENABLED
        assert original_host == Environment.SERVICE.HEALTH_HOST
        assert original_port == Environment.SERVICE.HEALTH_PORT

    def test_health_args_not_passed_to_bootstrap(
        self,
        mock_bootstrap: MagicMock,
        mock_loaders: MagicMock,
        service_type: MagicMock,
        cli_config: CLIConfig,
    ) -> None:
        """Test that health args are not forwarded to bootstrap_and_run_service."""
        service(
            service_type=service_type,
            cli_config=cli_config,
            health_host="0.0.0.0",
            health_port=8080,
        )

        call_kwargs = mock_bootstrap.call_args.kwargs
        assert "health_host" not in call_kwargs
        assert "health_port" not in call_kwargs
