# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for macOS-specific terminal corruption fixes in cli_runner.py"""

import multiprocessing
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.plugin.enums import UIType


class TestMacOSTerminalFixes:
    """Test the macOS-specific terminal corruption fixes in cli_runner.py"""

    @pytest.fixture
    def service_config_dashboard(self) -> ServiceConfig:
        """Create a ServiceConfig with Dashboard UI type."""
        config = ServiceConfig()
        config.ui_type = UIType.DASHBOARD
        return config

    @pytest.fixture
    def service_config_simple(self) -> ServiceConfig:
        """Create a ServiceConfig with Simple UI type."""
        config = ServiceConfig()
        config.ui_type = UIType.SIMPLE
        return config

    def test_spawn_method_set_on_macos_dashboard(
        self,
        service_config_dashboard: ServiceConfig,
        user_config: UserConfig,
        mock_platform_darwin: Mock,
        mock_multiprocessing_set_start_method: Mock,
        mock_bootstrap_and_run_service: Mock,
    ):
        """Test that spawn method is set when on macOS with Dashboard UI."""
        from aiperf.cli_runner import run_system_controller

        run_system_controller(user_config, service_config_dashboard)

        # Verify spawn method was set
        mock_multiprocessing_set_start_method.assert_called_once_with(
            "spawn", force=True
        )

    def test_spawn_method_not_set_on_linux(
        self,
        service_config_dashboard: ServiceConfig,
        user_config: UserConfig,
        mock_platform_linux: Mock,
        mock_multiprocessing_set_start_method: Mock,
        mock_bootstrap_and_run_service: Mock,
    ):
        """Test that spawn method is NOT set on Linux."""
        from aiperf.cli_runner import run_system_controller

        run_system_controller(user_config, service_config_dashboard)

        # Verify spawn method was NOT called on Linux
        mock_multiprocessing_set_start_method.assert_not_called()

    def test_spawn_method_not_set_for_simple_ui(
        self,
        service_config_simple: ServiceConfig,
        user_config: UserConfig,
        mock_platform_darwin: Mock,
        mock_multiprocessing_set_start_method: Mock,
        mock_bootstrap_and_run_service: Mock,
    ):
        """Test that spawn method is NOT set when not using Dashboard UI on macOS."""
        from aiperf.cli_runner import run_system_controller

        run_system_controller(user_config, service_config_simple)

        # Verify spawn method was NOT called for non-dashboard UI
        mock_multiprocessing_set_start_method.assert_not_called()

    @patch("fcntl.fcntl")
    def test_fd_cloexec_not_set_on_linux(
        self,
        mock_fcntl: Mock,
        service_config_dashboard: ServiceConfig,
        user_config: UserConfig,
        mock_platform_linux: Mock,
        mock_bootstrap_and_run_service: Mock,
        mock_get_global_log_queue: Mock,
    ):
        """Test that FD_CLOEXEC is NOT set on Linux."""
        from aiperf.cli_runner import run_system_controller

        mock_get_global_log_queue.return_value = MagicMock(spec=multiprocessing.Queue)

        run_system_controller(user_config, service_config_dashboard)

        # fcntl should not be called on Linux
        mock_fcntl.assert_not_called()

    def test_runtime_error_in_set_start_method_is_handled(
        self,
        service_config_dashboard: ServiceConfig,
        user_config: UserConfig,
        mock_platform_darwin: Mock,
        mock_multiprocessing_set_start_method: Mock,
        mock_bootstrap_and_run_service: Mock,
    ):
        """Test that RuntimeError when setting start method is gracefully handled."""
        from aiperf.cli_runner import run_system_controller

        mock_multiprocessing_set_start_method.side_effect = RuntimeError(
            "context already set"
        )

        # Should not raise an exception
        run_system_controller(user_config, service_config_dashboard)

        # Verify it tried to set the method
        mock_multiprocessing_set_start_method.assert_called_once()

    def test_log_queue_created_before_ui_on_dashboard(
        self,
        service_config_dashboard: ServiceConfig,
        user_config: UserConfig,
        mock_platform_darwin: Mock,
        mock_bootstrap_and_run_service: Mock,
        mock_get_global_log_queue: Mock,
    ):
        """Test that log_queue is created early when using Dashboard UI."""
        from aiperf.cli_runner import run_system_controller

        mock_queue = MagicMock(spec=multiprocessing.Queue)
        mock_get_global_log_queue.return_value = mock_queue

        run_system_controller(user_config, service_config_dashboard)

        # Verify log queue was created
        mock_get_global_log_queue.assert_called_once()

        # Verify it was passed to bootstrap_and_run_service
        mock_bootstrap_and_run_service.assert_called_once()
        call_kwargs = mock_bootstrap_and_run_service.call_args.kwargs
        assert "log_queue" in call_kwargs
        assert call_kwargs["log_queue"] == mock_queue
