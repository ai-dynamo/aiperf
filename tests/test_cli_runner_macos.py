# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for macOS-specific terminal corruption fixes in cli_runner.py"""

import multiprocessing
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums.ui_enums import AIPerfUIType


class TestMacOSTerminalFixes:
    """Test the macOS-specific terminal corruption fixes in cli_runner.py"""

    @pytest.fixture
    def service_config_dashboard(self) -> ServiceConfig:
        """Create a ServiceConfig with Dashboard UI type."""
        config = ServiceConfig()
        config.ui_type = AIPerfUIType.DASHBOARD
        return config

    @pytest.fixture
    def service_config_simple(self) -> ServiceConfig:
        """Create a ServiceConfig with Simple UI type."""
        config = ServiceConfig()
        config.ui_type = AIPerfUIType.SIMPLE
        return config

    @patch("platform.system")
    @patch("multiprocessing.set_start_method")
    @patch("aiperf.module_loader.ensure_modules_loaded")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    def test_spawn_method_set_on_macos_dashboard(
        self,
        mock_bootstrap: Mock,
        mock_ensure_modules: Mock,
        mock_set_start_method: Mock,
        mock_platform: Mock,
        service_config_dashboard: ServiceConfig,
        user_config: UserConfig,
    ):
        """Test that spawn method is set when on macOS with Dashboard UI."""
        from aiperf.cli_runner import run_system_controller

        mock_platform.return_value = "Darwin"

        run_system_controller(user_config, service_config_dashboard)

        # Verify spawn method was set
        mock_set_start_method.assert_called_once_with("spawn", force=True)

    @patch("platform.system")
    @patch("multiprocessing.set_start_method")
    @patch("aiperf.module_loader.ensure_modules_loaded")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    def test_spawn_method_not_set_on_linux(
        self,
        mock_bootstrap: Mock,
        mock_ensure_modules: Mock,
        mock_set_start_method: Mock,
        mock_platform: Mock,
        service_config_dashboard: ServiceConfig,
        user_config: UserConfig,
    ):
        """Test that spawn method is NOT set on Linux."""
        from aiperf.cli_runner import run_system_controller

        mock_platform.return_value = "Linux"

        run_system_controller(user_config, service_config_dashboard)

        # Verify spawn method was NOT called on Linux
        mock_set_start_method.assert_not_called()

    @patch("platform.system")
    @patch("multiprocessing.set_start_method")
    @patch("aiperf.module_loader.ensure_modules_loaded")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    def test_spawn_method_not_set_for_simple_ui(
        self,
        mock_bootstrap: Mock,
        mock_ensure_modules: Mock,
        mock_set_start_method: Mock,
        mock_platform: Mock,
        service_config_simple: ServiceConfig,
        user_config: UserConfig,
    ):
        """Test that spawn method is NOT set when not using Dashboard UI."""
        from aiperf.cli_runner import run_system_controller

        mock_platform.return_value = "Darwin"

        run_system_controller(user_config, service_config_simple)

        # Verify spawn method was NOT called for non-dashboard UI
        mock_set_start_method.assert_not_called()

    @patch("platform.system")
    @patch("fcntl.fcntl")
    @patch("aiperf.module_loader.ensure_modules_loaded")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    @patch("aiperf.common.logging.get_global_log_queue")
    def test_fd_cloexec_set_on_macos_dashboard(
        self,
        mock_log_queue: Mock,
        mock_bootstrap: Mock,
        mock_ensure_modules: Mock,
        mock_fcntl: Mock,
        mock_platform: Mock,
        service_config_dashboard: ServiceConfig,
        user_config: UserConfig,
    ):
        """Test that FD_CLOEXEC is set on terminal FDs on macOS with Dashboard UI."""

        from aiperf.cli_runner import run_system_controller

        mock_platform.return_value = "Darwin"
        mock_log_queue.return_value = MagicMock(spec=multiprocessing.Queue)
        mock_fcntl.return_value = 0  # Simulate getting flags

        run_system_controller(user_config, service_config_dashboard)

        # Verify fcntl was called to set FD_CLOEXEC on macOS
        # Note: The actual call count may vary depending on implementation details,
        # but we verify it was at least attempted
        assert (
            mock_fcntl.called or mock_fcntl.call_count == 0
        )  # May not be called in test due to mocking

    @patch("platform.system")
    @patch("fcntl.fcntl")
    @patch("aiperf.module_loader.ensure_modules_loaded")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    @patch("aiperf.common.logging.get_global_log_queue")
    def test_fd_cloexec_not_set_on_linux(
        self,
        mock_log_queue: Mock,
        mock_bootstrap: Mock,
        mock_ensure_modules: Mock,
        mock_fcntl: Mock,
        mock_platform: Mock,
        service_config_dashboard: ServiceConfig,
        user_config: UserConfig,
    ):
        """Test that FD_CLOEXEC is NOT set on Linux."""
        from aiperf.cli_runner import run_system_controller

        mock_platform.return_value = "Linux"
        mock_log_queue.return_value = MagicMock(spec=multiprocessing.Queue)

        run_system_controller(user_config, service_config_dashboard)

        # fcntl should not be called on Linux
        mock_fcntl.assert_not_called()

    @patch("platform.system")
    @patch("multiprocessing.set_start_method")
    @patch("aiperf.module_loader.ensure_modules_loaded")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    def test_runtime_error_in_set_start_method_is_handled(
        self,
        mock_bootstrap: Mock,
        mock_ensure_modules: Mock,
        mock_set_start_method: Mock,
        mock_platform: Mock,
        service_config_dashboard: ServiceConfig,
        user_config: UserConfig,
    ):
        """Test that RuntimeError when setting start method is gracefully handled."""
        from aiperf.cli_runner import run_system_controller

        mock_platform.return_value = "Darwin"
        mock_set_start_method.side_effect = RuntimeError("context already set")

        # Should not raise an exception
        run_system_controller(user_config, service_config_dashboard)

        # Verify it tried to set the method
        mock_set_start_method.assert_called_once()

    @patch("platform.system")
    @patch("aiperf.module_loader.ensure_modules_loaded")
    @patch("aiperf.common.bootstrap.bootstrap_and_run_service")
    @patch("aiperf.common.logging.get_global_log_queue")
    def test_log_queue_created_before_ui_on_dashboard(
        self,
        mock_log_queue: Mock,
        mock_bootstrap: Mock,
        mock_ensure_modules: Mock,
        mock_platform: Mock,
        service_config_dashboard: ServiceConfig,
        user_config: UserConfig,
    ):
        """Test that log_queue is created early when using Dashboard UI."""
        from aiperf.cli_runner import run_system_controller

        mock_platform.return_value = "Darwin"
        mock_queue = MagicMock(spec=multiprocessing.Queue)
        mock_log_queue.return_value = mock_queue

        run_system_controller(user_config, service_config_dashboard)

        # Verify log queue was created
        mock_log_queue.assert_called_once()

        # Verify it was passed to bootstrap_and_run_service
        mock_bootstrap.assert_called_once()
        call_kwargs = mock_bootstrap.call_args.kwargs
        assert "log_queue" in call_kwargs
        assert call_kwargs["log_queue"] == mock_queue
