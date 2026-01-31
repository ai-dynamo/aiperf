# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for macOS-specific terminal FD closing in bootstrap.py"""

from unittest.mock import patch

import pytest

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig, UserConfig


class TestBootstrapMacOSFixes:
    """Test the macOS-specific terminal FD closing in bootstrap.py"""

    @pytest.fixture(autouse=True)
    def setup_bootstrap_mocks(
        self,
        mock_psutil_process,
        mock_setup_child_process_logging,
        register_dummy_services,
    ):
        """Combine common bootstrap mocks that are used but not called in tests."""
        pass

    @pytest.mark.parametrize("capsys", [None], indirect=True)
    def test_terminal_fds_closed_in_macos_child_process(
        self,
        capsys,
        service_config_no_uvloop: ServiceConfig,
        user_config: UserConfig,
        mock_log_queue,
        mock_darwin_child_process,
    ):
        """Test that terminal FDs are closed in child processes on macOS."""
        # Disable pytest capture to avoid conflicts with FD mocking
        # Use the disabled context around the patches to ensure proper cleanup
        with (
            capsys.disabled(),
            patch("sys.stdin") as mock_stdin,
            patch("sys.stdout") as mock_stdout,
            patch("sys.stderr") as mock_stderr,
        ):
            # Setup FD mocks
            mock_stdin.fileno.return_value = 0
            mock_stdout.fileno.return_value = 1
            mock_stderr.fileno.return_value = 2

            bootstrap_and_run_service(
                "test_dummy",
                service_config=service_config_no_uvloop,
                user_config=user_config,
                log_queue=mock_log_queue,
                service_id="test_service",
            )

            # Verify FDs were closed
            mock_stdin.close.assert_called()
            mock_stdout.close.assert_called()
            mock_stderr.close.assert_called()

    def test_terminal_fds_not_closed_in_main_process(
        self,
        service_config_no_uvloop: ServiceConfig,
        user_config: UserConfig,
        mock_log_queue,
        mock_darwin_main_process,
    ):
        """Test that terminal FDs are NOT closed in the main process on macOS."""
        with patch("sys.stdin") as mock_stdin:
            bootstrap_and_run_service(
                "test_dummy",
                service_config=service_config_no_uvloop,
                user_config=user_config,
                log_queue=mock_log_queue,
                service_id="test_service",
            )

            # Verify stdin was NOT closed in main process
            mock_stdin.close.assert_not_called()

    def test_terminal_fds_not_closed_on_linux(
        self,
        service_config_no_uvloop: ServiceConfig,
        user_config: UserConfig,
        mock_log_queue,
        mock_linux_child_process,
    ):
        """Test that terminal FDs are NOT closed on Linux."""
        with patch("sys.stdin") as mock_stdin:
            bootstrap_and_run_service(
                "test_dummy",
                service_config=service_config_no_uvloop,
                user_config=user_config,
                log_queue=mock_log_queue,
                service_id="test_service",
            )

            # Verify stdin was NOT closed on Linux
            mock_stdin.close.assert_not_called()

    @pytest.mark.parametrize("capsys", [None], indirect=True)
    def test_terminal_fd_closing_handles_exceptions(
        self,
        capsys,
        service_config_no_uvloop: ServiceConfig,
        user_config: UserConfig,
        mock_log_queue,
        mock_darwin_child_process,
    ):
        """Test that exceptions during FD closing are handled gracefully."""
        # Disable pytest capture to avoid conflicts with FD mocking
        # Use the disabled context around the patches to ensure proper cleanup
        with (
            capsys.disabled(),
            patch("sys.stdin") as mock_stdin,
            patch("sys.stdout") as mock_stdout,
            patch("sys.stderr") as mock_stderr,
        ):
            # Setup FD mocks
            mock_stdin.fileno.return_value = 0
            mock_stdout.fileno.return_value = 1
            mock_stderr.fileno.return_value = 2

            # Make stdin.close() raise an exception
            mock_stdin.close.side_effect = OSError("File descriptor already closed")

            # Should not raise an exception despite the error
            try:
                bootstrap_and_run_service(
                    "test_dummy",
                    service_config=service_config_no_uvloop,
                    user_config=user_config,
                    log_queue=mock_log_queue,
                    service_id="test_service",
                )
            except OSError:
                pytest.fail("Exception should have been caught and handled")
