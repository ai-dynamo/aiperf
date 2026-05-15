# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for macOS-specific terminal FD redirection in bootstrap.py"""

import os
from unittest.mock import MagicMock, call, patch

import pytest

from aiperf.common.bootstrap import (
    _redirect_stdio_to_devnull,
    bootstrap_and_run_service,
)
from aiperf.config.flags.cli_config import CLIConfig
from tests.unit.conftest import make_run_from_cli


class TestBootstrapMacOSRedirect:
    """Test that _redirect_stdio_to_devnull is called under the right conditions."""

    @pytest.fixture(autouse=True)
    def setup_bootstrap_mocks(
        self,
        mock_psutil_process,
        mock_setup_child_process_logging,
        register_dummy_services,
    ):
        pass

    def test_redirect_called_in_macos_child_process(
        self,
        service_config_no_uvloop: CLIConfig,
        cli_config: CLIConfig,
        mock_log_queue,
        mock_darwin_child_process,
    ):
        """_redirect_stdio_to_devnull is called for Darwin child processes."""
        run = make_run_from_cli(cli_config)
        with patch(
            "aiperf.common.bootstrap._redirect_stdio_to_devnull"
        ) as mock_redirect:
            bootstrap_and_run_service(
                "test_dummy",
                run=run,
                log_queue=mock_log_queue,
                service_id="test_service",
            )
            mock_redirect.assert_called_once()

    def test_redirect_not_called_in_main_process(
        self,
        service_config_no_uvloop: CLIConfig,
        cli_config: CLIConfig,
        mock_log_queue,
        mock_darwin_main_process,
    ):
        """_redirect_stdio_to_devnull is NOT called in the main process."""
        run = make_run_from_cli(cli_config)
        with patch(
            "aiperf.common.bootstrap._redirect_stdio_to_devnull"
        ) as mock_redirect:
            bootstrap_and_run_service(
                "test_dummy",
                run=run,
                log_queue=mock_log_queue,
                service_id="test_service",
            )
            mock_redirect.assert_not_called()

    def test_redirect_not_called_on_linux(
        self,
        service_config_no_uvloop: CLIConfig,
        cli_config: CLIConfig,
        mock_log_queue,
        mock_linux_child_process,
    ):
        """_redirect_stdio_to_devnull is NOT called on Linux."""
        run = make_run_from_cli(cli_config)
        with patch(
            "aiperf.common.bootstrap._redirect_stdio_to_devnull"
        ) as mock_redirect:
            bootstrap_and_run_service(
                "test_dummy",
                run=run,
                log_queue=mock_log_queue,
                service_id="test_service",
            )
            mock_redirect.assert_not_called()


class TestRedirectStdioToDevnull:
    """Tests for _redirect_stdio_to_devnull OS-level FD redirection.

    These tests mock os.dup2/os.open/os.fdopen to avoid corrupting pytest's
    own FD capture mechanism.
    """

    def _patch_os(self):
        return (
            patch("aiperf.common.bootstrap.os.open", return_value=99),
            patch("aiperf.common.bootstrap.os.dup2"),
            patch("aiperf.common.bootstrap.os.close"),
            patch("aiperf.common.bootstrap.os.fdopen"),
        )

    def test_redirect_dup2_called_for_stdin_stdout_stderr(self):
        """os.dup2 redirects FDs 0, 1, 2 to /dev/null."""
        p_open, p_dup2, p_close, p_fdopen = self._patch_os()
        with (
            p_open as mock_open,
            p_dup2 as mock_dup2,
            p_close as mock_close,
            p_fdopen,
            patch("sys.stdin"),
            patch("sys.stdout"),
            patch("sys.stderr"),
        ):
            _redirect_stdio_to_devnull()

            mock_open.assert_called_once_with(os.devnull, os.O_RDWR)
            assert mock_dup2.call_args_list == [
                call(99, 0),
                call(99, 1),
                call(99, 2),
            ]
            mock_close.assert_called_once_with(99)

    def test_redirect_creates_python_streams_from_fds(self):
        """sys.stdin/stdout/stderr are recreated from OS-level FDs."""
        mock_streams = {0: MagicMock(), 1: MagicMock(), 2: MagicMock()}

        def fdopen_side_effect(fd, _mode="r", **_kwargs):
            return mock_streams[fd]

        p_open, p_dup2, p_close, _ = self._patch_os()
        with (
            p_open,
            p_dup2,
            p_close,
            patch(
                "aiperf.common.bootstrap.os.fdopen", side_effect=fdopen_side_effect
            ) as mock_fdopen,
            patch("sys.stdin"),
            patch("sys.stdout"),
            patch("sys.stderr"),
        ):
            import sys

            _redirect_stdio_to_devnull()

            mock_fdopen.assert_any_call(0, "r", closefd=False)
            mock_fdopen.assert_any_call(1, "w", closefd=False)
            mock_fdopen.assert_any_call(2, "w", closefd=False)
            assert sys.stdin is mock_streams[0]
            assert sys.stdout is mock_streams[1]
            assert sys.stderr is mock_streams[2]
