# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for macOS-specific terminal FD closing in bootstrap.py"""

import contextlib
import os
import sys
from unittest.mock import patch

import pytest

from aiperf.common.bootstrap import (
    _redirect_stdio_to_devnull,
    bootstrap_and_run_service,
)
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


def _close_devnull_streams(saved: tuple) -> None:
    """Close devnull streams opened by _redirect_stdio_to_devnull before restoring originals."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if stream is not None and stream not in saved:
            with contextlib.suppress(Exception):
                stream.close()


class TestRedirectStdioToDevnull:
    """Tests for _redirect_stdio_to_devnull handling of None streams.

    On macOS, FD_CLOEXEC is set on terminal FDs before spawning child processes.
    After exec, FDs 0/1/2 are closed, so Python sets sys.stdout/stderr to None.
    The redirect function must handle this without leaving streams as None,
    otherwise libraries like billiard crash on sys.stdout.flush().
    """

    def test_redirect_stdio_to_devnull_none_stdout_replaced_with_devnull(self, capsys):
        """Streams set to None (FD_CLOEXEC + spawn) are replaced with devnull."""
        with capsys.disabled():
            saved = sys.stdin, sys.stdout, sys.stderr
            try:
                sys.stdin = None
                sys.stdout = None
                sys.stderr = None
                _redirect_stdio_to_devnull()

                assert sys.stdout is not None
                sys.stdout.flush()  # must not raise
                sys.stdout.write("discard")  # must not raise
            finally:
                _close_devnull_streams(saved)
                sys.stdin, sys.stdout, sys.stderr = saved

    def test_redirect_stdio_to_devnull_all_none_streams_replaced_with_devnull(
        self, capsys
    ):
        """All three streams None — all must be replaced with valid objects."""
        with capsys.disabled():
            saved = sys.stdin, sys.stdout, sys.stderr
            try:
                sys.stdin = None
                sys.stdout = None
                sys.stderr = None
                _redirect_stdio_to_devnull()

                for stream in (sys.stdin, sys.stdout, sys.stderr):
                    assert stream is not None
                    stream.flush()
            finally:
                _close_devnull_streams(saved)
                sys.stdin, sys.stdout, sys.stderr = saved

    def test_redirect_stdio_to_devnull_valid_streams_redirected_to_devnull(
        self, capsys
    ):
        """Streams are redirected to /dev/null, not arbitrary objects."""
        with capsys.disabled():
            saved = sys.stdin, sys.stdout, sys.stderr
            try:
                sys.stdin = open(os.devnull)  # noqa: SIM115
                sys.stdout = open(os.devnull, "w")  # noqa: SIM115
                sys.stderr = open(os.devnull, "w")  # noqa: SIM115

                _redirect_stdio_to_devnull()

                assert sys.stdout.name == os.devnull
                assert sys.stderr.name == os.devnull
                assert sys.stdin.name == os.devnull
            finally:
                _close_devnull_streams(saved)
                sys.stdin, sys.stdout, sys.stderr = saved
