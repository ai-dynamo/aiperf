# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for macOS-specific terminal FD redirection in bootstrap.py"""

import os
from unittest.mock import MagicMock, call, patch

import pytest

from aiperf.common.bootstrap import (
    _redirect_stdio_to_devnull,
    bootstrap_and_run_service,
    sweep_stale_child_stderr_logs,
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
        # Distinct FDs for the two os.open() calls (devnull -> 99, stderr
        # file -> 100). Using a single value would let a bug where stderr
        # is duped from the devnull FD pass silently.
        return (
            patch("aiperf.common.bootstrap.os.open", side_effect=[99, 100]),
            patch("aiperf.common.bootstrap.os.dup2"),
            patch("aiperf.common.bootstrap.os.close"),
            patch("aiperf.common.bootstrap.os.fdopen"),
        )

    def test_redirect_dup2_called_for_stdin_stdout_stderr(self):
        """os.dup2 redirects FDs 0, 1 to /dev/null and FD 2 to a per-PID stderr file."""
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

            # Two opens: /dev/null for stdin/stdout, a per-PID file for stderr.
            # Stderr is preserved to a tmp file so uncaught Python tracebacks
            # from spawned children are recoverable for postmortem.
            assert mock_open.call_count == 2
            assert mock_open.call_args_list[0] == call(os.devnull, os.O_RDWR)
            stderr_open_args = mock_open.call_args_list[1].args
            assert "aiperf_child_" in stderr_open_args[0]
            assert stderr_open_args[0].endswith("_stderr.log")

            # stdin/stdout dup from devnull FD (99); stderr dups from the
            # distinct per-PID stderr file FD (100) — proves they're routed
            # to different sources, not aliased.
            assert mock_dup2.call_args_list == [
                call(99, 0),
                call(99, 1),
                call(100, 2),
            ]
            assert mock_close.call_count == 2

    def test_redirect_creates_python_streams_from_fds(self):
        """sys.stdin/stdout/stderr are recreated from OS-level FDs with UTF-8 encoding."""
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

            mock_fdopen.assert_any_call(
                0, "r", encoding="utf-8", errors="replace", closefd=False
            )
            mock_fdopen.assert_any_call(
                1, "w", encoding="utf-8", errors="replace", closefd=False
            )
            mock_fdopen.assert_any_call(
                2, "w", encoding="utf-8", errors="replace", closefd=False
            )
            assert sys.stdin is mock_streams[0]
            assert sys.stdout is mock_streams[1]
            assert sys.stderr is mock_streams[2]


class TestRemoveIfEmpty:
    """Tests for _remove_if_empty — atexit handler that cleans empty stderr files."""

    def test_removes_zero_byte_file(self, tmp_path):
        """An empty stderr file (clean exit, no traceback) is unlinked."""
        from aiperf.common.bootstrap import _remove_if_empty

        empty_file = tmp_path / "aiperf_child_12345_abcd_stderr.log"
        empty_file.write_bytes(b"")

        _remove_if_empty(str(empty_file))

        assert not empty_file.exists(), "Empty stderr file should be removed"

    def test_preserves_nonempty_file(self, tmp_path):
        """A non-empty stderr file (real crash with traceback) is preserved."""
        from aiperf.common.bootstrap import _remove_if_empty

        crash_file = tmp_path / "aiperf_child_12345_abcd_stderr.log"
        crash_file.write_text("Traceback (most recent call last):\n  ...\n")
        size_before = crash_file.stat().st_size

        _remove_if_empty(str(crash_file))

        assert crash_file.exists(), "Non-empty crash log must be preserved"
        assert crash_file.stat().st_size == size_before

    def test_swallows_oserror_for_missing_file(self, tmp_path):
        """Missing file (already cleaned up by something else) doesn't raise."""
        from aiperf.common.bootstrap import _remove_if_empty

        # Should not raise even though the path doesn't exist.
        _remove_if_empty(str(tmp_path / "never_existed.log"))


class TestSweepStaleChildStderrLogs:
    """Pins F-05: ``sweep_stale_child_stderr_logs`` removes zero-byte
    ``aiperf_child_*_stderr.log`` files older than the cutoff and preserves
    non-empty crash logs.
    """

    def test_removes_old_empty_file(self, tmp_path, monkeypatch):
        """Zero-byte file older than the cutoff is unlinked."""
        import time

        stale = tmp_path / "aiperf_child_111_aaaa_stderr.log"
        stale.touch()  # zero bytes
        # Backdate mtime by 48h.
        old_mtime = time.time() - 2 * 86400
        os.utime(stale, (old_mtime, old_mtime))

        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        sweep_stale_child_stderr_logs()

        assert not stale.exists(), "Old empty log should have been deleted"

    def test_preserves_recent_empty_file(self, tmp_path, monkeypatch):
        """Zero-byte file younger than the cutoff is NOT deleted — could be
        an in-flight child whose atexit hasn't fired yet."""
        recent = tmp_path / "aiperf_child_222_bbbb_stderr.log"
        recent.touch()  # zero bytes, fresh mtime
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

        sweep_stale_child_stderr_logs()

        assert recent.exists(), "Recent empty log must not be deleted"

    def test_preserves_old_non_empty_file(self, tmp_path, monkeypatch):
        """Non-empty file (real crash log) is preserved regardless of age —
        the user is supposed to be able to inspect tracebacks postmortem."""
        import time

        crash = tmp_path / "aiperf_child_333_cccc_stderr.log"
        crash.write_text("Traceback (most recent call last):\n  ...\n")
        # Backdate to ensure age-only-difference test isolation.
        old_mtime = time.time() - 2 * 86400
        os.utime(crash, (old_mtime, old_mtime))
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

        sweep_stale_child_stderr_logs()

        assert crash.exists(), "Non-empty crash log must be preserved"

    def test_ignores_unrelated_files(self, tmp_path, monkeypatch):
        """The sweep only touches files matching ``aiperf_child_*_stderr.log``;
        other tempfiles in the same directory must not be deleted."""
        import time

        unrelated = tmp_path / "some_other_file.log"
        unrelated.touch()
        old_mtime = time.time() - 2 * 86400
        os.utime(unrelated, (old_mtime, old_mtime))
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

        sweep_stale_child_stderr_logs()

        assert unrelated.exists(), "Sweep must not touch non-aiperf files"
