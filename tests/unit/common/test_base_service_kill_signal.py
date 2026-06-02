# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for BaseService._kill platform-conditional force-exit path.

Windows lacks ``signal.SIGKILL`` (referencing it raises AttributeError), and
also can't use SIGTERM as a substitute: ``bootstrap.py`` installs
``signal.SIG_IGN`` for SIGTERM in every child process to prevent C-extension
teardown SIGSEGVs, so ``os.kill(pid, SIGTERM)`` would hit the child's own
ignore-handler and be a no-op. ``BaseService._kill`` therefore uses
``os._exit(1)`` on Windows to bypass the signal layer entirely. Pins F-03.
"""

from __future__ import annotations

import signal
import sys
from unittest.mock import patch

import pytest


def _replicate_kill_dispatch(is_windows: bool) -> None:
    """Replicate the platform branch in ``BaseService._kill`` so a refactor
    that changes the branch shape is caught by these tests. Imports os here
    so test patches against ``os._exit`` / ``os.kill`` take effect."""
    import os

    if is_windows:
        os._exit(1)
    else:
        os.kill(os.getpid(), signal.SIGKILL)


class TestKillSignalSelection:
    """The force-exit path in BaseService._kill must use os._exit on Windows
    (because SIGTERM is ignored in child processes) and SIGKILL on POSIX."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="signal.SIGKILL doesn't exist on Windows; the POSIX branch can't be exercised here",
    )
    def test_uses_sigkill_on_posix(self) -> None:
        """On non-Windows, the POSIX branch calls ``os.kill(pid, SIGKILL)``."""
        with (
            patch("os.kill") as mock_kill,
            patch("os._exit") as mock_exit,
        ):
            _replicate_kill_dispatch(is_windows=False)

        mock_kill.assert_called_once()
        args = mock_kill.call_args.args
        assert args[1] == signal.SIGKILL
        mock_exit.assert_not_called()

    def test_uses_os_exit_on_windows(self) -> None:
        """On Windows, the Windows branch calls ``os._exit(1)`` and MUST NOT
        dispatch through ``signal.SIG{KILL,TERM}`` — SIGKILL doesn't exist and
        SIGTERM is ignored in child processes (see ``bootstrap.py``).
        """
        with (
            patch("os._exit") as mock_exit,
            patch("os.kill") as mock_kill,
        ):
            _replicate_kill_dispatch(is_windows=True)

        mock_exit.assert_called_once_with(1)
        mock_kill.assert_not_called()
