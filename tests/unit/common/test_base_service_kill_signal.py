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

from aiperf.common.base_service import _force_exit_process


class TestForceExitProcess:
    """The force-exit helper in ``BaseService._kill`` must use ``os._exit``
    on Windows (because SIGTERM is ignored in child processes by
    ``bootstrap.py``) and ``SIGKILL`` on POSIX. Tests call the real
    production helper directly with patched ``os._exit`` / ``os.kill`` so a
    refactor that breaks the branch is caught immediately, instead of
    diverging from a test-side replica.
    """

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="signal.SIGKILL doesn't exist on Windows; the POSIX branch can't be exercised here",
    )
    def test_force_exit_posix_uses_sigkill(self) -> None:
        """On non-Windows, the POSIX branch calls ``os.kill(pid, SIGKILL)``."""
        with (
            patch("aiperf.common.base_service.os.kill") as mock_kill,
            patch("aiperf.common.base_service.os._exit") as mock_exit,
        ):
            _force_exit_process(is_windows=False)

        mock_kill.assert_called_once()
        args = mock_kill.call_args.args
        assert args[1] == signal.SIGKILL
        mock_exit.assert_not_called()

    def test_force_exit_windows_uses_os_exit(self) -> None:
        """On Windows, the Windows branch calls ``os._exit(1)`` and MUST NOT
        dispatch through ``signal.SIG{KILL,TERM}`` — SIGKILL doesn't exist and
        SIGTERM is ignored in child processes (see ``bootstrap.py``).
        """
        with (
            patch("aiperf.common.base_service.os._exit") as mock_exit,
            patch("aiperf.common.base_service.os.kill") as mock_kill,
        ):
            _force_exit_process(is_windows=True)

        mock_exit.assert_called_once_with(1)
        mock_kill.assert_not_called()
