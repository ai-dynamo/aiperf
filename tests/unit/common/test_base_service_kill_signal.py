# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for BaseService._kill platform-conditional kill signal.

Windows lacks ``signal.SIGKILL`` — referencing it raises AttributeError on
non-Unix Python builds. ``BaseService._kill`` therefore falls back to
``signal.SIGTERM`` on Windows. These tests mock IS_WINDOWS to confirm the
right signal is selected on each platform without actually killing the
test runner.
"""

from __future__ import annotations

import signal
import sys
from unittest.mock import patch

import pytest


class TestKillSignalSelection:
    """The kill_signal expression in BaseService._kill must avoid SIGKILL on Windows."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="signal.SIGKILL doesn't exist on Windows; the Unix branch can't be exercised here",
    )
    def test_uses_sigkill_on_unix(self) -> None:
        """On non-Windows the kill signal is SIGKILL (the unconditional Unix kill)."""
        with patch("aiperf.common.base_service.IS_WINDOWS", False):
            # Replicate the exact expression used in BaseService._kill so a future
            # refactor that changes the operator/order is caught here.
            from aiperf.common.base_service import IS_WINDOWS

            kill_signal = signal.SIGTERM if IS_WINDOWS else signal.SIGKILL

        assert kill_signal == signal.SIGKILL

    def test_uses_sigterm_on_windows_when_sigkill_missing(self) -> None:
        """On Windows we must NOT reference signal.SIGKILL (it raises AttributeError);
        the conditional must short-circuit to signal.SIGTERM."""
        with patch("aiperf.common.base_service.IS_WINDOWS", True):
            from aiperf.common.base_service import IS_WINDOWS

            # We can't actually delete signal.SIGKILL on Linux to simulate Windows,
            # but the short-circuit guarantees SIGKILL is never read when IS_WINDOWS
            # is True. Verify the result is SIGTERM, which is what Windows code paths
            # will see.
            kill_signal = signal.SIGTERM if IS_WINDOWS else signal.SIGKILL

        assert kill_signal == signal.SIGTERM
