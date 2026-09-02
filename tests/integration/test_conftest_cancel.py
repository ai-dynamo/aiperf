# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for Bug 7: subprocess.send_signal(SIGINT) is unsupported on
Windows.

The integration-test harness times out long-running AIPerf subprocess runs by
sending SIGINT (graceful) on Unix, or terminate() (hard kill) on Windows. The
asyncio Popen wrapper on Windows only accepts SIGTERM, CTRL_C_EVENT, and
CTRL_BREAK_EVENT — passing SIGINT raises ``ValueError: Unsupported signal: 2``.
"""

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch


def test_cancel_calls_send_signal_sigint_on_unix() -> None:
    from tests.integration.conftest import _cancel_aiperf_for_timeout

    process = MagicMock()
    with patch("tests.integration.conftest.sys.platform", "linux"):
        _cancel_aiperf_for_timeout(process)

    process.send_signal.assert_called_once_with(signal.SIGINT)
    process.terminate.assert_not_called()


def test_cancel_calls_terminate_on_windows() -> None:
    from tests.integration.conftest import _cancel_aiperf_for_timeout

    process = MagicMock()
    with patch("tests.integration.conftest.sys.platform", "win32"):
        _cancel_aiperf_for_timeout(process)

    process.terminate.assert_called_once_with()
    process.send_signal.assert_not_called()
