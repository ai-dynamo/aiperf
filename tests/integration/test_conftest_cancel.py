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
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from aiperf.common.constants import IS_WINDOWS
from tests.harness.subprocess import _new_process_group_kwargs
from tests.harness.utils import AIPerfRunnerFn


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


@pytest.mark.asyncio
@pytest.mark.skipif(IS_WINDOWS, reason="POSIX process-group escalation")
@pytest.mark.parametrize("timeouts", [1, 2, 3])
async def test_runner_timeout_escalates_to_process_group(
    aiperf_runner: AIPerfRunnerFn, timeouts: int
) -> None:
    process = MagicMock()
    process.communicate = AsyncMock(
        side_effect=[TimeoutError() for _ in range(timeouts)] + [(b"", b"")]
    )
    with (
        patch(
            "tests.integration.conftest.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ) as spawn,
        patch("tests.integration.conftest._cancel_aiperf_for_timeout") as cancel,
        patch("tests.integration.conftest._killpg") as killpg,
        pytest.raises(RuntimeError, match="AIPerf timed out after 120s"),
    ):
        await aiperf_runner(["plugins"], 120)

    for key, value in _new_process_group_kwargs().items():
        assert spawn.call_args.kwargs[key] == value
    cancel.assert_called_once_with(process)
    assert killpg.call_args_list == [
        call(process, sig) for sig in [signal.SIGTERM, signal.SIGKILL][: timeouts - 1]
    ]
    assert process.communicate.await_count == min(timeouts + 1, 3)
