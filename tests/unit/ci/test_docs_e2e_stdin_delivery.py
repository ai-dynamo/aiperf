# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A broken pipe fails its own command, not the whole suite.

The command is written to ``docker exec`` over stdin inside the watchdog's
scope, so a timeout kill can break the pipe mid-write. Letting that
``BrokenPipeError`` propagate would leave ``run_tests`` altogether and silently
skip every server after this one -- the worst outcome for a suite whose job is
to report which guides work.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests/ci/test_docs_end_to_end"))

from test_runner import deliver_command_over_stdin  # noqa: E402


class _Stdin:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.written: list[str] = []
        self.closed = False

    def write(self, text: str) -> None:
        if self._error is not None:
            raise self._error
        self.written.append(text)

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self, error: Exception | None = None) -> None:
        self.stdin = _Stdin(error)
        self.killed = False

    def kill(self) -> None:
        self.killed = True


def test_a_delivered_command_reports_success() -> None:
    process = _Process()

    assert deliver_command_over_stdin(process, "aiperf profile", "test 1") is True
    assert process.stdin.written == ["aiperf profile\n"]
    assert process.stdin.closed
    assert not process.killed


def test_a_broken_pipe_is_reported_not_raised() -> None:
    process = _Process(BrokenPipeError(32, "Broken pipe"))

    assert deliver_command_over_stdin(process, "aiperf profile", "test 1") is False
    assert process.killed, "the process must be reaped, not left holding the container"
