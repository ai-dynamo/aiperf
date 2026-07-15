# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes subprocess wrappers.

Focuses on:
- Real subprocess stdout/stderr propagation across success and failure exits
- Timeout cleanup contracts for command probes that would otherwise hang kopf handlers
- Cancellation propagation without converting caller cancellation into success/failure results
- Streaming process termination, including SIGTERM-resistant children
- Shell-injection resistance from argv boundaries rather than shell parsing

Out of scope (covered elsewhere):
- Basic dataclass and mocked wrapper plumbing: tests/unit/kubernetes/test_subproc.py
- Kubernetes command call sites that consume these wrappers: sibling kube CLI/operator tests
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from pytest import param

from aiperf.kubernetes.subproc import (
    check_command,
    run_command,
    start_streaming_process,
    terminate_process,
)

# ============================================================================
# Helpers
# ============================================================================


def _python_cmd(script: str, *args: str) -> list[str]:
    """Build an isolated Python subprocess command with realistic argv values."""
    return [sys.executable, "-c", script, *args]


async def _read_stream_line(stream: asyncio.StreamReader | None) -> str:
    """Read one line from a subprocess stream and decode it for assertions."""
    assert stream is not None
    return (await stream.readline()).decode(errors="replace")


# ============================================================================
# run_command output and exit contracts
# ============================================================================


class TestRunCommandAdversarialOutput:
    """Verify captured output is never dropped just because the exit was unusual."""

    @pytest.mark.asyncio
    async def test_run_command_nonzero_exit_preserves_stdout_stderr_and_returncode(
        self,
    ) -> None:
        result = await run_command(
            _python_cmd(
                "import sys; "
                "sys.stdout.write('created aiperf-bench-7f2a\\n'); "
                "sys.stderr.write('kubectl rejected namespace perf-lab\\n'); "
                "raise SystemExit(42)"
            ),
            timeout=5.0,
        )

        assert result.ok is False
        assert result.returncode == 42
        assert result.stdout == "created aiperf-bench-7f2a\n"
        assert result.stderr == "kubectl rejected namespace perf-lab\n"

    @pytest.mark.asyncio
    async def test_run_command_invalid_utf8_replaces_bytes_in_both_streams(
        self,
    ) -> None:
        result = await run_command(
            _python_cmd(
                "import sys; "
                "sys.stdout.buffer.write(b'pod=aiperf-bench-7f2a\\xff\\n'); "
                "sys.stderr.buffer.write(b'phase=Pending\\xfe\\n')"
            ),
            timeout=5.0,
        )

        assert result.ok is True
        assert result.stdout == "pod=aiperf-bench-7f2a�\n"
        assert result.stderr == "phase=Pending�\n"

    @pytest.mark.asyncio
    async def test_run_command_shell_metacharacters_stay_inside_single_argument(
        self, tmp_path: Path
    ) -> None:
        sentinel = tmp_path / "shell-injection-created"
        payload = f"aiperf-bench-7f2a; touch {sentinel}"

        result = await run_command(
            _python_cmd(
                "import sys; sys.stdout.write(sys.argv[1]); sys.stdout.write('\\n')",
                payload,
            ),
            timeout=5.0,
        )

        assert result.ok is True
        assert result.stdout == f"{payload}\n"
        assert sentinel.exists() is False


# ============================================================================
# Timeout and cancellation contracts
# ============================================================================


class TestRunCommandAdversarialControlFlow:
    """Timeouts and cancellations must not masquerade as ordinary command exits."""

    @pytest.mark.asyncio
    async def test_run_command_timeout_terminates_child_and_raises_timeout(
        self,
    ) -> None:
        with pytest.raises(TimeoutError):
            await run_command(
                _python_cmd("import time; time.sleep(30)"),
                timeout=0.01,
            )

    @pytest.mark.asyncio
    async def test_check_command_timeout_returns_false_instead_of_raising(self) -> None:
        assert (
            await check_command(
                _python_cmd("import time; time.sleep(30)"),
                timeout=0.01,
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_check_command_nonzero_exit_returns_false_with_stderr_present(
        self,
    ) -> None:
        assert (
            await check_command(
                _python_cmd(
                    "import sys; "
                    "sys.stderr.write('helm template failed for aiperf-operator\\n'); "
                    "raise SystemExit(9)"
                ),
                timeout=5.0,
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_run_command_task_cancellation_propagates_cancelled_error(
        self,
    ) -> None:
        task = asyncio.create_task(
            run_command(
                _python_cmd("import time; time.sleep(30)"),
                timeout=60.0,
            )
        )
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


# ============================================================================
# Streaming process contracts
# ============================================================================


class TestStreamingProcessAdversarialTermination:
    """Verify long-running stream helpers expose output and can be cleaned up."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "merge_stderr,expected_stdout,stderr_is_merged",
        [
            (False, "stream-ready\n", False),
            param(True, "stream-ready\nwarning-from-sidecar\n", True, id="stderr-merged"),
        ],
    )  # fmt: skip
    async def test_start_streaming_process_stderr_merge_controls_visible_streams(
        self,
        merge_stderr: bool,
        expected_stdout: str,
        stderr_is_merged: bool,
    ) -> None:
        proc = await start_streaming_process(
            _python_cmd(
                "import sys, time; "
                "sys.stdout.write('stream-ready\\n'); sys.stdout.flush(); "
                "sys.stderr.write('warning-from-sidecar\\n'); sys.stderr.flush(); "
                "time.sleep(30)"
            ),
            merge_stderr=merge_stderr,
        )
        try:
            stdout_line = await _read_stream_line(proc.stdout)
            if merge_stderr:
                stdout_line += await _read_stream_line(proc.stdout)
        finally:
            await terminate_process(proc, timeout=1.0)

        assert stdout_line == expected_stdout
        assert (proc.stderr is None) is stderr_is_merged

    @pytest.mark.asyncio
    async def test_terminate_process_sigterm_resistant_child_kills_process(
        self,
    ) -> None:
        proc = await start_streaming_process(
            _python_cmd(
                "import signal, sys, time; "
                "signal.signal(signal.SIGTERM, lambda *_: None); "
                "sys.stdout.write('ignoring-sigterm\\n'); sys.stdout.flush(); "
                "time.sleep(30)"
            )
        )
        assert await _read_stream_line(proc.stdout) == "ignoring-sigterm\n"

        await terminate_process(proc, timeout=0.01)

        assert proc.returncode is not None

    @pytest.mark.asyncio
    async def test_terminate_process_already_exited_stream_keeps_returncode(
        self,
    ) -> None:
        proc = await start_streaming_process(
            _python_cmd("import sys; sys.stdout.write('done\\n')")
        )
        await proc.wait()
        returncode = proc.returncode

        await terminate_process(proc, timeout=0.01)

        assert returncode == 0
        assert proc.returncode == 0
