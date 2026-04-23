# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.kubernetes.subproc module.

Focuses on:
- CommandResult dataclass and its ok property
- run_command: captures stdout/stderr/returncode from subprocess
- check_command: boolean wrapper around run_command
- start_streaming_process: subprocess creation with optional stderr merging
- terminate_process: graceful termination with kill fallback
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.kubernetes.subproc import (
    CommandResult,
    check_command,
    run_command,
    start_streaming_process,
    terminate_process,
)

# ============================================================
# CommandResult
# ============================================================


class TestCommandResult:
    """Verify CommandResult dataclass behavior."""

    @pytest.mark.parametrize(
        "returncode,expected_ok",
        [
            (0, True),
            (1, False),
            (-1, False),
            (127, False),
            (255, False),
        ],
    )  # fmt: skip
    def test_ok_reflects_zero_returncode(
        self, returncode: int, expected_ok: bool
    ) -> None:
        result = CommandResult(returncode=returncode, stdout="", stderr="")
        assert result.ok is expected_ok

    def test_fields_are_accessible(self) -> None:
        result = CommandResult(returncode=0, stdout="out", stderr="err")
        assert result.returncode == 0
        assert result.stdout == "out"
        assert result.stderr == "err"

    def test_is_frozen(self) -> None:
        result = CommandResult(returncode=0, stdout="", stderr="")
        with pytest.raises(AttributeError):
            result.returncode = 1  # type: ignore[misc]


# ============================================================
# Helpers
# ============================================================


def _make_mock_process(
    returncode: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> AsyncMock:
    """Build a mock asyncio.subprocess.Process."""
    proc = AsyncMock(spec=asyncio.subprocess.Process)
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.pid = 12345
    proc.wait = AsyncMock()
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    return proc


# ============================================================
# run_command
# ============================================================


class TestRunCommand:
    """Verify run_command captures subprocess output correctly."""

    @pytest.mark.asyncio
    async def test_run_command_success_captures_output(self) -> None:
        proc = _make_mock_process(returncode=0, stdout=b"hello\n", stderr=b"")
        with patch(
            "aiperf.kubernetes.subproc.asyncio.create_subprocess_exec",
            return_value=proc,
        ):
            result = await run_command(["echo", "hello"])

        assert result.ok is True
        assert result.stdout == "hello\n"
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_run_command_failure_captures_stderr(self) -> None:
        proc = _make_mock_process(returncode=1, stdout=b"", stderr=b"not found\n")
        with patch(
            "aiperf.kubernetes.subproc.asyncio.create_subprocess_exec",
            return_value=proc,
        ):
            result = await run_command(["ls", "/nonexistent"])

        assert result.ok is False
        assert result.returncode == 1
        assert result.stderr == "not found\n"

    @pytest.mark.asyncio
    async def test_run_command_passes_correct_args(self) -> None:
        proc = _make_mock_process()
        cmd = ["kubectl", "get", "pods", "-n", "default"]
        with patch(
            "aiperf.kubernetes.subproc.asyncio.create_subprocess_exec",
            return_value=proc,
        ) as mock_exec:
            await run_command(cmd)

        mock_exec.assert_awaited_once_with(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    @pytest.mark.asyncio
    async def test_run_command_none_returncode_raises(self) -> None:
        proc = _make_mock_process()
        proc.returncode = None
        with (
            patch(
                "aiperf.kubernetes.subproc.asyncio.create_subprocess_exec",
                return_value=proc,
            ),
            pytest.raises(RuntimeError, match="without a return code"),
        ):
            await run_command(["true"])

    @pytest.mark.asyncio
    async def test_run_command_decodes_utf8(self) -> None:
        proc = _make_mock_process(stdout="héllo".encode(), stderr="wörld".encode())
        with patch(
            "aiperf.kubernetes.subproc.asyncio.create_subprocess_exec",
            return_value=proc,
        ):
            result = await run_command(["echo"])

        assert result.stdout == "héllo"
        assert result.stderr == "wörld"


# ============================================================
# check_command
# ============================================================


class TestCheckCommand:
    """Verify check_command returns boolean based on exit code."""

    @pytest.mark.asyncio
    async def test_check_command_returns_true_on_success(self) -> None:
        proc = _make_mock_process(returncode=0)
        with patch(
            "aiperf.kubernetes.subproc.asyncio.create_subprocess_exec",
            return_value=proc,
        ):
            assert await check_command(["true"]) is True

    @pytest.mark.asyncio
    async def test_check_command_returns_false_on_failure(self) -> None:
        proc = _make_mock_process(returncode=1)
        with patch(
            "aiperf.kubernetes.subproc.asyncio.create_subprocess_exec",
            return_value=proc,
        ):
            assert await check_command(["false"]) is False


# ============================================================
# start_streaming_process
# ============================================================


class TestStartStreamingProcess:
    """Verify start_streaming_process creates subprocess with correct options."""

    @pytest.mark.asyncio
    async def test_start_streaming_process_default_separate_stderr(self) -> None:
        proc = _make_mock_process()
        with patch(
            "aiperf.kubernetes.subproc.asyncio.create_subprocess_exec",
            return_value=proc,
        ) as mock_exec:
            result = await start_streaming_process(["tail", "-f", "/var/log/app.log"])

        assert result is proc
        mock_exec.assert_awaited_once_with(
            "tail",
            "-f",
            "/var/log/app.log",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    @pytest.mark.asyncio
    async def test_start_streaming_process_merge_stderr(self) -> None:
        proc = _make_mock_process()
        with patch(
            "aiperf.kubernetes.subproc.asyncio.create_subprocess_exec",
            return_value=proc,
        ) as mock_exec:
            await start_streaming_process(["cmd"], merge_stderr=True)

        mock_exec.assert_awaited_once_with(
            "cmd",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )


# ============================================================
# terminate_process
# ============================================================


class TestTerminateProcess:
    """Verify terminate_process graceful shutdown and kill fallback."""

    @pytest.mark.asyncio
    async def test_terminate_process_already_exited_is_noop(self) -> None:
        proc = _make_mock_process()
        proc.returncode = 0
        await terminate_process(proc)
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_terminate_process_graceful_shutdown(self, time_traveler) -> None:
        proc = _make_mock_process()
        proc.returncode = None

        async def _simulate_exit() -> int:
            proc.returncode = 0
            return 0

        proc.wait = AsyncMock(side_effect=_simulate_exit)

        await terminate_process(proc)

        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_terminate_process_kill_after_timeout(self) -> None:
        proc = _make_mock_process()
        proc.returncode = None
        proc.wait = AsyncMock(return_value=-9)

        with patch(
            "asyncio.wait_for",
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError,
        ):
            await terminate_process(proc, timeout=2.0)

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_terminate_process_custom_timeout(self) -> None:
        proc = _make_mock_process()
        proc.returncode = None
        proc.wait = AsyncMock(return_value=-9)

        with patch(
            "asyncio.wait_for",
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError,
        ):
            await terminate_process(proc, timeout=0.5)

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
