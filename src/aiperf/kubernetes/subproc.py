# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async subprocess helpers."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Async subprocess helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result of an async subprocess execution."""

    returncode: int
    """The exit code of the subprocess."""

    stdout: str
    """The standard output of the subprocess."""

    stderr: str
    """The standard error of the subprocess."""

    @property
    def ok(self) -> bool:
        """True if the command exited successfully."""
        return self.returncode == 0


async def run_command(
    cmd: list[str],
    *,
    timeout: float | None = 60.0,
) -> CommandResult:
    """Run a command asynchronously and capture output.

    Args:
        cmd: Command and arguments to execute.
        timeout: Seconds to wait for the command to exit. Defaults to 60s to
            prevent callers (kubectl/helm probes, log dumps, preflight checks)
            from hanging on an unreachable apiserver. Pass None to disable.

    Returns:
        CommandResult with returncode, stdout, and stderr.

    Raises:
        asyncio.TimeoutError: If the command does not exit within ``timeout``.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        raw_stdout, raw_stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        await terminate_process(proc)
        raise
    if proc.returncode is None:
        raise RuntimeError("Process exited without a return code after communicate()")
    return CommandResult(
        returncode=proc.returncode,
        stdout=raw_stdout.decode(errors="replace"),
        stderr=raw_stderr.decode(errors="replace"),
    )


async def check_command(cmd: list[str], *, timeout: float | None = 60.0) -> bool:
    """Run a command and return True if it exits with code 0.

    Args:
        cmd: Command and arguments to execute.
        timeout: Seconds to wait for the command to exit. Defaults to 60s.

    Returns:
        True if the command succeeded. Returns False on timeout.
    """
    try:
        result = await run_command(cmd, timeout=timeout)
    except asyncio.TimeoutError:
        return False
    return result.ok


async def start_streaming_process(
    cmd: list[str],
    *,
    merge_stderr: bool = False,
) -> asyncio.subprocess.Process:
    """Start a long-running subprocess for line-by-line streaming.

    Args:
        cmd: Command and arguments to execute.
        merge_stderr: If True, redirect stderr into stdout.

    Returns:
        The running subprocess (caller must manage cleanup via terminate_process).
    """
    stderr = asyncio.subprocess.STDOUT if merge_stderr else asyncio.subprocess.PIPE
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=stderr,
    )


async def terminate_process(
    proc: asyncio.subprocess.Process,
    timeout: float = 5.0,
) -> None:
    """Gracefully terminate a subprocess with timeout fallback to kill.

    Args:
        proc: The subprocess to terminate.
        timeout: Seconds to wait for graceful exit before killing.
    """
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=1.0)
