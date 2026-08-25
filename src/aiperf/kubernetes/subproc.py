# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async subprocess helpers."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

from aiperf.kubernetes.credential_retry import (
    credential_retry_delay,
    interactive_credential_wait_enabled,
    is_kubectl_authentication_error,
    print_credential_wait,
    print_credentials_restored,
)

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
    wait_for_credentials: bool | None = None,
) -> CommandResult:
    """Run a command asynchronously and capture output.

    Args:
        cmd: Command and arguments to execute.
        timeout: Seconds to wait for the command to exit. Defaults to 60s to
            prevent callers (kubectl/helm probes, log dumps, preflight checks)
            from hanging on an unreachable apiserver. Pass None to disable.
        wait_for_credentials: Retry kubectl authentication failures after the
            user logs in. ``None`` enables this only in interactive terminals.

    Returns:
        CommandResult with returncode, stdout, and stderr.

    Raises:
        TimeoutError: If the command does not exit within ``timeout``.
    """
    is_kubectl = bool(cmd and cmd[0].rsplit("/", maxsplit=1)[-1] == "kubectl")
    should_wait = is_kubectl and (
        interactive_credential_wait_enabled()
        if wait_for_credentials is None
        else wait_for_credentials
    )
    announced = False
    attempt = 0
    while True:
        result = await _run_command_once(cmd, timeout=timeout)
        if not should_wait or not is_kubectl_authentication_error(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        ):
            if announced:
                print_credentials_restored(None)
            return result
        if not announced:
            announced = True
            print_credential_wait(None)
        await asyncio.sleep(credential_retry_delay(attempt))
        attempt += 1


async def _run_command_once(
    cmd: list[str],
    *,
    timeout: float | None,
) -> CommandResult:
    """Run one subprocess attempt and capture its output."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        raw_stdout, raw_stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.CancelledError:
        await terminate_process(proc)
        raise
    except TimeoutError:
        await terminate_process(proc)
        raise
    if proc.returncode is None:
        raise RuntimeError(
            f"Process {cmd!r} exited without a return code after communicate() (pid={proc.pid})"
        )
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
        True if the command succeeded. Returns False on timeout or missing executable.
    """
    try:
        result = await run_command(cmd, timeout=timeout)
    except (TimeoutError, FileNotFoundError):
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
    except TimeoutError:
        proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=1.0)
