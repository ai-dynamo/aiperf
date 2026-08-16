# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent local shell session for tool execution.

Holds one long-lived shell and writes commands to its stdin, delimited by a
sentinel carrying the exit status. The alternative -- spawning a process per
command -- costs a process launch every time; against a containerized session
that same structure saves ~37 ms per command (see
``docs/specs/sandbox-resident-executor.md``).

Each recorded command runs as a FRESH ``bash -lc`` inside the session. The
session is a transport, not a shell the trajectory shares: recordings were
captured with per-call ``bash -lc``, so a persisted working directory or
exported variable would diverge from the run being replayed.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import time
import uuid
from pathlib import Path

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.environment import Environment
from aiperf.graph.sandbox.protocols import ToolResult

_logger = AIPerfLogger(__name__)

DEFAULT_INTERPRETER = ("bash", "-lc")


class LocalSessionSandbox:
    """Run recorded commands in a persistent local shell rooted at a workspace."""

    def __init__(
        self,
        workspace: Path,
        *,
        interpreter: tuple[str, ...] = DEFAULT_INTERPRETER,
        default_timeout_s: float | None = None,
    ) -> None:
        self._workspace = Path(workspace)
        # The path the session shell sees, which a containerized subclass
        # remaps to the mount point inside its container.
        self._workspace_in_sandbox = str(workspace)
        self._interpreter = interpreter
        # Resolved at construction, not at each run, so one trace's commands
        # cannot straddle an env change mid-run.
        self._default_timeout_s = (
            default_timeout_s
            if default_timeout_s is not None
            else Environment.GRAPH_TOOL.COMMAND_TIMEOUT
        )
        self._proc: asyncio.subprocess.Process | None = None

    def _next_sentinel(self) -> str:
        """Mint the end-of-command marker for a single command.

        Per command, not per session: the marker is written into the session
        shell's stdin, so a command could learn it and emit it verbatim. A
        stale marker must not terminate a later read, or the reader would
        return partial output and stay one command behind for the whole trace.
        """
        return f"__AIPERF_TOOL_DONE_{uuid.uuid4().hex}__"

    def _session_argv(self) -> list[str]:
        """The argv that opens the session shell. Overridden by the docker backend."""
        return ["bash"]

    async def open(self) -> None:
        if self._proc is not None:
            # Already open — close the existing session so the caller does not
            # have to track whether open() is re-entrant (e.g. timeout recycle).
            await self.close()
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._proc = await asyncio.create_subprocess_exec(
            *self._session_argv(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(self._workspace),
            # Own process group. On timeout the session shell alone is not
            # enough to kill: the `bash -lc` child it spawned keeps running and
            # competes for CPU with every LATER measured command, silently
            # corrupting the whole duration series. Signalling the group reaches
            # the child and anything it spawned. Harmless on the docker backend,
            # where this wraps the local `docker exec` client.
            start_new_session=True,
        )

    async def run(self, command: str, timeout_s: float | None = None) -> ToolResult:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("sandbox session is not open; call open() first")

        # Each command is a fresh interpreter invocation rooted at the workspace,
        # so state cannot leak between recorded steps.
        sentinel = self._next_sentinel()
        inner = " ".join(shlex.quote(part) for part in self._interpreter)
        # The bare `printf '\n'` before the sentinel is load-bearing: the reader
        # matches the sentinel at the START of a line, so a command whose stdout
        # ends WITHOUT a trailing newline (`printf 'no-newline'`) would otherwise
        # put the sentinel mid-line and the read would never match -- the command
        # would hang until the timeout and then recycle the session (on docker,
        # destroying and recreating the container mid-trace). The extra newline
        # cannot corrupt normal stdout because `_read_until_sentinel` drops the
        # single empty line directly preceding the sentinel, which is exactly the
        # one this printf contributes for a newline-terminated command.
        framed = (
            f"cd {shlex.quote(self._workspace_in_sandbox)} && "
            f"{inner} {shlex.quote(command)}\n"
            "__aiperf_rc=$?\n"
            "printf '\\n'\n"
            f"printf '%s%d\\n' {shlex.quote(sentinel)} $__aiperf_rc\n"
        )
        try:
            self._proc.stdin.write(framed.encode())
            await self._proc.stdin.drain()
        except (RuntimeError, BrokenPipeError, ConnectionResetError) as exc:
            # The session shell is gone. The raw error names an asyncio
            # transport and nothing else, which is useless to an operator whose
            # real problem is usually a task image with no `bash` on PATH.
            raise RuntimeError(await self._session_died_message()) from exc

        started = time.perf_counter()
        try:
            stdout, returncode = await asyncio.wait_for(
                self._read_until_sentinel(sentinel),
                timeout=timeout_s if timeout_s is not None else self._default_timeout_s,
            )
            duration_s = time.perf_counter() - started
            timed_out = False
        except TimeoutError:
            # Stop the clock before recycling: the reported duration is the
            # command's measured cost, not the harness's cleanup cost.
            duration_s = time.perf_counter() - started
            # The session is still mid-command. Recycle it so the next recorded
            # step runs in a clean shell rather than colliding with the
            # abandoned command's output.
            _logger.warning(lambda: f"tool command timed out: {command[:80]!r}")
            await self.close()
            await self.open()
            stdout, returncode, timed_out = "", -1, True
        return ToolResult(
            stdout=stdout,
            returncode=returncode,
            duration_s=duration_s,
            timed_out=timed_out,
        )

    async def _read_until_sentinel(self, sentinel: str) -> tuple[str, int]:
        """Collect output up to the sentinel line, returning it plus the status.

        The framing's own ``printf '\\n'`` contributes exactly one line
        immediately before the sentinel, so that line is dropped when it is
        empty. That makes capture EXACT for every newline-terminated command --
        including one whose output ends in several blank lines, since only the
        single framing newline is removed. A command whose stdout does NOT end
        in a newline gains one, because the framing newline is the only thing
        that terminated its final line; that is the irreducible cost of a
        line-oriented sentinel, and far cheaper than the alternative (never
        matching the sentinel, and hanging until the timeout).
        """
        assert self._proc is not None and self._proc.stdout is not None
        lines: list[str] = []
        marker = sentinel.encode()
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                raise RuntimeError(await self._session_died_message())
            if raw.startswith(marker):
                if lines and lines[-1] == "\n":
                    lines.pop()
                return "".join(lines), int(raw[len(marker) :].strip() or 0)
            lines.append(raw.decode(errors="replace"))

    async def _session_died_message(self) -> str:
        """Explain a dead session shell in terms an operator can act on.

        The bare failure is an asyncio transport error or a silent EOF, neither
        of which names the cause. The overwhelmingly common cause is a session
        argv the environment cannot run -- a container image with no ``bash`` on
        PATH -- and the shell's own output says exactly that, so it is quoted.
        """
        proc, self._proc = self._proc, None
        if proc is None:
            return "sandbox session closed unexpectedly"
        tail = ""
        if proc.stdout is not None:
            with contextlib.suppress(Exception):
                tail = (await proc.stdout.read()).decode(errors="replace").strip()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                proc.wait(), timeout=Environment.GRAPH_TOOL.SESSION_CLOSE_GRACE
            )
        return (
            f"sandbox session {self._session_argv()!r} died (exit "
            f"{proc.returncode}). A tool sandbox needs an interpreter it can "
            f"hold open; a container image without `bash` on PATH is the usual "
            f"cause. Session output: {tail or '<none>'}"
        )

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.stdin is not None and not proc.stdin.is_closing():
            proc.stdin.close()
        try:
            await asyncio.wait_for(
                proc.wait(), timeout=Environment.GRAPH_TOOL.SESSION_CLOSE_GRACE
            )
            return
        except TimeoutError:
            pass
        except ProcessLookupError:
            return
        # Kill the whole GROUP, not just the session shell: `close` is the
        # timeout path's recycle step, and the abandoned `bash -lc` child is
        # precisely what must not survive into the next measured command.
        self._kill_process_group(proc)
        # A killed process that is never reaped stays a zombie and its transport
        # never closes, so the await is not optional.
        with contextlib.suppress(ProcessLookupError, ChildProcessError):
            await proc.wait()

    @staticmethod
    def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
        """SIGKILL the session's process group, falling back to the leader alone.

        Refuses to signal our OWN group: every session here is spawned with
        ``start_new_session=True`` so it leads its own group, but a caller that
        forgot would otherwise have this kill the whole benchmark.
        """
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            pgid = os.getpgid(proc.pid)
            if pgid != os.getpgrp():
                os.killpg(pgid, signal.SIGKILL)
                return
        # The group is gone, unavailable, or is our own; the leader itself is
        # still worth killing.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
