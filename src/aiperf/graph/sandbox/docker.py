# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trace Docker container sandbox for tool execution.

Starts one detached container per trace instance with the workspace bind-mounted.
Each recorded command runs as a fresh ``docker exec`` against the same container,
matching Agent Trace Replay's own execution model: the container is persistent across
commands, but no bash session is shared between them.

The executor stays on the HOST -- it is not installed into the task image.
Injecting a Python runtime into a curated task image would change the
environment the tools observe (a task shelling `python`, or reading `PATH` or
`site-packages`, would see the harness), which is a fidelity bug rather than
mere bloat. See ``docs/specs/sandbox-resident-executor.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from pathlib import Path

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.environment import Environment
from aiperf.graph.sandbox.local import (
    DEFAULT_INTERPRETER,
    LocalSessionSandbox,
)
from aiperf.graph.sandbox.protocols import ToolResult

_logger = AIPerfLogger(__name__)

CONTAINER_WORKSPACE = "/workspace"


class DockerSessionSandbox(LocalSessionSandbox):
    """Run recorded commands inside a per-trace container.

    The container is started on ``open()`` and removed on ``close()``.

    Two execution modes are available via ``persistent_session``:

    * ``False`` (default) — fresh ``docker exec`` per command, matching Agent
      Trace Replay's execution model. Container startup cost lands in
      ``sandbox_setup_s``; per-command overhead includes the Docker daemon
      roundtrip (~37ms on bare metal). Use this to measure tool latency as OL
      would see it.

    * ``True`` — a single ``docker exec -i bash`` session is kept open for the
      lifetime of the trace; commands are piped through stdin as
      ``bash -lc <cmd>`` delimited by per-command sentinels. Container startup
      cost is shared across commands (~37ms amortised), making per-command
      overhead negligible. Use this to isolate model latency without Docker
      exec overhead.
    """

    def __init__(
        self,
        image: str,
        workspace: Path,
        *,
        network: str = "none",
        container_name: str | None = None,
        cwd: str = CONTAINER_WORKSPACE,
        interpreter: tuple[str, ...] = DEFAULT_INTERPRETER,
        default_timeout_s: float | None = None,
        persistent_session: bool = False,
    ) -> None:
        super().__init__(
            workspace=workspace,
            interpreter=interpreter,
            default_timeout_s=default_timeout_s,
        )
        self._image = image
        self._network = network
        self.container_name = container_name or f"aiperf-tool-{uuid.uuid4().hex[:12]}"
        self._persistent_session = persistent_session
        self._retired = False
        # Commands run against the mount point inside the container, not the
        # host path the bind mount came from.
        self._workspace_in_sandbox = cwd

    def _session_argv(self) -> list[str]:
        """Open a persistent interactive bash session inside the container."""
        return ["docker", "exec", "-i", self.container_name, "bash"]

    def start_argv(self) -> list[str]:
        """The argv that starts the detached container."""
        return [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            self.container_name,
            "--network",
            self._network,
            "-v",
            f"{self._workspace}:{CONTAINER_WORKSPACE}",
            "-w",
            self._workspace_in_sandbox,
            self._image,
            "sleep",
            "infinity",
        ]

    def stop_argv(self) -> list[str]:
        """The argv that force-removes the container."""
        return ["docker", "rm", "-f", self.container_name]

    async def open(self) -> None:
        """Start the container; open a bash session if in persistent-session mode."""
        if self._retired:
            raise RuntimeError("timed-out Docker sandbox is retired")
        self._workspace.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            *self.start_argv(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to start tool container {self.container_name!r} from "
                f"image {self._image!r}: {stderr.decode(errors='replace').strip()}"
            )
        _logger.debug(lambda: f"started tool container {self.container_name}")
        if self._persistent_session:
            await super().open()

    async def run(self, command: str, timeout_s: float | None = None) -> ToolResult:
        if self._persistent_session:
            return await super().run(command, timeout_s)
        return await self._run_fresh_exec(command, timeout_s)

    async def _run_fresh_exec(
        self, command: str, timeout_s: float | None = None
    ) -> ToolResult:
        """Run one command in the container via a fresh ``docker exec``."""
        if self._retired:
            raise RuntimeError("timed-out Docker sandbox is retired")
        effective_timeout = (
            timeout_s if timeout_s is not None else self._default_timeout_s
        )
        exec_argv = [
            "docker",
            "exec",
            "-w",
            self._workspace_in_sandbox,
            self.container_name,
            *self._interpreter,
            command,
        ]
        started = time.perf_counter()
        timed_out = False
        proc = await asyncio.create_subprocess_exec(
            *exec_argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=effective_timeout
            )
            duration_s = time.perf_counter() - started
            stdout = stdout_bytes.decode(errors="replace")
            returncode = proc.returncode if proc.returncode is not None else 0
        except TimeoutError:
            duration_s = time.perf_counter() - started
            timed_out = True
            stdout = ""
            returncode = -1
            _logger.warning(lambda: f"tool command timed out: {command[:80]!r}")
            self._kill_process_group(proc)
            with contextlib.suppress(ProcessLookupError, ChildProcessError):
                await proc.wait()
            # Killing the docker exec client does not guarantee that the command
            # inside the container stopped. Retire the whole container so no
            # later command can overlap with the timed-out process.
            self._retired = True
            await self.close()
        return ToolResult(
            stdout=stdout,
            returncode=returncode,
            duration_s=duration_s,
            timed_out=timed_out,
        )

    async def close(self) -> None:
        """Close bash session (persistent mode), then remove the container."""
        if self._persistent_session:
            await super().close()
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.stop_argv(),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            await asyncio.wait_for(
                proc.wait(), timeout=Environment.GRAPH_TOOL.CONTAINER_STOP_TIMEOUT
            )
        except Exception as exc:
            if proc is not None and proc.returncode is None:
                self._kill_process_group(proc)
                with contextlib.suppress(ProcessLookupError, ChildProcessError):
                    await proc.wait()
            _logger.warning(
                lambda exc=exc: f"failed to remove tool container "
                f"{self.container_name!r}; it may have leaked: {exc!r}"
            )
