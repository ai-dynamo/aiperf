# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-started container pool for zero-cost sandbox opens.

When ``DockerSandboxProvider`` is given a ``pool_size``, it starts that many
containers per image during ``setup_phase()`` (before timing begins). Traces
then check out a slot with ``checkout()`` — an already-running container whose
workspace is bind-mounted and ready — so ``sandbox_setup_s`` drops to
effectively zero.

Pool lifecycle:
    ContainerPool.start()    -- called from DockerSandboxProvider.setup()
    PooledDockerSandbox.open() -- async checkout; blocks until a slot is free
    PooledDockerSandbox.run()  -- fresh docker exec per command (same model as
                                   non-pooled fresh-exec mode)
    PooledDockerSandbox.close() -- return slot; host-side workspace clear
    ContainerPool.stop()     -- called from DockerSandboxProvider.teardown()

Only fresh-exec mode is supported in the pool (persistent_session=True falls
back to the per-trace container path in DockerSandboxProvider).
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.environment import Environment
from aiperf.graph.sandbox.docker import CONTAINER_WORKSPACE
from aiperf.graph.sandbox.local import DEFAULT_INTERPRETER, LocalSessionSandbox
from aiperf.graph.sandbox.protocols import ToolResult

__all__ = ["ContainerPool", "PooledDockerSandbox", "PoolSlot"]

_logger = AIPerfLogger(__name__)


@dataclass(slots=True)
class PoolSlot:
    """One pre-started container available for checkout."""

    image: str
    container_name: str
    workspace: Path


class ContainerPool:
    """A fixed-size pool of pre-started containers for a single Docker image.

    Concurrent traces block on ``checkout()`` when all slots are busy and
    resume as soon as any slot is returned. The pool owns the container
    lifecycle: ``start()`` fires all containers, ``stop()`` tears them all
    down best-effort after the phase ends.
    """

    def __init__(
        self,
        image: str,
        pool_size: int,
        workspace_root: Path,
        *,
        network: str = "none",
        interpreter: tuple[str, ...] = DEFAULT_INTERPRETER,
    ) -> None:
        self._image = image
        self._pool_size = pool_size
        self._workspace_root = workspace_root
        self._network = network
        self._interpreter = interpreter
        self._default_timeout_s: float = Environment.GRAPH_TOOL.COMMAND_TIMEOUT
        self._slots: asyncio.Queue[PoolSlot] = asyncio.Queue()
        self._all_slots: list[PoolSlot] = []

    async def start(self) -> None:
        """Start all containers concurrently. Called before timing begins."""
        pool_id = uuid.uuid4().hex[:8]
        slots = [
            PoolSlot(
                image=self._image,
                container_name=f"aiperf-pool-{pool_id}-{i}",
                workspace=self._workspace_root / f"{pool_id}-{i}",
            )
            for i in range(self._pool_size)
        ]
        _logger.info(
            lambda: f"starting {self._pool_size} pool container(s) for {self._image!r}"
        )
        await asyncio.gather(*[self._start_one(slot) for slot in slots])
        self._all_slots = slots
        for slot in slots:
            self._slots.put_nowait(slot)
        _logger.info(
            lambda: f"container pool ready: {self._pool_size} slot(s) for {self._image!r}"
        )

    async def _start_one(self, slot: PoolSlot) -> None:
        slot.workspace.mkdir(parents=True, exist_ok=True)
        argv = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            slot.container_name,
            "--network",
            self._network,
            "-v",
            f"{slot.workspace}:{CONTAINER_WORKSPACE}",
            "-w",
            CONTAINER_WORKSPACE,
            self._image,
            "sleep",
            "infinity",
        ]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=Environment.GRAPH_TOOL.CONTAINER_START_TIMEOUT,
            )
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            raise RuntimeError(
                f"pool: timed out starting container {slot.container_name!r} from "
                f"{self._image!r} after {Environment.GRAPH_TOOL.CONTAINER_START_TIMEOUT}s"
            ) from None
        if proc.returncode != 0:
            raise RuntimeError(
                f"pool: failed to start container {slot.container_name!r} from "
                f"{self._image!r}: {stderr.decode(errors='replace').strip()}"
            )
        _logger.debug(lambda n=slot.container_name: f"pool slot ready: {n}")

    async def checkout(self) -> PoolSlot:
        """Block until a slot is available and return it."""
        return await self._slots.get()

    async def return_slot(self, slot: PoolSlot) -> None:
        """Return a slot to the pool after clearing its workspace."""
        _clear_workspace(slot.workspace)
        self._slots.put_nowait(slot)

    async def retire_slot(self, slot: PoolSlot) -> None:
        """Retire an unsafe slot and enqueue a freshly started replacement."""
        await self._stop_one(slot)
        replacement_id = uuid.uuid4().hex[:12]
        replacement = PoolSlot(
            image=self._image,
            container_name=f"aiperf-pool-{replacement_id}",
            workspace=self._workspace_root / replacement_id,
        )
        await self._start_one(replacement)
        # Keep the retired slot under pool ownership so teardown retries its
        # removal if the best-effort retirement above stalled.
        self._all_slots.append(replacement)
        self._slots.put_nowait(replacement)

    async def stop(self) -> None:
        """Stop all containers best-effort. Called from teardown_phase()."""
        _logger.info(
            lambda: f"stopping {len(self._all_slots)} pool container(s) for {self._image!r}"
        )
        await asyncio.gather(
            *[self._stop_one(slot) for slot in self._all_slots],
            return_exceptions=True,
        )

    async def _stop_one(self, slot: PoolSlot) -> None:
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                slot.container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            await asyncio.wait_for(
                proc.wait(), timeout=Environment.GRAPH_TOOL.CONTAINER_STOP_TIMEOUT
            )
        except Exception as exc:
            if proc is not None and proc.returncode is None:
                LocalSessionSandbox._kill_process_group(proc)
                with contextlib.suppress(ProcessLookupError, ChildProcessError):
                    await proc.wait()
            _logger.warning(
                lambda exc=exc,
                n=slot.container_name: f"pool: container removal stalled "
                f"({n!r}): {exc!r}"
            )


def _clear_workspace(workspace: Path) -> None:
    """Remove workspace contents between trace checkouts (host-side, no exec needed)."""
    if not workspace.exists():
        return
    for child in workspace.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            pass


class PooledDockerSandbox:
    """Sandbox backed by a pre-started container slot checked out from a pool.

    ``open()`` blocks asynchronously until a slot is available then returns
    immediately — the container is already running, so there is no startup cost
    charged to the trace's timing window.  ``run()`` fires a fresh ``docker
    exec`` per command (same model as non-pooled fresh-exec mode).  ``close()``
    returns the slot to the pool.
    """

    def __init__(
        self,
        pool: ContainerPool,
        *,
        cwd: str = CONTAINER_WORKSPACE,
        interpreter: tuple[str, ...] = DEFAULT_INTERPRETER,
    ) -> None:
        self._pool = pool
        self._slot: PoolSlot | None = None
        self._cwd = cwd
        self._interpreter = interpreter

    async def open(self) -> None:
        """Check out a pre-started container slot. Blocks if all slots are busy."""
        self._slot = await self._pool.checkout()

    async def run(self, command: str, timeout_s: float | None = None) -> ToolResult:
        if self._slot is None:
            raise RuntimeError("PooledDockerSandbox.open() must be called before run()")
        effective_timeout = (
            timeout_s if timeout_s is not None else self._pool._default_timeout_s
        )
        exec_argv = [
            "docker",
            "exec",
            "-w",
            self._cwd,
            self._slot.container_name,
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
            _logger.warning(
                lambda cmd=command: f"pool: tool command timed out: {cmd[:80]!r}"
            )
            LocalSessionSandbox._kill_process_group(proc)
            with contextlib.suppress(ProcessLookupError, ChildProcessError):
                await proc.wait()
            # The in-container command may outlive its docker exec client, so
            # retire the slot before the dispatcher can issue its next command.
            slot, self._slot = self._slot, None
            assert slot is not None
            await self._pool.retire_slot(slot)
        return ToolResult(
            stdout=stdout,
            returncode=returncode,
            duration_s=duration_s,
            timed_out=timed_out,
        )

    async def close(self) -> None:
        """Return the slot to the pool and clear its workspace."""
        if self._slot is not None:
            slot, self._slot = self._slot, None
            await self._pool.return_slot(slot)
