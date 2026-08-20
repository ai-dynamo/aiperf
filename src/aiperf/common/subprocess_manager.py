# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared subprocess management utilities.

Provides reusable subprocess spawning and lifecycle management for any
component that launches AIPerf services as child processes (the controller's
service managers and the in-pod worker runtime).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from typing import TYPE_CHECKING

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.environment import Environment
from aiperf.common.error_queue import ErrorQueue
from aiperf.common.logging import LogQueue
from aiperf.common.mp_context import get_mp_context
from aiperf.common.subprocess_models import SubprocessInfo
from aiperf.common.types import ServiceTypeT
from aiperf.plugin.enums import ServiceType

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

__all__ = ["SubprocessInfo", "SubprocessManager", "get_mp_context"]


class _SubprocessLogger:
    """Structural type for the optional logger accepted by ``SubprocessManager``."""

    def debug(self, message: str) -> None: ...

    def warning(self, message: str) -> None: ...


class SubprocessManager:
    """Manages spawning and lifecycle of AIPerf service subprocesses.

    Example:
        ```python
        manager = SubprocessManager(run=run, log_queue=log_queue)
        info = await manager.spawn_service(ServiceType.WORKER, service_id="worker_0")
        ...
        await manager.stop_all()
        ```
    """

    def __init__(
        self,
        run: BenchmarkRun,
        log_queue: LogQueue | None = None,
        error_queue: ErrorQueue | None = None,
        logger: _SubprocessLogger | None = None,
    ) -> None:
        """Initialize the subprocess manager.

        Args:
            run: BenchmarkRun handed to each spawned service.
            log_queue: Optional multiprocessing queue for centralized logging.
            error_queue: Optional multiprocessing queue for error reporting from
                child processes.
            logger: Optional logger object with ``debug``/``warning`` methods.
        """
        self.run = run
        self.log_queue = log_queue
        self.error_queue = error_queue
        self.subprocesses: list[SubprocessInfo] = []
        self._logger = logger
        # Strong refs to detached reaper tasks so the event loop does not GC
        # them mid-flight (asyncio only holds weak references to tasks).
        self._spawn_reapers: set[asyncio.Task[None]] = set()

    def _debug(self, msg: str) -> None:
        """Log a debug message if a logger was supplied."""
        if self._logger and hasattr(self._logger, "debug"):
            self._logger.debug(msg)

    def _warning(self, msg: str) -> None:
        """Log a warning message if a logger was supplied."""
        if self._logger and hasattr(self._logger, "warning"):
            self._logger.warning(msg)

    async def _start_process(
        self,
        *,
        service_type: ServiceTypeT,
        service_id: str,
    ) -> SubprocessInfo:
        """Start one subprocess and track its runtime metadata.

        Raises:
            RuntimeError: If ``Process.start()`` exceeds
                ``Environment.SERVICE.SPAWN_TIMEOUT``.
        """
        kwargs: dict[str, object] = {
            "service_type": service_type,
            "service_id": service_id,
            "run": self.run,
            "log_queue": self.log_queue,
            # Controller PID for the child's PR_SET_PDEATHSIG guard, so a
            # hard-killed parent cannot orphan its service processes.
            "controller_pid": os.getpid(),
            # Backchannel for failures the child accumulates: without it a
            # crashed service's errors die with the process.
            "error_queue": self.error_queue,
        }
        # WorkerGroupManager spawns Worker/RecordProcessor subprocesses of its
        # own, and Python's multiprocessing disallows daemonic processes from
        # having children (AssertionError at spawn time). Keep every other
        # service daemonic so a controller crash takes its children with it.
        _spawns_children = service_type == ServiceType.WORKER_GROUP_MANAGER
        process = get_mp_context().Process(
            target=bootstrap_and_run_service,
            name=f"{service_type}_process",
            kwargs=kwargs,
            daemon=not _spawns_children,
        )

        # Keep a handle to the in-flight start() so that on timeout we can hand
        # it to a detached reaper instead of killing a process whose _popen may
        # not exist yet (Process.kill() before start raises ValueError).
        spawn_timeout = Environment.SERVICE.SPAWN_TIMEOUT
        start_task = asyncio.ensure_future(asyncio.to_thread(process.start))
        try:
            await asyncio.wait_for(asyncio.shield(start_task), timeout=spawn_timeout)
        except TimeoutError:
            self._reap_timed_out_spawn(process, service_type, service_id, start_task)
            raise RuntimeError(
                f"Timed out spawning {service_type} subprocess "
                f"(id: {service_id}) after {spawn_timeout}s"
            ) from None

        self._debug(
            f"Spawned {service_type} subprocess (pid: {process.pid}, id: {service_id})"
        )

        info = SubprocessInfo(
            process=process,
            service_type=service_type,
            service_id=service_id,
        )
        self.subprocesses.append(info)
        return info

    def _reap_timed_out_spawn(
        self,
        process: object,
        service_type: ServiceTypeT,
        service_id: str,
        start_task: asyncio.Future[None],
    ) -> None:
        """Detach a reaper for a spawn that exceeded the spawn timeout.

        The in-flight ``start()`` may still complete after the timeout, leaving
        a live, untracked child that ``stop_all``/``kill_all`` cannot reap. The
        reaper waits for ``start()`` to settle (so ``_popen``/``pid`` exists),
        then kills and joins the child off the event loop.
        """

        async def _reaper() -> None:
            with contextlib.suppress(Exception):
                await start_task
            # Only kill if start() actually produced a live process; killing a
            # never-started Process raises ValueError.
            if getattr(process, "pid", None) is not None:
                with contextlib.suppress(Exception):
                    process.kill()  # type: ignore[attr-defined]
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(
                        process.join,  # type: ignore[attr-defined]
                        Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT,
                    )
            self._warning(f"Reaped timed-out {service_type} spawn (id: {service_id})")

        reaper = asyncio.ensure_future(_reaper())
        self._spawn_reapers.add(reaper)
        reaper.add_done_callback(self._spawn_reapers.discard)

    async def spawn_service(
        self,
        service_type: ServiceTypeT,
        service_id: str | None = None,
        replicable: bool = True,
    ) -> SubprocessInfo:
        """Spawn a single service as a subprocess.

        Args:
            service_type: The type of service to spawn.
            service_id: Optional specific service ID. Generated when None.
            replicable: Whether the service can have multiple replicas. Non-
                replicable services get the service type itself as their ID.

        Returns:
            SubprocessInfo with the spawned process details.
        """
        if service_id is None:
            service_id = (
                f"{service_type}_{uuid.uuid4().hex[:8]}"
                if replicable
                else str(service_type)
            )

        return await self._start_process(
            service_type=service_type,
            service_id=service_id,
        )

    async def spawn_services(
        self,
        service_type: ServiceTypeT,
        num_replicas: int,
        replicable: bool = True,
    ) -> list[SubprocessInfo]:
        """Spawn multiple replicas of a service type.

        Args:
            service_type: The type of service to spawn.
            num_replicas: Number of replicas to spawn.
            replicable: Whether the service can have multiple replicas.

        Returns:
            List of SubprocessInfo for all spawned processes.
        """
        return [
            await self.spawn_service(service_type, replicable=replicable)
            for _ in range(num_replicas)
        ]

    async def stop_process(
        self,
        info: SubprocessInfo,
        timeout: float = Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT,
    ) -> None:
        """Stop a single subprocess gracefully, killing it if it does not exit.

        Args:
            info: The subprocess info to stop.
            timeout: Timeout in seconds for graceful termination.
        """
        if not info.process or not info.process.is_alive():
            return

        info.process.terminate()
        await asyncio.to_thread(info.process.join, timeout=timeout)
        if info.process.is_alive():
            self._warning(
                f"Subprocess {info.service_id} did not terminate gracefully, killing"
            )
            info.process.kill()
            await asyncio.to_thread(info.process.join, timeout=timeout)
        else:
            self._debug(
                f"Subprocess {info.service_type} ({info.service_id}) stopped "
                f"(pid: {info.process.pid})"
            )

    async def stop_service(
        self,
        service_type: ServiceTypeT,
        service_id: str | None = None,
    ) -> list[BaseException | None]:
        """Stop all subprocesses of a given service type.

        Args:
            service_type: The type of service to stop.
            service_id: Optional specific service ID to stop. None stops every
                subprocess of ``service_type``.

        Returns:
            List of exceptions raised while stopping, or None per successful stop.
        """
        self._debug(f"Stopping {service_type} subprocess(es) with id: {service_id}")
        to_stop = [
            info
            for info in self.subprocesses
            if info.service_type == service_type
            and (service_id is None or info.service_id == service_id)
        ]
        for info in to_stop:
            self.subprocesses.remove(info)
        return await asyncio.gather(
            *[self.stop_process(info) for info in to_stop],
            return_exceptions=True,
        )

    async def stop_all(self) -> list[BaseException | None]:
        """Stop all managed subprocesses gracefully.

        Returns:
            List of exceptions raised while stopping, or None per successful stop.
        """
        self._debug("Stopping all subprocesses")
        to_stop = list(self.subprocesses)
        self.subprocesses.clear()
        return await asyncio.gather(
            *[self.stop_process(info) for info in to_stop],
            return_exceptions=True,
        )

    async def kill_all(self) -> list[BaseException | None]:
        """Kill all managed subprocesses immediately.

        Returns:
            List of exceptions raised while killing, or None per successful kill.
        """
        self._debug("Killing all subprocesses")
        to_kill = list(self.subprocesses)
        self.subprocesses.clear()

        for info in to_kill:
            if info.process and info.process.is_alive():
                info.process.kill()

        async def _join(info: SubprocessInfo) -> None:
            if info.process:
                await asyncio.to_thread(
                    info.process.join,
                    Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT,
                )

        return await asyncio.gather(
            *[_join(info) for info in to_kill],
            return_exceptions=True,
        )

    def get_by_type(self, service_type: ServiceTypeT) -> list[SubprocessInfo]:
        """Get all tracked subprocesses of a given service type.

        Args:
            service_type: The service type to filter by.

        Returns:
            List of SubprocessInfo matching the service type.
        """
        return [s for s in self.subprocesses if s.service_type == service_type]

    def check_alive(self) -> list[SubprocessInfo]:
        """Check which tracked subprocesses have died.

        Returns:
            List of SubprocessInfo for dead subprocesses. Entries with no
            process handle are treated as not-yet-started, not dead.
        """
        return [
            info
            for info in self.subprocesses
            if info.process and not info.process.is_alive()
        ]

    def remove(self, info: SubprocessInfo) -> None:
        """Remove a subprocess from tracking without stopping it.

        Args:
            info: The subprocess info to remove.
        """
        if info in self.subprocesses:
            self.subprocesses.remove(info)

    def clear(self) -> None:
        """Clear all subprocess tracking without stopping any process."""
        self.subprocesses.clear()
