# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared subprocess management utilities.

This module provides reusable subprocess spawning and lifecycle management
used by both MultiProcessServiceManager (for control-plane services) and
WorkerGroupManager (for worker pod subprocesses).
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import platform
import uuid
from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.context import ForkProcess, ForkServerProcess, SpawnProcess
from typing import TYPE_CHECKING

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.environment import Environment
from aiperf.common.error_queue import ErrorQueue
from aiperf.common.logging import LogQueue
from aiperf.common.messages.worker_messages import WorkerStatusSummaryMessage
from aiperf.common.types import ServiceTypeT
from aiperf.plugin.enums import ServiceType
from aiperf.workers.group_runtime import GroupRuntimeRegistration

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


@dataclass(slots=True)
class LocalWorkerGroupManagerAdapter:
    """Local runtime adapter that models a WorkerGroupManager boundary."""

    service_id: str
    """Stable identifier for the local worker-group manager boundary."""

    declared_worker_capacity: int
    """Declared local worker capacity exposed by the adapter."""

    declared_record_processor_capacity: int
    """Declared local record-processor capacity exposed by the adapter."""

    @property
    def group_id(self) -> str:
        """Return the stable runtime group identifier."""
        return self.service_id

    @property
    def declared_workers(self) -> int:
        """Return the declared worker capacity using runtime-contract naming."""
        return self.declared_worker_capacity

    @property
    def declared_record_processors(self) -> int:
        """Return the declared record-processor capacity using runtime naming."""
        return self.declared_record_processor_capacity

    def build_registration(self) -> GroupRuntimeRegistration:
        """Build the runtime registration consumed by WorkerGroupManager."""
        return GroupRuntimeRegistration(
            group_id=self.group_id,
            declared_workers=self.declared_workers,
            declared_record_processors=self.declared_record_processors,
        )

    async def publish_summary(self, summary: WorkerStatusSummaryMessage) -> None:
        """Accept summary publication for local mode without external transport."""
        del summary


@dataclass(slots=True)
class SubprocessInfo:
    """Information about a subprocess managed by SubprocessManager."""

    service_type: ServiceTypeT
    """Type of service running in the process"""

    service_id: str
    """ID of the service running in the process"""

    process: Process | SpawnProcess | ForkProcess | ForkServerProcess | None = None
    """The underlying multiprocessing process instance"""

    launch_adapter: LocalWorkerGroupManagerAdapter | None = None
    """Optional runtime adapter associated with this subprocess entry."""

    parent_service_id: str | None = None
    """Optional parent worker-group manager service ID for local children."""

    @property
    def exitcode(self) -> int | None:
        """Exit code of the process, or None if still running or no process."""
        return self.process.exitcode if self.process else None

    @property
    def pid(self) -> int | None:
        """PID of the process, or None if no process."""
        return self.process.pid if self.process else None


_SPAWN_TIMEOUT = 60.0
"""Safety-net timeout for process.start(). Normal spawns complete in
milliseconds; this guards against extreme system conditions (memory
pressure, exhausted forkserver) blocking the event loop indefinitely."""

_FORKSERVER_PRELOAD = [
    # -- aiperf core (shared by all services) --
    "aiperf.common.bootstrap",
    "aiperf.config",
    "aiperf.common.environment",
    "aiperf.common.logging",
    "aiperf.common.enums",
    "aiperf.common.hooks",
    "aiperf.common.messages",
    "aiperf.common.models",
    "aiperf.common.control_structs",
    "aiperf.common.types",
    "aiperf.plugin",
    "aiperf.plugin.enums",
    "aiperf.common.base_service",
    "aiperf.common.base_component_service",
    "aiperf.common.mixins",
    # -- Worker (replicable: num_workers instances) --
    "aiperf.workers.worker",
    "aiperf.workers.inference_client",
    "aiperf.workers.session_manager",
    "aiperf.credit",
    "aiperf.credit.issuer",
    "aiperf.transports",
    "aiperf.transports.aiohttp_client",
    # -- RecordProcessor (replicable: num_record_processors instances) --
    "aiperf.records.record_processor_service",
    "aiperf.metrics",
    "aiperf.post_processors",
    # -- heavy third-party deps --
    "pydantic",
    "numpy",
    "zmq",
    "uvloop",
    "orjson",
    "msgspec",
    "rich.console",
    "rich.logging",
    "aiohttp",
    "aiofiles",
    "psutil",
]

_mp_context: multiprocessing.context.BaseContext | None = None


def get_mp_context() -> multiprocessing.context.BaseContext:
    """Return the forkserver (Linux) or spawn (macOS) multiprocessing context.

    Lazily created on first call to avoid side-effects at import time
    (e.g. during pytest-xdist worker collection).
    """
    global _mp_context
    if _mp_context is None:
        method = "forkserver" if platform.system() == "Linux" else "spawn"
        _mp_context = multiprocessing.get_context(method)
        if platform.system() == "Linux":
            _mp_context.set_forkserver_preload(_FORKSERVER_PRELOAD)
    return _mp_context


class SubprocessManager:
    """Manages spawning and lifecycle of service subprocesses.

    This utility class provides common subprocess management functionality
    used by both service managers and service components that need to spawn
    child processes.

    Example usage:
        manager = SubprocessManager(run, log_queue)
        await manager.spawn_service(ServiceType.WORKER, "worker_0")
        await manager.stop_all()
    """

    def __init__(
        self,
        run: BenchmarkRun,
        log_queue: LogQueue | None = None,
        error_queue: ErrorQueue | None = None,
        logger: object | None = None,
    ) -> None:
        """Initialize the subprocess manager.

        Args:
            run: BenchmarkRun for spawned services.
            log_queue: Optional multiprocessing queue for centralized logging.
            error_queue: Optional multiprocessing queue for error reporting from child processes.
            logger: Optional logger object with debug/warning/error methods.
        """
        self.run = run
        self.log_queue = log_queue
        self.error_queue = error_queue
        self.subprocesses: list[SubprocessInfo] = []
        self._logger = logger
        self._local_worker_group_manager: SubprocessInfo | None = None
        # Serializes _ensure_local_worker_group_manager across concurrent
        # spawn_service calls (e.g. BaseServiceManager.start_services gathers
        # run_service coroutines for multiple service types in parallel).
        self._local_wgm_lock = asyncio.Lock()

    @property
    def local_worker_group_runtime_adapter(
        self,
    ) -> LocalWorkerGroupManagerAdapter | None:
        """Return the local worker-group runtime adapter when local mode has one."""
        if self._local_worker_group_manager is None:
            return None
        return self._local_worker_group_manager.launch_adapter

    def _debug(self, msg: str) -> None:
        """Log a debug message if logger is available."""
        if self._logger and hasattr(self._logger, "debug"):
            self._logger.debug(msg)

    def _warning(self, msg: str) -> None:
        """Log a warning message if logger is available."""
        if self._logger and hasattr(self._logger, "warning"):
            self._logger.warning(msg)

    def _build_local_worker_group_manager_adapter(
        self,
        service_type: ServiceTypeT,
    ) -> LocalWorkerGroupManagerAdapter | None:
        """Build the local worker-group runtime adapter for child services."""
        if not self.run.cfg.runtime.uses_local_worker_group_manager:
            return None
        if service_type not in {
            ServiceType.WORKER_GROUP_MANAGER,
            ServiceType.WORKER,
            ServiceType.RECORD_PROCESSOR,
        }:
            return None
        return LocalWorkerGroupManagerAdapter(
            service_id="worker_group_manager_local",
            declared_worker_capacity=self.run.cfg.worker_group_declared_worker_capacity,
            declared_record_processor_capacity=(
                self.run.cfg.worker_group_declared_record_processor_capacity
            ),
        )

    async def _start_process(
        self,
        *,
        service_type: ServiceTypeT,
        service_id: str,
        process_kwargs: dict[str, object] | None = None,
        launch_adapter: LocalWorkerGroupManagerAdapter | None = None,
        parent_service_id: str | None = None,
    ) -> SubprocessInfo:
        """Start one subprocess and track its runtime metadata."""
        kwargs = {
            "service_type": service_type,
            "service_id": service_id,
            "run": self.run,
            "log_queue": self.log_queue,
            "error_queue": self.error_queue,
        }
        if process_kwargs:
            kwargs.update(process_kwargs)

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

        try:
            await asyncio.wait_for(
                asyncio.to_thread(process.start),
                timeout=_SPAWN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                process.kill()
            raise RuntimeError(
                f"Timed out spawning {service_type} subprocess "
                f"(id: {service_id}) after {_SPAWN_TIMEOUT}s"
            ) from None

        self._debug(
            f"Spawned {service_type} subprocess (pid: {process.pid}, id: {service_id})"
        )

        info = SubprocessInfo(
            process=process,
            service_type=service_type,
            service_id=service_id,
            launch_adapter=launch_adapter,
            parent_service_id=parent_service_id,
        )
        self.subprocesses.append(info)
        return info

    async def _ensure_local_worker_group_manager(
        self,
        service_type: ServiceTypeT,
    ) -> LocalWorkerGroupManagerAdapter | None:
        """Ensure a local worker-group manager boundary exists before child spawns."""
        adapter = self._build_local_worker_group_manager_adapter(service_type)
        if adapter is None:
            return None
        async with self._local_wgm_lock:
            if self._local_worker_group_manager is None:
                self._local_worker_group_manager = await self._start_process(
                    service_type=ServiceType.WORKER_GROUP_MANAGER,
                    service_id=adapter.service_id,
                    process_kwargs={"runtime_adapter": adapter},
                    launch_adapter=adapter,
                )
                self._debug(
                    "Started local worker-group manager boundary before child launch"
                )
        return adapter

    async def spawn_service(
        self,
        service_type: ServiceTypeT,
        service_id: str | None = None,
        replicable: bool = True,
    ) -> SubprocessInfo:
        """Spawn a single service as a subprocess.

        Args:
            service_type: The type of service to spawn.
            service_id: Optional specific service ID. If None, generates one.
            replicable: Whether the service can have multiple replicas.

        Returns:
            SubprocessInfo with the spawned process details.
        """
        local_runtime_adapter = self._build_local_worker_group_manager_adapter(
            service_type
        )
        if (
            service_type == ServiceType.WORKER_GROUP_MANAGER
            and local_runtime_adapter is not None
        ):
            if service_id is None:
                service_id = local_runtime_adapter.service_id
            else:
                local_runtime_adapter.service_id = service_id
            info = await self._start_process(
                service_type=service_type,
                service_id=service_id,
                process_kwargs={"runtime_adapter": local_runtime_adapter},
                launch_adapter=local_runtime_adapter,
            )
            self._local_worker_group_manager = info
            return info

        local_group_manager = await self._ensure_local_worker_group_manager(
            service_type
        )

        if service_id is None:
            service_id = (
                f"{service_type}_{uuid.uuid4().hex[:8]}"
                if replicable
                else str(service_type)
            )

        return await self._start_process(
            service_type=service_type,
            service_id=service_id,
            launch_adapter=local_group_manager,
            parent_service_id=(
                local_group_manager.service_id
                if local_group_manager is not None
                else None
            ),
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
        infos = []
        for _ in range(num_replicas):
            info = await self.spawn_service(service_type, replicable=replicable)
            infos.append(info)
        return infos

    async def stop_process(
        self,
        info: SubprocessInfo,
        timeout: float = Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT,
    ) -> None:
        """Stop a single subprocess gracefully, killing if necessary.

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
            service_id: Optional specific service ID to stop.

        Returns:
            List of exceptions that occurred during stop, or None for success.
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
            if info is self._local_worker_group_manager:
                self._local_worker_group_manager = None
        return await asyncio.gather(
            *[self.stop_process(info) for info in to_stop],
            return_exceptions=True,
        )

    async def stop_all(self) -> list[BaseException | None]:
        """Stop all managed subprocesses gracefully.

        Returns:
            List of exceptions that occurred during stop, or None for success.
        """
        self._debug("Stopping all subprocesses")
        to_stop = list(self.subprocesses)
        self.subprocesses.clear()
        self._local_worker_group_manager = None
        return await asyncio.gather(
            *[self.stop_process(info) for info in to_stop],
            return_exceptions=True,
        )

    async def kill_all(self) -> list[BaseException | None]:
        """Kill all managed subprocesses immediately.

        Returns:
            List of exceptions that occurred during kill, or None for success.
        """
        self._debug("Killing all subprocesses")
        to_kill = list(self.subprocesses)
        self.subprocesses.clear()
        self._local_worker_group_manager = None

        for info in to_kill:
            if info.process and info.process.is_alive():
                info.process.kill()

        async def _join(info: SubprocessInfo) -> None:
            if info.process:
                await asyncio.to_thread(
                    info.process.join,
                    timeout=Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT,
                )

        return await asyncio.gather(
            *[_join(info) for info in to_kill],
            return_exceptions=True,
        )

    def get_by_type(self, service_type: ServiceTypeT) -> list[SubprocessInfo]:
        """Get all subprocesses of a given service type.

        Args:
            service_type: The service type to filter by.

        Returns:
            List of SubprocessInfo matching the service type.
        """
        return [s for s in self.subprocesses if s.service_type == service_type]

    def check_alive(self) -> list[SubprocessInfo]:
        """Check which subprocesses have died.

        Returns:
            List of SubprocessInfo for dead subprocesses.
        """
        dead: list[SubprocessInfo] = []
        for info in self.subprocesses:
            if info.process and not info.process.is_alive():
                dead.append(info)
        return dead

    def remove(self, info: SubprocessInfo) -> None:
        """Remove a subprocess from tracking.

        Args:
            info: The subprocess info to remove.
        """
        if info in self.subprocesses:
            self.subprocesses.remove(info)
        if info is self._local_worker_group_manager:
            self._local_worker_group_manager = None

    def clear(self) -> None:
        """Clear all subprocess tracking."""
        self.subprocesses.clear()
        self._local_worker_group_manager = None
