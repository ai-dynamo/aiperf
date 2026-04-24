# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.error_queue import ErrorQueue
from aiperf.common.hooks import background_task
from aiperf.common.logging import LogQueue
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.subprocess_manager import SubprocessManager
from aiperf.common.types import ServiceTypeT
from aiperf.controller.base_service_manager import BaseServiceManager
from aiperf.plugin import plugins

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class MultiProcessServiceManager(BaseServiceManager):
    """Service Manager for starting and stopping services as multiprocessing processes.

    Uses SubprocessManager for process lifecycle management and ServiceRegistry
    for centralized registration tracking.
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        run: BenchmarkRun,
        log_queue: LogQueue | None = None,
        error_queue: ErrorQueue | None = None,
        **kwargs,
    ):
        super().__init__(required_services, run, **kwargs)
        self._subprocess_manager = SubprocessManager(
            run=run,
            log_queue=log_queue,
            error_queue=error_queue,
            logger=self,
        )
        self.log_queue = log_queue
        self.error_queue = error_queue
        self._id_counters: dict[ServiceTypeT, int] = {}

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        """Run a service with the given number of replicas."""
        service_metadata = plugins.get_service_metadata(service_type)
        for _ in range(num_replicas):
            idx = self._id_counters.get(service_type, 0)
            self._id_counters[service_type] = idx + 1
            service_id = (
                f"{service_type}_{idx}" if service_metadata.replicable else None
            )
            info = await self._subprocess_manager.spawn_service(
                service_type=service_type,
                service_id=service_id,
                replicable=service_metadata.replicable,
            )
            ServiceRegistry.expect_service(info.service_id, service_type)

    async def stop_service(
        self, service_type: ServiceTypeT, service_id: str | None = None
    ) -> list[BaseException | None]:
        return await self._subprocess_manager.stop_service(service_type, service_id)

    async def shutdown_all_services(self) -> list[BaseException | None]:
        """Stop all required services as multiprocessing processes."""
        self._shutdown_complete = True
        self.debug("Stopping all service processes")
        return await self._subprocess_manager.stop_all()

    async def kill_all_services(self) -> list[BaseException | None]:
        """Kill all required services as multiprocessing processes."""
        self._shutdown_complete = True
        self.debug("Killing all service processes")
        return await self._subprocess_manager.kill_all()

    async def wait_for_all_services_registration(
        self,
        timeout_seconds: float = Environment.SERVICE.REGISTRATION_TIMEOUT,
    ) -> None:
        """Wait for all required services to be registered.

        Raises:
            ServiceProcessDiedError: If a required service process dies while waiting.
            ServiceRegistrationTimeoutError: If services don't register within timeout.
        """
        self.info("Waiting for all required services to register...")
        await ServiceRegistry.wait_for_all(timeout_seconds)

    # -- Process health monitoring --

    @background_task(
        interval=lambda self: Environment.SERVICE.PROCESS_MONITOR_INTERVAL,
        immediate=False,
    )
    async def _monitor_processes(self) -> None:
        """Periodically check for dead service processes.

        Required services trigger a fatal registry failure (waking all waiters).
        Optional services are silently forgotten so the system continues.
        During shutdown, process exits are expected and logged at debug level.
        """
        if self._shutdown_complete or self.stop_requested:
            return

        dead = self._subprocess_manager.check_alive()
        for info in dead:
            self._subprocess_manager.remove(info)
            service_metadata = plugins.get_service_metadata(info.service_type)

            if service_metadata.required:
                self.error(
                    f"Required service process '{info.service_id}' ({info.service_type}) "
                    f"died with exit code {info.exitcode}"
                )
                ServiceRegistry.fail_service(info.service_id, info.service_type)
            else:
                self.warning(
                    f"Optional service process '{info.service_id}' ({info.service_type}) "
                    f"exited with code {info.exitcode} — forgetting it"
                )
                ServiceRegistry.forget(info.service_id)
