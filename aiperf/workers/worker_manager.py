# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import multiprocessing
import sys
import uuid
from multiprocessing import Process
from typing import Any

from pydantic import ConfigDict, Field

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import TASK_CANCEL_TIMEOUT_LONG, TASK_CANCEL_TIMEOUT_SHORT
from aiperf.common.enums import MessageType, ServiceRunType, ServiceType
from aiperf.common.exceptions import ConfigurationError
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    on_cleanup,
    on_init,
    on_stop,
)
from aiperf.common.pydantic_utils import AIPerfBaseModel
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.common.worker_models import WorkerHealthMessage
from aiperf.workers.worker import Worker


class WorkerProcessInfo(AIPerfBaseModel):
    """Information about a worker process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    worker_id: str = Field(..., description="ID of the worker process")
    process: Any = Field(None, description="Process object or task")


@ServiceFactory.register(ServiceType.WORKER_MANAGER)
class WorkerManager(BaseComponentService):
    """
    The WorkerManager service is primary responsibility to manage the worker processes.
    It will spawn the workers, monitor their health, and stop them when the service is stopped.
    In the future it will also be responsible for the auto-scaling of the workers.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig | None = None,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
        )
        self.logger.debug("Initializing worker manager")
        self.workers: dict[str, WorkerProcessInfo] = {}
        self.worker_health: dict[str, WorkerHealthMessage] = {}

        self.cpu_count = multiprocessing.cpu_count()
        self.logger.debug("Detected %s CPU cores/threads", self.cpu_count)

        self.max_concurrency = self.user_config.load.concurrency
        self.max_workers = self.service_config.max_workers
        if self.max_workers is None:
            # Default to the number of CPU cores - 1
            self.max_workers = self.cpu_count - 1

        # Cap the worker count to the max concurrency + 1, but only if the user in in concurrency mode.
        if self.max_concurrency > 1:
            self.max_workers = min(
                self.max_concurrency + 1,
                self.max_workers,
            )

        # Ensure we have at least the min workers
        self.max_workers = max(
            self.max_workers,
            self.service_config.min_workers or 0,
        )
        self.initial_workers = self.max_workers

    @property
    def service_type(self) -> ServiceType:
        return ServiceType.WORKER_MANAGER

    @on_init
    async def _initialize(self) -> None:
        """Initialize worker manager-specific components."""
        self.logger.debug("Initializing worker manager")

        await self.sub_client.subscribe(
            MessageType.WORKER_HEALTH, self._on_worker_health
        )

        # Spawn workers
        # TODO: This logic can be refactored to make use of the ServiceManager class
        if self.service_config.service_run_type == ServiceRunType.MULTIPROCESSING:
            await self._spawn_multiprocessing_workers()

        elif self.service_config.service_run_type == ServiceRunType.KUBERNETES:
            await self._spawn_kubernetes_workers()

        else:
            raise ConfigurationError(
                f"Unsupported run type: {self.service_config.service_run_type}",
            )

    async def _on_worker_health(self, message: WorkerHealthMessage) -> None:
        self.logger.debug("Received worker health message: %s", message)
        self.worker_health[message.service_id] = message

    @on_stop
    async def _stop(self) -> None:
        self.logger.debug("Stopping worker manager")

        # Stop all workers
        # TODO: This logic can be refactored to make use of the ServiceManager class
        if self.service_config.service_run_type == ServiceRunType.MULTIPROCESSING:
            await self._stop_multiprocessing_workers()
        elif self.service_config.service_run_type == ServiceRunType.KUBERNETES:
            await self._stop_kubernetes_workers()
        else:
            raise ConfigurationError(
                f"Unsupported run type: {self.service_config.service_run_type}",
            )

    @on_cleanup
    async def _cleanup(self) -> None:
        self.logger.debug("Cleaning up worker manager")
        self.workers.clear()

    async def _spawn_kubernetes_workers(self) -> None:
        self.logger.debug("Spawning %s worker pods", self.initial_workers)
        # TODO: Implement Kubernetes start
        raise NotImplementedError("Kubernetes start not implemented")

    async def _stop_kubernetes_workers(self) -> None:
        self.logger.debug("Stopping all worker processes")
        # TODO: Implement Kubernetes stop
        raise NotImplementedError("Kubernetes stop not implemented")

    async def _spawn_multiprocessing_workers(self) -> None:
        self.logger.debug("Spawning %s worker processes", self.initial_workers)

        # Get the global log queue for child process logging
        from aiperf.common.logging import get_global_log_queue

        log_queue = get_global_log_queue()

        for _ in range(self.initial_workers):
            worker_id = f"worker_{uuid.uuid4().hex[:8]}"

            process = Process(
                target=bootstrap_and_run_service,
                name=f"{worker_id}_process",
                kwargs={
                    "service_class": Worker,
                    "service_config": self.service_config,
                    "user_config": self.user_config,
                    "log_queue": log_queue,
                    "service_id": worker_id,
                },
                daemon=True,
            )
            process.start()

            self.workers[worker_id] = WorkerProcessInfo(
                worker_id=worker_id,
                process=process,
            )
            self.logger.debug(
                "Started worker process %s (pid: %s)", worker_id, process.pid
            )

    async def _stop_multiprocessing_workers(self) -> None:
        self.logger.debug("Stopping all worker processes")

        # First terminate all processes
        for worker_id, worker_info in self.workers.items():
            self.logger.debug("Stopping worker process %s %s", worker_id, worker_info)
            process = worker_info.process
            if process and process.is_alive():
                self.logger.debug(
                    "Terminating worker process %s (pid: %s)", worker_id, process.pid
                )
                process.terminate()

        # Then wait for all to finish
        await asyncio.gather(
            *[
                self._wait_for_process(worker_id, worker_info.process)
                for worker_id, worker_info in self.workers.items()
                if worker_info.process
            ]
        )

        self.logger.debug("All worker processes stopped")

    async def _wait_for_process(self, worker_id: str, process: Process) -> None:
        """Wait for a process to terminate with timeout handling."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    process.join, timeout=TASK_CANCEL_TIMEOUT_SHORT
                ),  # Add timeout to join
                timeout=TASK_CANCEL_TIMEOUT_LONG,  # Overall timeout
            )
            self.logger.debug(
                "Worker process %s (pid: %s) stopped", worker_id, process.pid
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "Worker process %s (pid: %s) did not terminate gracefully, killing",
                worker_id,
                process.pid,
            )
            process.kill()


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(WorkerManager)


if __name__ == "__main__":
    sys.exit(main())
