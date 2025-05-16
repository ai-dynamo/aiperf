#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import asyncio
import multiprocessing
import sys
from typing import Any

from pydantic import BaseModel, Field

from aiperf.common.bootstrap_utils import bootstrap_and_run_service
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import ServiceRunType, ServiceType
from aiperf.common.models.payload_models import BasePayload
from aiperf.common.service.base_component_service import BaseComponentService
from aiperf.services.worker.worker import Worker


class WorkerProcess(BaseModel):
    """Information about a worker process."""

    worker_id: str = Field(..., description="ID of the worker process")
    process: Any = Field(None, description="Process object or task")


class WorkerManager(BaseComponentService):
    def __init__(self, service_config: ServiceConfig, service_id: str = None) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.logger.debug("Initializing worker manager")
        self.workers: dict[str, WorkerProcess] = {}
        # TODO: Need to implement some sort of max workers
        self.cpu_count = multiprocessing.cpu_count()
        self.worker_count = self.cpu_count
        self.logger.debug(
            f"Detected {self.cpu_count} CPU threads. "
            f"Spawning {self.worker_count} workers"
        )

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.WORKER_MANAGER

    async def _initialize(self) -> None:
        """Initialize worker manager-specific components."""
        self.logger.debug("Initializing worker manager")

    async def _on_start(self) -> None:
        """Start the worker manager."""
        self.logger.debug("Starting worker manager")

        # Spawn workers based on CPU count
        if self.service_config.service_run_type == ServiceRunType.MULTIPROCESSING:
            await self._spawn_multiprocessing_workers()
        elif self.service_config.service_run_type == ServiceRunType.KUBERNETES:
            await self._spawn_kubernetes_workers()
        else:
            self.logger.warning(
                f"Unsupported run type: {self.service_config.service_run_type}"
            )

    async def _on_stop(self) -> None:
        """Stop the worker manager."""
        self.logger.debug("Stopping worker manager")
        # TODO: This needs to be investigated, as currently we handle the exit signal
        #       by all workers already, so need to understand best way to handle this
        # # Stop all workers
        # if self.service_config.service_run_type == ServiceRunType.MULTIPROCESSING:
        #     await self._stop_multiprocessing_workers()
        # elif self.service_config.service_run_type == ServiceRunType.KUBERNETES:
        #     await self._stop_kubernetes_workers()
        # else:
        #     self.logger.warning(
        #         f"Unsupported run type: {self.service_config.service_run_type}"
        #     )

    async def _cleanup(self) -> None:
        """Clean up worker manager-specific components."""
        self.logger.debug("Cleaning up worker manager")
        self.workers.clear()

    async def _spawn_kubernetes_workers(self) -> None:
        """Spawn worker processes using Kubernetes."""
        self.logger.debug(f"Spawning {self.worker_count} worker processes")

        # TODO: Implement Kubernetes start
        raise NotImplementedError("Kubernetes start not implemented")

    async def _stop_kubernetes_workers(self) -> None:
        """Stop worker processes using Kubernetes."""
        self.logger.debug("Stopping all worker processes")

        # TODO: Implement Kubernetes stop
        raise NotImplementedError("Kubernetes stop not implemented")

    async def _spawn_multiprocessing_workers(self) -> None:
        """Spawn worker processes using multiprocessing."""
        self.logger.debug(f"Spawning {self.worker_count} worker processes")

        for i in range(self.worker_count):
            worker_id = f"worker_{i}"
            process = multiprocessing.Process(
                target=bootstrap_and_run_service,
                name=f"worker_{i}_process",
                args=(Worker, self.service_config),
                daemon=True,
            )
            process.start()
            self.workers[worker_id] = WorkerProcess(
                worker_id=worker_id, process=process
            )
            self.logger.debug(
                f"Started worker process {worker_id} (pid: {process.pid})"
            )

    async def _stop_multiprocessing_workers(self) -> None:
        """Stop all multiprocessing worker processes."""
        self.logger.debug("Stopping all worker processes")

        # First terminate all processes
        for worker_id, worker_info in self.workers.items():
            self.logger.debug(f"Stopping worker process {worker_id} {worker_info}")
            process = worker_info.process
            if process and process.is_alive():
                self.logger.debug(
                    f"Terminating worker process {worker_id} (pid: {process.pid})"
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

    async def _wait_for_process(
        self, worker_id: str, process: multiprocessing.Process
    ) -> None:
        """Wait for a process to terminate with timeout handling."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(process.join, timeout=1.0),  # Add timeout to join
                timeout=5.0,  # Overall timeout
            )
            self.logger.debug(
                f"Worker process {worker_id} (pid: {process.pid}) stopped"
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Worker process {worker_id} (pid: {process.pid}) did not "
                f"terminate gracefully, killing"
            )
            process.kill()

    async def _configure(self, payload: BasePayload) -> None:
        """Configure the worker manager."""
        self.logger.debug(f"Configuring worker manager with payload: {payload}")
        # TODO: Implement worker manager configuration


def main() -> None:
    from aiperf.common.bootstrap_utils import bootstrap_and_run_service

    bootstrap_and_run_service(WorkerManager)


if __name__ == "__main__":
    sys.exit(main())
