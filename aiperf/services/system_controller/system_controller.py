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
import sys
from multiprocessing import Process
from typing import Any

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import ServiceRunType, Topic
from aiperf.common.models.messages import BaseMessage, RegistrationMessage
from aiperf.common.service import BaseService
from aiperf.services.dataset_manager import DatasetManager
from aiperf.services.post_processor_manager import PostProcessorManager
from aiperf.services.records_manager import RecordsManager
from aiperf.services.timing_manager import TimingManager
from aiperf.services.worker_manager import WorkerManager


class SystemController(BaseService):
    def __init__(self, config: ServiceConfig) -> None:
        super().__init__(
            service_type="system_controller", config=config, autostart=True
        )
        self.services: dict[str, Any] = {}

    async def _initialize(self) -> None:
        """Initialize system controller-specific components."""
        self.logger.debug("Initializing System Controller")

    async def _on_start(self) -> None:
        """Start the system controller and launch required services."""
        self.logger.debug("Starting System Controller")
        await self._start_all_services()

    async def _start_all_services(self) -> None:
        """Start all required services."""
        self.logger.debug("Starting all required services")

        if self.config.service_run_type == ServiceRunType.ASYNC:
            await self._start_all_services_asyncio()
        elif self.config.service_run_type == ServiceRunType.MULTIPROCESSING:
            await self._start_all_services_multiprocessing()
        elif self.config.service_run_type == ServiceRunType.KUBERNETES:
            await self._start_all_services_kubernetes()
        else:
            raise ValueError(
                f"Unsupported service run type: {self.config.service_run_type}"
            )

    async def _stop_all_services(self) -> None:
        """Stop all required services."""
        self.logger.debug("Stopping all required services")

        if self.config.service_run_type == ServiceRunType.ASYNC:
            await self._stop_all_services_asyncio()
        elif self.config.service_run_type == ServiceRunType.MULTIPROCESSING:
            await self._stop_all_services_multiprocessing()
        elif self.config.service_run_type == ServiceRunType.KUBERNETES:
            await self._stop_all_services_kubernetes()
        else:
            raise ValueError(
                f"Unsupported service run type: {self.config.service_run_type}"
            )

    async def _on_stop(self) -> None:
        """Stop the system controller and all running services."""
        self.logger.debug("Stopping System Controller")
        await self._stop_all_services()

    async def _cleanup(self) -> None:
        """Clean up system controller-specific components."""
        self.logger.debug("Cleaning up System Controller")

    async def _process_message(self, topic: Topic, message: BaseMessage) -> None:
        self.logger.debug(
            f"Processing message in System Controller: {topic}, {message}"
        )
        if topic == Topic.REGISTRATION:
            await self._process_registration_message(message)
        # TODO: Process other message types

    async def _process_registration_message(self, message: RegistrationMessage) -> None:
        self.logger.debug(f"Processing registration message: {message}")
        # TODO: Process registration message
        raise NotImplementedError

    async def _start_all_services_asyncio(self) -> None:
        """Start all required services as asyncio tasks in the same event loop."""
        self.logger.debug("Starting all required services as tasks")

        # TODO: better way to define these
        service_configs = [
            ("dataset_manager", DatasetManager),
            ("timing_manager", TimingManager),
            ("worker_manager", WorkerManager),
            ("records_manager", RecordsManager),
            ("post_processor_manager", PostProcessorManager),
        ]

        # Create and start all service tasks
        self.service_tasks: dict[str, asyncio.Task] = {}
        for service_name, service_class in service_configs:
            service_instance = service_class(self.config)
            task = asyncio.create_task(service_instance.run())
            task.set_name(f"{service_name}_task")

            self.service_tasks[service_name] = task
            # TODO: Implement a more robust way to track services, shared between the run types using ServiceRunInfo
            self.services[service_name] = {
                "task": task,
                "instance": service_instance,
                "service_class": service_class,
            }

            self.logger.info(f"Service {service_name} started as asyncio task")

    async def _stop_all_services_asyncio(self) -> None:
        """Cancel all service tasks."""
        self.logger.debug("Cancelling all service tasks")

        for service_name, task in self.service_tasks.items():
            self.logger.info(f"Cancelling {service_name} task")
            task.cancel()

        # Wait for all tasks to be cancelled
        pending_tasks = list(self.service_tasks.values())
        if pending_tasks:
            try:
                await asyncio.wait(pending_tasks, timeout=5.0)
            except asyncio.CancelledError:
                self.logger.info("Tasks cancelled successfully")
            except Exception as e:
                self.logger.error(f"Error cancelling tasks: {e}")

    async def _start_all_services_multiprocessing(self) -> None:
        """Start all required services as multiprocessing processes."""
        self.logger.debug("Starting all required services as multiprocessing processes")
        # TODO: better way to define these
        service_configs = [
            ("dataset_manager", DatasetManager),
            ("timing_manager", TimingManager),
            ("worker_manager", WorkerManager),
            ("records_manager", RecordsManager),
            ("post_processor_manager", PostProcessorManager),
        ]

        # Create and start all service processes
        self.service_processes: dict[str, Process] = {}
        for service_name, service_class in service_configs:
            process = Process(
                target=bootstrap_and_run_service,
                name=f"{service_name}_process",
                args=(service_class, self.config),
                daemon=True,
            )
            process.start()
            # TODO: Implement a more robust way to track services, shared between the run types using ServiceRunInfo
            self.service_processes[service_name] = process
            self.logger.info(
                f"Service {service_name} started as multiprocessing process"
            )

    async def _stop_all_services_multiprocessing(self) -> None:
        """Stop all required services as multiprocessing processes."""
        self.logger.debug("Stopping all required services as multiprocessing processes")

        # First terminate all processes
        for service_name, process in self.service_processes.items():
            self.logger.info(f"Stopping {service_name} process (pid: {process.pid})")
            process.terminate()

        # Then wait for all to finish in parallel
        await asyncio.gather(
            *[
                self._wait_for_process(service_name, process)
                for service_name, process in self.service_processes.items()
            ]
        )

    async def _wait_for_process(self, service_name: str, process: Process) -> None:
        """Wait for a process to terminate with timeout handling."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(process.join, timeout=1.0),  # Add timeout to join
                timeout=5.0,  # Overall timeout
            )
            self.logger.info(f"{service_name} process stopped (pid: {process.pid})")
        except asyncio.TimeoutError:
            self.logger.warning(
                f"{service_name} process (pid: {process.pid}) did not terminate gracefully, killing"
            )
            process.kill()

    async def _start_all_services_kubernetes(self) -> None:
        """Start all required services as Kubernetes pods."""
        self.logger.debug("Starting all required services as Kubernetes pods")
        # TODO: Implement Kubernetes
        raise NotImplementedError

    async def _stop_all_services_kubernetes(self) -> None:
        """Stop all required services as Kubernetes pods."""
        self.logger.debug("Stopping all required services as Kubernetes pods")
        # TODO: Implement Kubernetes
        raise NotImplementedError


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(SystemController)


if __name__ == "__main__":
    sys.exit(main())
