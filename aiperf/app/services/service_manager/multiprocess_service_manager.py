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
from multiprocessing import Process

from pydantic import BaseModel, ConfigDict, Field

from aiperf.app.services.service_manager.base_service_manager import BaseServiceManager
from aiperf.common.bootstrap_utils import bootstrap_and_run_service
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import ServiceRegistrationStatus, ServiceType


class MultiProcessRunInfo(BaseModel):
    """Information about a service running as a multiprocessing process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    process: Process | None = Field(default=None)
    service_type: ServiceType = Field(
        ...,
        description="Type of service running in the process",
    )


class MultiProcessServiceManager(BaseServiceManager):
    """
    Service Manager for starting and stopping services as multiprocessing processes.
    """

    def __init__(
        self, required_service_types: list[ServiceType], config: ServiceConfig
    ):
        super().__init__(required_service_types, config)
        self.multi_process_info: list[MultiProcessRunInfo] = []

    async def initialize_all_services(self) -> None:
        """Start all required services as multiprocessing processes."""
        self.logger.debug("Starting all required services as multiprocessing processes")

        # TODO: This is a hack to get the service classes
        # TODO: We should find a better way to do this
        from aiperf.app.services.dataset_manager import DatasetManager
        from aiperf.app.services.post_processor_manager import PostProcessorManager
        from aiperf.app.services.records_manager import RecordsManager
        from aiperf.app.services.timing_manager import TimingManager
        from aiperf.app.services.worker_manager import WorkerManager

        service_class_map = {
            ServiceType.DATASET_MANAGER: DatasetManager,
            ServiceType.TIMING_MANAGER: TimingManager,
            ServiceType.WORKER_MANAGER: WorkerManager,
            ServiceType.RECORDS_MANAGER: RecordsManager,
            ServiceType.POST_PROCESSOR_MANAGER: PostProcessorManager,
        }

        # Create and start all service processes
        for service_type in self.required_service_types:
            service_class = service_class_map.get(service_type)
            if not service_class:
                self.logger.error(f"No service class found for {service_type}")
                continue

            process = Process(
                target=bootstrap_and_run_service,
                name=f"{service_type}_process",
                args=(service_class, self.config),
                daemon=False,
            )
            process.start()

            self.logger.debug(
                "Service %s started as process (pid: %d)",
                service_type,
                process.pid,
            )

            self.multi_process_info.append(
                MultiProcessRunInfo(process=process, service_type=service_type)
            )

            # Sleep to allow the service to register
            await asyncio.sleep(0.01)

    async def stop_all_services(self) -> None:
        """Stop all required services as multiprocessing processes."""
        self.logger.debug("Stopping all service processes")

        # Wait for all to finish in parallel
        await asyncio.gather(
            *[self._wait_for_process(info) for info in self.multi_process_info]
        )

    async def wait_for_all_services_registration(
        self, stop_event: asyncio.Event, timeout_seconds: int = 30
    ) -> None:
        """Wait for all required services to be registered.

        Args:
            stop_event: Event to check if operation should be cancelled
            timeout_seconds: Maximum time to wait in seconds

        Raises:
            Exception if any service failed to register, None otherwise
        """
        self.logger.debug("Waiting for all required services to register...")

        # Get the set of required service types for checking completion
        required_types = set(self.required_service_types)

        # TODO: Can this be done better by using asyncio.Event()?

        try:
            async with asyncio.timeout(timeout_seconds):
                while not stop_event.is_set():
                    # Get all registered service types from the id map
                    registered_types = {
                        service_info.service_type
                        for service_info in self.service_id_map.values()
                        if service_info.registration_status
                        == ServiceRegistrationStatus.REGISTERED
                    }

                    # Check if all required types are registered
                    if required_types.issubset(registered_types):
                        return

                    # Wait a bit before checking again
                    await asyncio.sleep(0.5)

        except asyncio.TimeoutError:
            # Log which services didn't register in time
            registered_types = {
                service_info.service_type
                for service_info in self.service_id_map.values()
                if service_info.registration_status
                == ServiceRegistrationStatus.REGISTERED
            }

            for service_type in required_types - registered_types:
                self.logger.warning(
                    f"Service {service_type} failed to register within timeout"
                )

    async def wait_for_all_services_start(self) -> None:
        """Wait for all required services to be started."""
        self.logger.debug("Waiting for all required services to start...")

        # TODO: Implement this
        self.logger.warning("wait_for_all_services_start not implemented")

    async def _wait_for_process(self, info: MultiProcessRunInfo) -> None:
        """Wait for a process to terminate with timeout handling."""
        if not info.process or not info.process.is_alive():
            return

        try:
            info.process.terminate()
            await asyncio.wait_for(
                asyncio.to_thread(
                    info.process.join, timeout=1.0
                ),  # Add timeout to join
                timeout=5.0,  # Overall timeout
            )
            self.logger.debug(
                "Service %s process stopped (pid: %d)",
                info.service_type,
                info.process.pid,
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "Service %s process (pid: %d) did not terminate gracefully, killing",
                info.service_type,
                info.process.pid,
            )
            info.process.kill()
