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
from datetime import datetime, timedelta
from multiprocessing import Process
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import ServiceRegistrationStatus, ServiceType
from aiperf.services.system_controller.service_manager import BaseServiceManager


class MultiProcessRunInfo(BaseModel):
    """Information about a service running as a multiprocessing process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    process: Optional[Process] = Field(default=None)
    service_type: ServiceType = Field(
        ...,
        description="Type of service running in the process",
    )


class MultiProcessManager(BaseServiceManager):
    """
    Service Manager for starting and stopping services as multiprocessing processes.
    """

    def __init__(
        self, required_service_types: List[ServiceType], config: ServiceConfig
    ):
        super().__init__(required_service_types, config)

    async def initialize_all_services(self) -> None:
        """Start all required services as multiprocessing processes."""
        self.logger.debug("Starting all required services as multiprocessing processes")

        # TODO: This is a hack to get the service classes
        # TODO: We should find a better way to do this
        from aiperf.services.dataset_manager import DatasetManager
        from aiperf.services.post_processor_manager import PostProcessorManager
        from aiperf.services.records_manager import RecordsManager
        from aiperf.services.timing_manager import TimingManager
        from aiperf.services.worker_manager import WorkerManager

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

            self.logger.info(
                f"Service {service_type} started as process (pid: {process.pid})"
            )

    async def stop_all_services(self) -> None:
        """Stop all required services as multiprocessing processes."""
        self.logger.debug("Stopping all service processes")

        # TODO: Implement this (is it still needed)

        # # First terminate all processes
        # for service_info in self.service_id_map.values():
        #     if service_info.process:
        #         service_info.process.terminate()

        # # Then wait for all to finish in parallel
        # await asyncio.gather(
        #     *[
        #         self._wait_for_process(service_info)
        #         for service_info in self.service_id_map.values()
        #         if service_info.process
        #     ]
        # )

    async def wait_for_all_services_registration(
        self, stop_event: asyncio.Event, timeout_seconds: int = 30
    ) -> bool:
        """Wait for all required services to be registered.

        Args:
            stop_event: Event to check if operation should be cancelled
            timeout_seconds: Maximum time to wait in seconds

        Returns:
            True if all services registered successfully, False otherwise
        """
        self.logger.info("Waiting for all required services to register...")

        # Set the deadline
        deadline = datetime.now() + timedelta(seconds=timeout_seconds)

        # Get the set of required service types for checking completion
        required_types = set(self.required_service_types)

        # Wait until all services are registered or timeout
        while datetime.now() < deadline and not stop_event.is_set():
            # Get all registered service types from the id map
            registered_types = {
                service_info.service_type
                for service_info in self.service_id_map.values()
                if service_info.registration_status
                == ServiceRegistrationStatus.REGISTERED
            }

            # Check if all required types are registered
            if required_types.issubset(registered_types):
                return True

            # Wait a bit before checking again
            await asyncio.sleep(0.5)

        # Log which services didn't register in time
        registered_types = {
            service_info.service_type
            for service_info in self.service_id_map.values()
            if service_info.registration_status == ServiceRegistrationStatus.REGISTERED
        }

        for service_type in required_types - registered_types:
            self.logger.warning(
                f"Service {service_type} failed to register within timeout"
            )

        return False

    async def wait_for_all_services_start(self) -> bool:
        pass

    async def _wait_for_process(self, multi_process_info: MultiProcessRunInfo) -> None:
        """Wait for a process to terminate with timeout handling."""
        if not multi_process_info.process or not multi_process_info.process.is_alive():
            return

        try:
            multi_process_info.process.terminate()
            await asyncio.wait_for(
                asyncio.to_thread(
                    multi_process_info.process.join, timeout=1.0
                ),  # Add timeout to join
                timeout=5.0,  # Overall timeout
            )
            self.logger.info(
                f"{multi_process_info.service_type} process stopped (pid: {multi_process_info.process.pid})"
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                f"{multi_process_info.service_type} process (pid: {multi_process_info.process.pid}) did not terminate gracefully, killing"
            )
            multi_process_info.process.kill()

    async def wait_for_all_services_stop(self) -> bool:
        """Wait for all services to stop."""
        self.logger.debug("Waiting for all services to stop")
        for service_info in self.service_id_map.values():
            if service_info.run_info.process:
                await self._wait_for_process(service_info)
        return True
