# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import multiprocessing
from multiprocessing import Process
from multiprocessing.context import ForkProcess, SpawnProcess

from pydantic import BaseModel, ConfigDict, Field

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import (
    GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS,
    TASK_CANCEL_TIMEOUT_SHORT,
)
from aiperf.common.enums import ServiceRegistrationStatus, ServiceType
from aiperf.common.exceptions import ServiceError
from aiperf.common.factories import ServiceFactory
from aiperf.services.service_manager.base import BaseServiceManager


class MultiProcessRunInfo(BaseModel):
    """Information about a service running as a multiprocessing process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    process: Process | SpawnProcess | ForkProcess | None = Field(default=None)
    service_type: ServiceType = Field(
        ...,
        description="Type of service running in the process",
    )


class MultiProcessServiceManager(BaseServiceManager):
    """
    Service Manager for starting and stopping services as multiprocessing processes.
    """

    def __init__(
        self,
        required_services: dict[ServiceType, int],
        config: ServiceConfig,
        user_config: UserConfig | None = None,
        log_queue: "multiprocessing.Queue | None" = None,
    ):
        super().__init__(required_services, config)
        self.multi_process_info: list[MultiProcessRunInfo] = []
        self.log_queue = log_queue
        self.user_config = user_config

    async def _run_services(self, service_types: dict[ServiceType, int]) -> None:
        """Run a list of services as multiprocessing processes."""

        # Create and start all service processes
        for service_type, count in service_types.items():
            service_class = ServiceFactory.get_class_from_type(service_type)

            for _ in range(count):
                process = Process(
                    target=bootstrap_and_run_service,
                    name=f"{service_type}_process",
                    kwargs={
                        "service_class": service_class,
                        "service_id": service_type.value if count == 1 else None,
                        "service_config": self.config,
                        "user_config": self.user_config,
                        "log_queue": self.log_queue,
                    },
                    daemon=True,
                )
                if service_type in [
                    ServiceType.WORKER_MANAGER,
                ]:
                    process.daemon = False  # Worker manager cannot be a daemon because it needs to be able to spawn worker processes

                process.start()

                self.debug(
                    lambda pid=process.pid,
                    type=service_type: f"Service {type} started as process (pid: {pid})"
                )

                self.multi_process_info.append(
                    MultiProcessRunInfo(process=process, service_type=service_type)
                )

    async def run_all_services(self) -> None:
        """Start all required services as multiprocessing processes."""
        self.logger.debug("Starting all required services as multiprocessing processes")

        try:
            await self._run_services(self.required_services)
        except Exception as e:
            self.logger.error("Error starting services: %s", e)
            raise e

    async def shutdown_all_services(self) -> None:
        """Stop all required services as multiprocessing processes."""
        self.logger.debug("Stopping all service processes")

        # Wait for all to finish in parallel
        await asyncio.gather(
            *[self._wait_for_process(info) for info in self.multi_process_info]
        )

    async def kill_all_services(self) -> None:
        """Kill all required services as multiprocessing processes."""
        self.logger.debug("Killing all service processes")

        # Kill all processes
        for info in self.multi_process_info:
            if info.process:
                info.process.kill()

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
        required_types = set(self.required_services)

        # TODO: Can this be done better by using asyncio.Event()?

        async def _wait_for_registration():
            required_types_set = set(typ for typ, _ in required_types)

            while not stop_event.is_set():
                # Get all registered service types from the id map
                registered_types = {
                    service_info.service_type
                    for service_info in self.service_id_map.values()
                    if service_info.registration_status
                    == ServiceRegistrationStatus.REGISTERED
                }

                # Check if all required types are registered
                if required_types_set.issubset(registered_types):
                    return

                # Wait a bit before checking again
                await asyncio.sleep(0.5)

        try:
            await asyncio.wait_for(_wait_for_registration(), timeout=timeout_seconds)
        except asyncio.TimeoutError as e:
            # Log which services didn't register in time
            registered_types_set = set(
                service_info.service_type
                for service_info in self.service_id_map.values()
                if service_info.registration_status
                == ServiceRegistrationStatus.REGISTERED
            )

            for service_type, _ in required_types:
                if service_type not in registered_types_set:
                    self.logger.error(
                        f"Service {service_type} failed to register within timeout"
                    )

            raise ServiceError(
                "Some services failed to register within timeout",
                ServiceType.SYSTEM_CONTROLLER,
                "system_controller",  # TODO: Get the service ID from the system controller
            ) from e

    async def _wait_for_process(self, info: MultiProcessRunInfo) -> None:
        """Wait for a process to terminate with timeout handling."""
        if not info.process or not info.process.is_alive():
            return

        try:
            info.process.terminate()
            await asyncio.wait_for(
                asyncio.to_thread(
                    info.process.join, timeout=TASK_CANCEL_TIMEOUT_SHORT
                ),  # Add timeout to join
                timeout=GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS,  # Overall timeout
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
