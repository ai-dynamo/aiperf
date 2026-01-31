# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import multiprocessing
import uuid
from multiprocessing import Process
from multiprocessing.context import ForkProcess, SpawnProcess

from pydantic import BaseModel, ConfigDict, Field

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import ServiceRegistrationStatus
from aiperf.common.environment import Environment
from aiperf.common.exceptions import AIPerfError
from aiperf.common.types import ServiceTypeT
from aiperf.controller.base_service_manager import BaseServiceManager


class MultiProcessRunInfo(BaseModel):
    """Information about a service running as a multiprocessing process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    process: Process | SpawnProcess | ForkProcess | None = Field(default=None)
    service_type: ServiceTypeT = Field(
        ...,
        description="Type of service running in the process",
    )
    service_id: str = Field(
        ...,
        description="ID of the service running in the process",
    )


class MultiProcessServiceManager(BaseServiceManager):
    """
    Service Manager for starting and stopping services as multiprocessing processes.
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        service_config: ServiceConfig,
        user_config: UserConfig,
        log_queue: "multiprocessing.Queue | None" = None,
        **kwargs,
    ):
        super().__init__(required_services, service_config, user_config, **kwargs)
        self.multi_process_info: list[MultiProcessRunInfo] = []
        self.log_queue = log_queue

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        """Run a service with the given number of replicas."""
        from aiperf.plugin import plugins

        service_metadata = plugins.get_service_metadata(service_type)
        for _ in range(num_replicas):
            service_id = (
                f"{service_type}_{uuid.uuid4().hex[:8]}"
                if service_metadata.replicable
                else str(service_type)
            )
            process = Process(
                target=bootstrap_and_run_service,
                name=f"{service_type}_process",
                kwargs={
                    "service_type": service_type,
                    "service_id": service_id,
                    "service_config": self.service_config,
                    "user_config": self.user_config,
                    "log_queue": self.log_queue,
                },
                daemon=True,
            )

            process.start()

            self.debug(
                lambda pid=process.pid,
                type=service_type: f"Service {type} started as process (pid: {pid})"
            )

            self.multi_process_info.append(
                MultiProcessRunInfo(
                    process=process,
                    service_type=service_type,
                    service_id=service_id,
                )
            )

    async def stop_service(
        self, service_type: ServiceTypeT, service_id: str | None = None
    ) -> list[BaseException | None]:
        self.debug(lambda: f"Stopping {service_type} process(es) with id: {service_id}")
        tasks = []
        for info in list(self.multi_process_info):
            if info.service_type == service_type and (
                service_id is None or info.service_id == service_id
            ):
                task = asyncio.create_task(self._wait_for_process(info))
                task.add_done_callback(
                    lambda _, info=info: self.multi_process_info.remove(info)
                )
                tasks.append(task)
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown_all_services(self) -> list[BaseException | None]:
        """Stop all required services as multiprocessing processes."""
        self.debug("Stopping all service processes")

        # Wait for all to finish in parallel
        return await asyncio.gather(
            *[self._wait_for_process(info) for info in self.multi_process_info],
            return_exceptions=True,
        )

    async def kill_all_services(self) -> list[BaseException | None]:
        """Kill all required services as multiprocessing processes."""
        self.debug("Killing all service processes")

        # Kill all processes
        for info in self.multi_process_info:
            if info.process:
                info.process.kill()

        # Wait for all to finish in parallel
        return await asyncio.gather(
            *[self._wait_for_process(info) for info in self.multi_process_info],
            return_exceptions=True,
        )

    async def wait_for_all_services_registration(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = Environment.SERVICE.REGISTRATION_TIMEOUT,
    ) -> None:
        """Wait for all required services to be registered.

        Args:
            stop_event: Event to check if operation should be cancelled
            timeout_seconds: Maximum time to wait in seconds

        Raises:
            Exception if any service failed to register, None otherwise
        """
        self.debug("Waiting for all required services to register...")

        # Get the set of required service types for checking completion
        required_types = set(self.required_services.keys())

        # TODO: Can this be done better by using asyncio.Event()?

        async def _wait_for_registration():
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

                for process in self.multi_process_info:
                    if not process.process or not process.process.is_alive():
                        raise AIPerfError(
                            f"Service process {process.service_id} died before registering"
                        )

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

            for service_type in required_types:
                if service_type not in registered_types_set:
                    self.error(
                        f"Service {service_type} failed to register within timeout"
                    )

            raise AIPerfError("Some services failed to register within timeout") from e

    async def _wait_for_process(self, info: MultiProcessRunInfo) -> None:
        """Wait for a process to terminate with timeout handling."""
        if not info.process or not info.process.is_alive():
            return

        try:
            info.process.terminate()
            await asyncio.to_thread(
                info.process.join, timeout=Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT
            )
            self.debug(
                f"Service {info.service_type} process stopped (pid: {info.process.pid})"
            )
        except asyncio.TimeoutError:
            self.warning(
                f"Service {info.service_type} process (pid: {info.process.pid}) did not terminate gracefully, killing"
            )
            info.process.kill()

    async def wait_for_all_services_start(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = Environment.SERVICE.START_TIMEOUT,
    ) -> None:
        """Wait for all required services to be started."""
        self.debug("Waiting for all required services to start...")
        self.warning(
            "Waiting for all required services to start is not implemented for multiprocessing"
        )
