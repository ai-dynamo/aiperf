# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import multiprocessing
import uuid
from multiprocessing import Process
from multiprocessing.context import SpawnProcess

from pydantic import BaseModel, ConfigDict, Field

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.constants import IS_WINDOWS
from aiperf.common.enums import ServiceRegistrationStatus
from aiperf.common.environment import Environment
from aiperf.common.exceptions import AIPerfError
from aiperf.common.types import ServiceTypeT

if IS_WINDOWS:
    # Windows multiprocessing has no fork context — ``ForkProcess`` is
    # undefined on ``multiprocessing.context`` there. Define a stub so the
    # type union below evaluates at class-definition time without the
    # import raising. The stub is never instantiated on Windows because
    # spawn is the only start method available there; it exists purely
    # so Pydantic's annotation resolution doesn't NameError.
    class ForkProcess:  # type: ignore[no-redef]
        pass
else:
    from multiprocessing.context import ForkProcess
from aiperf.controller.base_service_manager import BaseServiceManager


class MultiProcessRunInfo(BaseModel):
    """Information about a service running as a multiprocessing process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    process: Process | SpawnProcess | ForkProcess | None = Field(
        default=None,
        description=(
            "The multiprocessing Process handle for the spawned service. "
            "Subclass varies by start method: ``ForkProcess`` on Linux "
            "fork-context, ``SpawnProcess`` on Windows/macOS spawn-context."
        ),
    )
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
        log_queue: "multiprocessing.Queue | None" = None,
        **kwargs,
    ):
        super().__init__(required_services, **kwargs)
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
                    "run": self.run,
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

        # Wait for every service we've actually spawned, not just the ones in
        # required_services. Optional services (GPU telemetry, server metrics,
        # API) are started via run_service() and tracked in multi_process_info
        # but never added to required_services. On slow targets (Windows VDI
        # multiprocessing.spawn) those optional services register hundreds of
        # ms after the core ones — if we only waited on required_services, the
        # ProfileConfigureCommand would broadcast before the optionals had
        # subscribed, leaving them un-configured and their data missing from
        # the final export.
        required_types = set(
            info.service_type for info in self.multi_process_info
        ) or set(self.required_services.keys())

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

                self._reap_dead_processes_during_registration(required_types)

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

    def _reap_dead_processes_during_registration(
        self, required_types: set[ServiceTypeT]
    ) -> None:
        """Reap dead processes mid-registration: required dying is fatal,
        optional dying gets a warning and is dropped from the wait set.

        Without this differentiation, an optional service crashing during
        init (e.g. GPU telemetry missing a DCGM endpoint) would kill the
        entire benchmark. ``process is None`` is treated as dead — a None
        process means the spawn call failed before producing a handle.

        Mutates ``required_types`` (discards optional dead types) and
        ``self.multi_process_info`` (removes optional dead entries) so the
        caller's wait loop can converge.

        Raises:
            AIPerfError: if any required service has died.
        """
        for info in list(self.multi_process_info):
            is_dead = info.process is None or not info.process.is_alive()
            if not is_dead:
                continue
            exit_code = info.process.exitcode if info.process else None
            if info.service_type in self.required_services:
                raise AIPerfError(
                    f"Required service {info.service_id} died before "
                    f"registering (exit code {exit_code})"
                )
            self.warning(
                f"Optional service {info.service_id!r} exited before "
                f"registering (exit code {exit_code}); continuing "
                f"benchmark without it."
            )
            required_types.discard(info.service_type)
            self.multi_process_info.remove(info)

    async def _wait_for_process(self, info: MultiProcessRunInfo) -> None:
        """Wait for a process to terminate with timeout handling."""
        if not info.process or not info.process.is_alive():
            return

        info.process.terminate()
        await asyncio.to_thread(
            info.process.join, timeout=Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT
        )
        if info.process.is_alive():
            self.warning(
                f"Service {info.service_type} process (pid: {info.process.pid}) did not terminate gracefully, killing"
            )
            info.process.kill()
        else:
            self.debug(
                lambda: f"Service {info.service_type} process stopped (pid: {info.process.pid})"
            )

    async def wait_for_all_services_start(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = Environment.SERVICE.START_TIMEOUT,
    ) -> None:
        """Wait for all required services to be started."""
        raise NotImplementedError(
            "MultiprocessServiceManager.wait_for_all_services_start is not wired up. "
            "Multiprocess services self-report START via heartbeats; if a future caller "
            "needs an explicit barrier here, implement it rather than calling this stub."
        )
