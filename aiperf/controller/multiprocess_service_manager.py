# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import sys
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import (
    DEFAULT_SERVICE_REGISTRATION_TIMEOUT,
    DEFAULT_SERVICE_START_TIMEOUT,
    TASK_CANCEL_TIMEOUT_SHORT,
)
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ServiceRegistrationStatus, ServiceRunType
from aiperf.common.exceptions import AIPerfError
from aiperf.common.factories import ServiceManagerFactory
from aiperf.common.logging import handle_subprocess_log_line
from aiperf.common.protocols import ServiceManagerProtocol
from aiperf.common.types import ServiceTypeT
from aiperf.common.utils import yield_to_event_loop
from aiperf.controller.base_service_manager import BaseServiceManager


class AsyncSubprocessRunInfo(BaseModel):
    """Information about a service running as an asyncio subprocess."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    process: asyncio.subprocess.Process | None = Field(default=None)
    service_type: ServiceTypeT = Field(
        ...,
        description="Type of service running in the subprocess",
    )
    service_id: str = Field(
        ...,
        description="ID of the service running in the subprocess",
    )


@implements_protocol(ServiceManagerProtocol)
@ServiceManagerFactory.register(ServiceRunType.MULTIPROCESSING)
class MultiProcessServiceManager(BaseServiceManager):
    """
    Service Manager for starting and stopping services as asyncio subprocesses.
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        service_config: ServiceConfig,
        user_config: UserConfig,
        **kwargs,
    ):
        super().__init__(required_services, service_config, user_config, **kwargs)
        self.subprocess_info: list[AsyncSubprocessRunInfo] = []

    def _get_service_module_path(self, service_type: ServiceTypeT) -> str:
        """Get the module path for a service type from the registered factory."""
        try:
            # Get the service class from the factory
            from aiperf.common.factories import ServiceFactory

            service_class = ServiceFactory.get_class_from_type(service_type)
            # Return the full module path where the service class is defined
            return service_class.__module__
        except Exception as e:
            raise ValueError(
                f"Failed to get module path for service type {service_type}: {e}"
            ) from e

    async def _run_service_replica(
        self,
        service_type: ServiceTypeT,
        service_id: str,
        user_config_json: str,
        service_config_json: str,
        env: dict[str, str],
        current_dir: Path,
    ) -> None:
        """Run a service replica with the given service id."""
        # Create subprocess arguments using cli runner
        service_module = self._get_service_module_path(service_type)
        args = [
            sys.executable,
            "-m",
            service_module,
            service_config_json,
            user_config_json,
            "--service-id",
            service_id,
            "--use-structured-logging",
        ]

        self.debug(
            lambda args=args: f"Starting subprocess for {service_type} with command: {' '.join(args[:3])} ..."
        )

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=current_dir,
        )

        self.info(f"Service {service_id} started as subprocess (PID: {process.pid})")

        info = AsyncSubprocessRunInfo(
            process=process,
            service_type=service_type,
            service_id=service_id,
        )
        self.subprocess_info.append(info)

        # self.execute_async(self._watch_subprocess(info))

        self.execute_async(self._handle_subprocess_output(process, service_id))
        await yield_to_event_loop()

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        """Run a service with the given number of replicas."""
        tasks = []

        service_config_json = self.service_config.model_dump_json()
        user_config_json = self.user_config.model_dump_json(exclude_unset=True)

        # Prepare environment for subprocess
        env = os.environ.copy()

        # Ensure the current working directory is in PYTHONPATH
        current_dir = Path.cwd()
        python_path = env.get("PYTHONPATH", "")
        if python_path:
            env["PYTHONPATH"] = f"{current_dir}:{python_path}"
        else:
            env["PYTHONPATH"] = str(current_dir)

        for _ in range(num_replicas):
            service_id = f"{service_type}_{uuid.uuid4().hex[:8]}"

            task = asyncio.create_task(
                self._run_service_replica(
                    service_type,
                    service_id,
                    user_config_json,
                    service_config_json,
                    env,
                    current_dir,
                )
            )
            tasks.append(task)
            await yield_to_event_loop()

        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_service(
        self, service_type: ServiceTypeT, service_id: str | None = None
    ) -> list[BaseException | None]:
        self.debug(
            lambda: f"Stopping {service_type} subprocess(es) with id: {service_id}"
        )
        tasks = []
        for info in list(self.subprocess_info):
            if info.service_type == service_type and (
                service_id is None or info.service_id == service_id
            ):
                task = asyncio.create_task(self._wait_for_subprocess(info))
                task.add_done_callback(
                    lambda _, info=info: self.subprocess_info.remove(info)
                )
                tasks.append(task)
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown_all_services(self) -> list[BaseException | None]:
        """Stop all required services as asyncio subprocesses."""
        self.debug("Stopping all service subprocesses")

        # Wait for all to finish in parallel
        return await asyncio.gather(
            *[self._wait_for_subprocess(info) for info in self.subprocess_info],
            return_exceptions=True,
        )

    async def kill_all_services(self) -> list[BaseException | None]:
        """Kill all required services as asyncio subprocesses."""
        self.debug("Killing all service subprocesses")

        # Kill all subprocesses
        for info in self.subprocess_info:
            if info.process and info.process.returncode is None:
                info.process.kill()

        # Wait for all to finish in parallel
        return await asyncio.gather(
            *[self._wait_for_subprocess(info) for info in self.subprocess_info],
            return_exceptions=True,
        )

    async def wait_for_all_services_registration(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = DEFAULT_SERVICE_REGISTRATION_TIMEOUT,
    ) -> None:
        """Wait for all required services to be registered.

        Args:
            stop_event: Event to check if operation should be cancelled
            timeout_seconds: Maximum time to wait in seconds

        Raises:
            AIPerfError if any service failed to register or died before registering
        """
        self.debug("Waiting for all required services to register...")

        # Get the set of required service types for checking completion
        required_types = set(self.required_services.keys())

        # TODO: Can this be done better by using asyncio.Event()?

        async def _wait_for_registration():
            while not stop_event.is_set():
                # Check for dead processes first
                for info in self.subprocess_info:
                    # Check if process is None (failed to start) or has terminated
                    if info.process is None or info.process.returncode is not None:
                        raise AIPerfError(
                            f"Service process {info.service_id} died before registering"
                        )

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

    async def _handle_subprocess_output(
        self, process: asyncio.subprocess.Process, service_id: str
    ) -> None:
        """Handle stdout and stderr output from subprocess."""
        self.debug(
            lambda: f"Starting output handling for subprocess {service_id} (PID: {process.pid})"
        )

        async def _read_stream(stream, stream_name):
            if stream is None:
                return
            try:
                buffer_chunks = []

                while True:
                    chunk = await stream.read(8192)  # Read up to 8KB at a time
                    if not chunk:
                        # Process any remaining data in buffer
                        if buffer_chunks:
                            remaining_data = b"".join(buffer_chunks)
                            decoded_line = remaining_data.decode().rstrip()
                            if decoded_line:
                                handle_subprocess_log_line(decoded_line, service_id)
                        break

                    buffer_chunks.append(chunk)

                    # Process complete lines when we have newlines
                    if b"\n" in chunk:
                        # Join all chunks to process lines
                        buffer_data = b"".join(buffer_chunks)
                        lines = buffer_data.split(b"\n")

                        # Process all complete lines (all except the last)
                        for line in lines[:-1]:
                            decoded_line = line.decode().rstrip()
                            if decoded_line:
                                handle_subprocess_log_line(decoded_line, service_id)

                        # Keep the last part (incomplete line) in buffer
                        buffer_chunks = [lines[-1]] if lines[-1] else []
                        await asyncio.sleep(0.01)

                    # Yield to the event loop to prevent starvation of other tasks because
                    # of reading too frequently from the subprocess
                    await yield_to_event_loop()

            except Exception as e:
                self.warning(f"Error reading {stream_name} for {service_id}: {e}")

        try:
            await asyncio.gather(
                _read_stream(process.stdout, "stdout"),
                _read_stream(process.stderr, "stderr"),
                return_exceptions=True,
            )
        except Exception as e:
            self.warning(f"Error in subprocess output handling for {service_id}: {e}")

        # Wait for process to complete and log exit code
        try:
            exit_code = await process.wait()
            if exit_code != 0:
                self.warning(
                    f"Subprocess {service_id} (PID: {process.pid}) exited with code: {exit_code}"
                )
            else:
                self.debug(
                    lambda: f"Subprocess {service_id} (PID: {process.pid}) exited successfully"
                )
        except Exception as e:
            self.warning(f"Error waiting for subprocess {service_id}: {e}")

    async def _watch_subprocess(self, info: AsyncSubprocessRunInfo) -> None:
        """Watch a subprocess for output and handle it."""
        if not info.process:
            return
        try:
            await info.process.wait()
            self.debug(
                lambda: f"Service {info.service_type} subprocess stopped gracefully (pid: {info.process.pid})"
            )
        except Exception as e:
            self.warning(f"Error watching subprocess {info.service_id}: {e}")

    async def _wait_for_subprocess(self, info: AsyncSubprocessRunInfo) -> None:
        """Wait for a subprocess to terminate with timeout handling."""
        if not info.process or info.process.returncode is not None:
            return

        try:
            info.process.terminate()

            try:
                await asyncio.wait_for(
                    info.process.wait(), timeout=TASK_CANCEL_TIMEOUT_SHORT
                )
                self.debug(
                    lambda: f"Service {info.service_type} subprocess stopped gracefully (pid: {info.process.pid})"
                )
            except asyncio.TimeoutError:
                self.warning(
                    f"Service {info.service_type} subprocess (pid: {info.process.pid}) did not terminate gracefully, killing"
                )
                info.process.kill()
                await info.process.wait()

        except ProcessLookupError:
            self.debug(
                f"Service {info.service_type} subprocess (pid: {info.process.pid}) already terminated"
            )

    async def wait_for_all_services_start(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = DEFAULT_SERVICE_START_TIMEOUT,
    ) -> None:
        """Wait for all required services to be started."""
        self.debug("Waiting for all required services to start...")
        self.warning(
            "Waiting for all required services to start is not implemented for subprocess management"
        )
