# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import tempfile
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.config.zmq_config import ZMQIPCConfig, ZMQTCPConfig
from aiperf.common.constants import (
    DEFAULT_SERVICE_REGISTRATION_TIMEOUT,
    DEFAULT_SERVICE_START_TIMEOUT,
    MILLIS_PER_SECOND,
    TASK_CANCEL_TIMEOUT_SHORT,
)
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ServiceRegistrationStatus, ServiceRunType, ServiceType
from aiperf.common.exceptions import AIPerfError
from aiperf.common.factories import ServiceManagerFactory
from aiperf.common.logging import handle_subprocess_log_line
from aiperf.common.messages import ServiceFailedMessage
from aiperf.common.mixins import MessageBusClientMixin
from aiperf.common.models.error_models import ErrorDetails
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
    user_config_file: Path | None = Field(
        default=None,
        description="Path to the temporary user config file",
    )
    service_config_file: Path | None = Field(
        default=None,
        description="Path to the temporary service config file",
    )


@implements_protocol(ServiceManagerProtocol)
@ServiceManagerFactory.register(ServiceRunType.MULTIPROCESSING)
class MultiProcessServiceManager(BaseServiceManager, MessageBusClientMixin):
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
        super().__init__(
            required_services,
            service_config,
            user_config,
            id="multiprocess_service_manager",
            **kwargs,
        )
        self.subprocess_info_map: dict[str, AsyncSubprocessRunInfo] = {}
        self.subprocess_map_lock = asyncio.Lock()

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
        info = AsyncSubprocessRunInfo(
            service_type=service_type,
            service_id=service_id,
        )
        async with self.subprocess_map_lock:
            self.subprocess_info_map[service_id] = info

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix=f"aiperf_user_config_{service_id}_",
                delete=False,
            ) as user_config_file:
                user_config_file.write(user_config_json)
                info.user_config_file = Path(user_config_file.name)

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix=f"aiperf_service_config_{service_id}_",
                delete=False,
            ) as service_config_file:
                service_config_file.write(service_config_json)
                info.service_config_file = Path(service_config_file.name)

            args = [
                "aiperf",
                "service",
                service_type,
                "--user-config-file",
                str(info.user_config_file),
                "--service-config-file",
                str(info.service_config_file),
                "--service-id",
                service_id,
            ]

            self.debug(
                lambda args=args: f"Starting subprocess for {service_type} with command: {' '.join(args[:3])} ..."
            )

            info.process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=current_dir,
            )

            self.info(
                f"Service {service_id} started as subprocess (PID: {info.process.pid})"
            )

            self.execute_async(self._watch_subprocess(info))
            self.execute_async(self._handle_subprocess_output(info.process, service_id))
            await yield_to_event_loop()

        except Exception:
            if info.user_config_file and info.user_config_file.exists():
                info.user_config_file.unlink(missing_ok=True)
            if info.service_config_file and info.service_config_file.exists():
                info.service_config_file.unlink(missing_ok=True)
            raise

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        """Run a service with the given number of replicas."""
        tasks = []

        if isinstance(self.service_config.comm_config, ZMQTCPConfig):
            self.service_config.zmq_tcp = self.service_config.comm_config
        elif isinstance(self.service_config.comm_config, ZMQIPCConfig):
            self.service_config.zmq_ipc = self.service_config.comm_config
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

        # Ensure structured logging is enabled for subprocesses
        env["AIPERF_STRUCTURED_LOGGING"] = "true"

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
        for info in self.subprocess_info_map.values():
            if info.service_type == service_type and (
                service_id is None or info.service_id == service_id
            ):
                task = asyncio.create_task(self._wait_for_subprocess(info))
                task.add_done_callback(
                    lambda _, info=info: self.execute_async(
                        self._remove_subprocess_info(info)
                    )
                )
                tasks.append(task)
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _remove_subprocess_info(self, info: AsyncSubprocessRunInfo) -> None:
        async with self.subprocess_map_lock:
            self.subprocess_info_map.pop(info.service_id)

    async def shutdown_all_services(self) -> list[BaseException | None]:
        """Stop all required services as asyncio subprocesses."""
        self.debug("Stopping all service subprocesses")

        # Wait for all to finish in parallel
        return await asyncio.gather(
            *[
                self._wait_for_subprocess(info)
                for info in self.subprocess_info_map.values()
            ],
            return_exceptions=True,
        )

    async def kill_all_services(self) -> list[BaseException | None]:
        """Kill all required services as asyncio subprocesses."""
        self.debug("Killing all service subprocesses")

        # Kill all subprocesses
        async with self.subprocess_map_lock:
            for info in self.subprocess_info_map.values():
                if info.process and info.process.returncode is None:
                    info.process.kill()

        # Wait for all to finish in parallel
        return await asyncio.gather(
            *[
                self._wait_for_subprocess(info)
                for info in self.subprocess_info_map.values()
            ],
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

        await asyncio.sleep(2)

        # TODO: Can this be done better by using asyncio.Event()?

        async def _wait_for_registration():
            while not stop_event.is_set():
                # Get all registered service types from the id map
                missing_required_services = False
                for service_info in self.service_id_map.values():
                    process = self.subprocess_info_map.get(service_info.service_id)
                    if (
                        not process
                        or service_info.registration_status
                        != ServiceRegistrationStatus.REGISTERED
                        and service_info.required
                    ):
                        missing_required_services = True
                        break

                if missing_required_services:
                    self.debug("Waiting for required services to register...")
                    await asyncio.sleep(0.5)
                else:
                    return

        try:
            await asyncio.wait_for(_wait_for_registration(), timeout=timeout_seconds)
        except asyncio.TimeoutError as e:
            # Log which services didn't register in time
            _ = set(
                service_info.service_type
                for service_info in self.service_id_map.values()
                if service_info.registration_status
                == ServiceRegistrationStatus.REGISTERED
            )

            # for service_type in required_types:
            #     if service_type not in registered_types_set:
            #         self.error(
            #             f"Service {service_type} failed to register within timeout"
            #         )

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
                    chunk = await stream.read(1024)  # Read up to 1KB at a time
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
                        # Wait for 1 millisecond to avoid reading too frequently from the subprocess
                        await asyncio.sleep(1 / MILLIS_PER_SECOND)
                    else:
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

    async def _watch_subprocess(self, info: AsyncSubprocessRunInfo) -> None:
        """Watch a subprocess for output and handle it."""
        if not info.process:
            self.warning(f"Subprocess {info.service_id} not found")
            return
        try:
            await info.process.wait()
            self.warning(
                f"Service {info.service_id} subprocess stopped gracefully (pid: {info.process.pid}) (return code: {info.process.returncode})"
            )
        except Exception as e:
            self.warning(f"Error watching subprocess {info.service_id}: {e}")
        finally:
            if not self.stop_requested:
                await self.publish(
                    ServiceFailedMessage(
                        service_id=info.service_id,
                        error=ErrorDetails(
                            message=f"Service {info.service_id} subprocess exited with code: {info.process.returncode}"
                        ),
                        target_service_type=ServiceType.SYSTEM_CONTROLLER,
                    )
                )
            self.warning(
                f"Service {info.service_id} subprocess stopped (pid: {info.process.pid}) (return code: {info.process.returncode})"
            )
            if info.process.returncode:
                self.warning(
                    f"Service {info.service_id} subprocess exited with code: {info.process.returncode}"
                )

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
                    f"Service {info.service_id} subprocess (pid: {info.process.pid}) (return code: {info.process.returncode}) stopped"
                )
            except asyncio.TimeoutError:
                self.warning(
                    f"Service {info.service_id} subprocess (pid: {info.process.pid}) (return code: {info.process.returncode}) did not terminate gracefully, killing"
                )
                info.process.kill()
                await info.process.wait()

        except ProcessLookupError:
            self.debug(
                f"Service {info.service_id} subprocess (pid: {info.process.pid}) (return code: {info.process.returncode}) already terminated"
            )
        finally:
            if info.user_config_file and info.user_config_file.exists():
                try:
                    info.user_config_file.unlink()
                except Exception as e:
                    self.warning(f"Failed to delete user config file: {e}")
            if info.service_config_file and info.service_config_file.exists():
                try:
                    info.service_config_file.unlink()
                except Exception as e:
                    self.warning(f"Failed to delete service config file: {e}")

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
