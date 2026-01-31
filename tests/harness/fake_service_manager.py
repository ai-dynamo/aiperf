# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process fake service manager for component testing.

Runs all services in the current process/event loop instead of spawning
subprocesses, enabling fast isolated testing of the full service mesh.
"""

import asyncio
import sys
import uuid

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import LifecycleState, ServiceRegistrationStatus
from aiperf.common.environment import Environment
from aiperf.common.exceptions import AIPerfError
from aiperf.common.models import ServiceRunInfo
from aiperf.common.protocols import ServiceProtocol
from aiperf.common.types import ServiceTypeT
from aiperf.controller.base_service_manager import BaseServiceManager
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceRunType
from tests.harness.fake_communication import FakeCommunication


class FakeServiceManager(BaseServiceManager):
    """In-process service manager replacing multiprocessing (test double: Fake).

    Instead of spawning subprocesses, creates service instances directly in the
    current event loop. Combined with FakeCommunication, enables fast isolated
    testing of the full service mesh without process or network overhead.
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        service_config: ServiceConfig,
        user_config: UserConfig,
        **kwargs,
    ):
        super().__init__(required_services, service_config, user_config, **kwargs)
        self.services: dict[str, ServiceProtocol] = {}
        self.warning(
            "*** Using FakeServiceManager in-process mode to bypass multiprocessing. This is for component integration testing only. ***"
        )

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        """Run a service with the given number of replicas in the current process."""
        ServiceClass = plugins.get_class(PluginType.SERVICE, service_type)

        for _ in range(num_replicas):
            service_id = f"{service_type}_{uuid.uuid4().hex[:8]}"

            # Deep copy configs to simulate separate process behavior
            # (in production each process deserializes its own copy)
            service = ServiceClass(
                service_config=self.service_config.model_copy(deep=True),
                user_config=self.user_config.model_copy(deep=True),
                service_id=service_id,
            )

            await service.initialize()
            await service.start()

            async def watch_service_stopped(service: ServiceProtocol) -> None:
                await service.stopped_event.wait()
                self.info(f"Service {service.service_id} stopped")
                self.services.pop(service.service_id, None)

            self.execute_async(watch_service_stopped(service))
            self.services[service.service_id] = service

            # Track in service maps
            info = ServiceRunInfo(
                service_type=service_type,
                service_id=service_id,
                registration_status=ServiceRegistrationStatus.REGISTERED,
            )
            self.service_map.setdefault(service_type, []).append(info)
            self.service_id_map[service_id] = info

            self.debug(f"Service {service_type} started in-process (id: {service_id})")

    async def stop_service(
        self, service_type: ServiceTypeT, service_id: str | None = None
    ) -> list[BaseException | None]:
        """Stop services matching the given type and optional id."""
        self.debug(f"Stopping {service_type} service(s) with id: {service_id}")
        results: list[BaseException | None] = []

        for service in self.services.values():
            if service.service_type == service_type and (
                service_id is None or service.service_id == service_id
            ):
                try:
                    await service.stop()
                    results.append(None)
                except Exception as e:
                    self.error(f"Error stopping service {service.service_id}: {e!r}")
                    results.append(e)
                finally:
                    # Always remove from tracking, regardless of stop success
                    self.services.pop(service.service_id, None)
                    if service.service_id in self.service_id_map:
                        del self.service_id_map[service.service_id]
                    if service_type in self.service_map:
                        self.service_map[service_type] = [
                            info
                            for info in self.service_map[service_type]
                            if info.service_id != service.service_id
                        ]

        return results

    async def shutdown_all_services(self) -> list[BaseException | None]:
        """Stop all services gracefully."""
        self.debug("Stopping all in-process services")
        results = await asyncio.gather(
            *[
                self._stop_service_gracefully(service)
                for service in self.services.values()
            ],
            return_exceptions=True,
        )
        # Clear all tracking state after shutdown
        self.services.clear()
        self.service_map.clear()
        self.service_id_map.clear()
        # Clean up shared bus
        FakeCommunication.clear_shared_bus()
        return results

    async def kill_all_services(self) -> list[BaseException | None]:
        """Kill all services (for in-process, same as shutdown)."""
        self.debug("Killing all in-process services")
        # For in-process, kill = stop (no process to kill)
        return await self.shutdown_all_services()

    async def wait_for_all_services_registration(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = Environment.SERVICE.REGISTRATION_TIMEOUT,
    ) -> None:
        """Wait for all required services to be registered.

        For in-process mode, services are already registered by the time
        run_service returns, so this is essentially a no-op that validates
        all expected services are present.
        """
        self.debug("Checking all required services are registered (in-process)...")

        required_types = set(self.required_services.keys())
        registered_types = {
            service_info.service_type
            for service_info in self.service_id_map.values()
            if service_info.registration_status == ServiceRegistrationStatus.REGISTERED
        }

        if not required_types.issubset(registered_types):
            missing = required_types - registered_types
            raise AIPerfError(f"Services not registered: {missing}")

    async def wait_for_all_services_start(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = Environment.SERVICE.START_TIMEOUT,
    ) -> None:
        """Wait for all required services to be started.

        For in-process mode, services are already started by the time
        run_service returns. This validates all services are in RUNNING state.
        """
        self.debug("Checking all required services are started (in-process)...")

        for service in self.services.values():
            if service.state != LifecycleState.RUNNING:
                raise AIPerfError(
                    f"Service {service.service_id} is not running: {service.state}"
                )

    async def _stop_service_gracefully(
        self, service: ServiceProtocol
    ) -> BaseException | None:
        """Stop a single service gracefully."""
        try:
            await service.stop()
            self.debug(f"Service {service.service_id} stopped")
            return None
        except Exception as e:
            self.error(f"Error stopping service {service.service_id}: {e!r}")
            return e


# =============================================================================
# Plugin Registration - Hot-swap production implementations when imported
# =============================================================================

# Register FakeServiceManager for multiprocessing run type at max priority
plugins.register(
    PluginType.SERVICE_MANAGER,
    ServiceRunType.MULTIPROCESSING,
    FakeServiceManager,
    priority=sys.maxsize,
)
