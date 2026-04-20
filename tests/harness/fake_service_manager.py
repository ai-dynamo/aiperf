# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process fake service manager for component testing.

Runs all services in the current process/event loop instead of spawning
subprocesses, enabling fast isolated testing of the full service mesh.
"""

import asyncio
import os
import sys
import uuid

from aiperf.common.environment import Environment
from aiperf.common.protocols import ServiceProtocol
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.types import ServiceTypeT
from aiperf.config import BenchmarkRun
from aiperf.controller.base_service_manager import BaseServiceManager
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceRunType, ServiceType
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
        run: BenchmarkRun,
        **kwargs,
    ):
        super().__init__(required_services, run=run, **kwargs)
        self.services: dict[str, ServiceProtocol] = {}

        # Disable health server for in-process services to prevent port conflicts.
        # Multiple services in the same process would all try to bind the same port.
        Environment.SERVICE.HEALTH_ENABLED = False

        # Signal to workers and other services that the WorkerGroupManager
        # boundary is simulated (all services share one process and one
        # FakeCommunication bus), so they should use the direct control path
        # instead of waiting on pod-lifecycle channels that are never wired.
        os.environ["AIPERF_FAKE_IN_PROCESS_MODE"] = "1"

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

            # Deep copy run to simulate separate process behavior
            # (in production each process deserializes its own copy)
            service = ServiceClass(
                run=self.run.model_copy(deep=True),
                service_id=service_id,
            )

            # Set expectation before initialize/start so the service can't
            # register via the message bus before the expectation exists.
            ServiceRegistry.expect_service(service_id, service_type)

            await service.initialize()
            await service.start()

            async def watch_service_stopped(service: ServiceProtocol) -> None:
                await service.stopped_event.wait()
                self.info(f"Service {service.service_id} stopped")
                self.services.pop(service.service_id, None)

            self.execute_async(watch_service_stopped(service))
            self.services[service.service_id] = service

            self.debug(f"Service {service_type} started in-process (id: {service_id})")

        # In real multiprocessing mode, WorkerGroupManager spawns Worker and
        # RecordProcessor subprocesses under itself. The in-process fake
        # service manager bypasses that boundary: spawn them directly as peer
        # services so the controller can drive them over FakeCommunication.
        if service_type == ServiceType.WORKER_GROUP_MANAGER:
            worker_count = self.run.cfg.worker_group_declared_worker_capacity
            rp_count = self.run.cfg.worker_group_declared_record_processor_capacity
            if worker_count > 0:
                await self.run_service(ServiceType.WORKER, worker_count)
            if rp_count > 0:
                await self.run_service(ServiceType.RECORD_PROCESSOR, rp_count)

    async def stop_service(
        self, service_type: ServiceTypeT, service_id: str | None = None
    ) -> list[BaseException | None]:
        """Stop services matching the given type and optional id."""
        self.debug(f"Stopping {service_type} service(s) with id: {service_id}")
        results: list[BaseException | None] = []

        for service in list(self.services.values()):
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
                    self.services.pop(service.service_id, None)
                    ServiceRegistry.forget(service.service_id)

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
        self.services.clear()
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
        timeout_seconds: float = Environment.SERVICE.REGISTRATION_TIMEOUT,
    ) -> None:
        """Wait for all required services to be registered.

        For in-process mode, services are already registered by the time
        run_service returns, so this delegates to ServiceRegistry.
        """
        self.debug("Checking all required services are registered (in-process)...")
        await ServiceRegistry.wait_for_all(timeout_seconds)

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
