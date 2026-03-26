# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_start, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.types import ServiceTypeT

if TYPE_CHECKING:
    from aiperf.common.models.service_models import ServiceRunInfo
    from aiperf.config import BenchmarkRun


class BaseServiceManager(AIPerfLifecycleMixin, ABC):
    """
    Base class for service managers. It provides a common interface for managing services.
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        run: BenchmarkRun,
        **kwargs,
    ):
        super().__init__(run=run, **kwargs)
        self.required_services = required_services
        self.run = run
        self.kwargs = kwargs
        # Maps to track service information
        self.service_map: dict[ServiceTypeT, list[ServiceRunInfo]] = {}
        self._shutdown_complete = False
        self._heartbeat_monitoring_active = False
        self._pod_monitoring_active = False
        self.pod_failure_abort_event = asyncio.Event()
        self.pod_failure_abort_reason: str = ""

    def notify_shutdown(self) -> None:
        """Signal that shutdown has been initiated.

        Suppresses heartbeat and process monitors from reporting expected
        process exits as errors. Called by the system controller before
        broadcasting the shutdown command.
        """
        self._shutdown_complete = True

    def activate_pod_monitoring(self) -> None:
        """Enable Kubernetes pod health monitoring.

        Called by the system controller after spawning services, before waiting
        for registration/configuration. Unlike heartbeat monitoring, pod phase
        checks are safe during startup — a pod in Failed/Unknown state is always
        an error, regardless of whether services have registered yet. This allows
        fast failure detection during the registration/configuration phase.
        """
        self._pod_monitoring_active = True

    def activate_heartbeat_monitoring(self) -> None:
        """Enable heartbeat-based stale service detection.

        Called by the system controller after all services have registered.
        Prevents false positives during the startup/registration phase when
        services may not yet be sending regular heartbeats.
        """
        self._heartbeat_monitoring_active = True

    @on_start
    async def _start_service_manager(self) -> None:
        await self.run_required_services()

    @on_stop
    async def _stop_service_manager(self) -> None:
        await self.shutdown_all_services()

    async def run_services(
        self, service_types: dict[ServiceTypeT, int]
    ) -> list[BaseException | None]:
        return await asyncio.gather(
            *[
                self.run_service(service_type, num_replicas)
                for service_type, num_replicas in service_types.items()
            ],
            return_exceptions=True,
        )

    @abstractmethod
    async def stop_service(
        self, service_type: ServiceTypeT, service_id: str | None = None
    ) -> list[BaseException | None]: ...

    async def stop_services_by_type(
        self, service_types: list[ServiceTypeT]
    ) -> list[BaseException | None]:
        """Stop a set of services."""
        results = await asyncio.gather(
            *[self.stop_service(service_type) for service_type in service_types],
            return_exceptions=True,
        )
        output: list[BaseException | None] = []
        for result in results:
            if isinstance(result, list):
                output.extend(result)
            else:
                output.append(result)
        return output

    async def run_required_services(self) -> None:
        results = await self.run_services(self.required_services)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            for error in errors:
                self.exception(f"Error starting required service: {error!r}")
            raise errors[0]

    @abstractmethod
    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        pass

    @abstractmethod
    async def shutdown_all_services(self) -> list[BaseException | None]:
        pass

    @abstractmethod
    async def kill_all_services(self) -> list[BaseException | None]:
        pass

    @abstractmethod
    async def wait_for_all_services_registration(
        self,
        timeout_seconds: float = Environment.SERVICE.REGISTRATION_TIMEOUT,
    ) -> None:
        pass

    @background_task(
        interval=lambda self: Environment.SERVICE.HEARTBEAT_INTERVAL,
        immediate=False,
    )
    async def _monitor_heartbeats(self) -> None:
        """Detect registered services that have stopped sending heartbeats.

        Marks stale services as failed via ServiceRegistry.fail_service,
        which wakes all pending waiters.
        """
        if (
            self._shutdown_complete
            or self.stop_requested
            or not self._heartbeat_monitoring_active
        ):
            return
        threshold_sec = (
            Environment.SERVICE.HEARTBEAT_INTERVAL
            * Environment.SERVICE.HEARTBEAT_MISSED_THRESHOLD
        )
        stale = ServiceRegistry.get_stale_services(threshold_sec)
        for info in stale:
            self.warning(
                f"Service '{info.service_id}' ({info.service_type}) "
                f"missed heartbeats — marking as failed"
            )
            ServiceRegistry.fail_service(info.service_id, info.service_type)

    async def wait_for_api_subprocess(self) -> None:
        """Block until the API service runtime terminates (Kubernetes mode only).

        Default implementation is a no-op. Override in Kubernetes service
        managers that keep a local API runtime alive after benchmarking.
        """
        pass

    def get_pod_summary(self) -> dict[str, str]:
        """Get pod state summary for diagnostics (Kubernetes mode only).

        Default implementation returns an empty dict. Override in
        KubernetesServiceManager to return actual pod states.
        """
        return {}

    async def check_pods_healthy(self) -> None:
        """Verify all tracked pods are healthy before starting profiling.

        Default implementation is a no-op. Override in KubernetesServiceManager
        to check pod phases and fail fast if any pods are in a terminal state.
        """
