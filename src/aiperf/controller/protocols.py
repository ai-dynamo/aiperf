# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.environment import Environment
from aiperf.common.protocols import AIPerfLifecycleProtocol

if TYPE_CHECKING:
    import multiprocessing

    from aiperf.common.models import ServiceRunInfo
    from aiperf.common.types import ServiceTypeT
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.controller.multiprocess_service_manager import MultiProcessRunInfo


@runtime_checkable
class ServiceManagerProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for a service manager that manages the running of services using the specific ServiceRunType.
    Abstracts away the details of service deployment and management.
    see :class:`aiperf.controller.base_service_manager.BaseServiceManager` for more details.
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        run: BenchmarkRun,
        log_queue: multiprocessing.Queue | None = None,
    ): ...

    required_services: dict[ServiceTypeT, int]
    service_map: dict[ServiceTypeT, list[ServiceRunInfo]]
    service_id_map: dict[str, ServiceRunInfo]

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None: ...

    async def run_services(self, service_types: dict[ServiceTypeT, int]) -> None: ...
    async def run_required_services(self) -> None: ...
    async def shutdown_all_services(self) -> list[BaseException | None]: ...
    async def kill_all_services(self) -> list[BaseException | None]: ...
    async def stop_service(
        self, service_type: ServiceTypeT, service_id: str | None = None
    ) -> list[BaseException | None]: ...
    async def stop_services_by_type(
        self, service_types: list[ServiceTypeT]
    ) -> list[BaseException | None]: ...
    async def wait_for_all_services_registration(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = Environment.SERVICE.REGISTRATION_TIMEOUT,
    ) -> None: ...

    async def wait_for_all_services_start(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = Environment.SERVICE.START_TIMEOUT,
    ) -> None: ...

    def activate_heartbeat_monitoring(self) -> None:
        """Begin failing services that stop heartbeating."""
        ...

    def notify_shutdown(self) -> None:
        """Stop treating service exits as failures once teardown has begun."""
        ...


@runtime_checkable
class LocalProcessServiceManagerProtocol(ServiceManagerProtocol, Protocol):
    """Capability exposed by managers that own local service processes."""

    multi_process_info: list[MultiProcessRunInfo]


@runtime_checkable
class KubernetesServiceManagerProtocol(ServiceManagerProtocol, Protocol):
    """Kubernetes-only pod-health capability used by the controller."""

    pod_failure_abort_event: asyncio.Event
    """Set when failed worker pods breach the configured abort threshold."""
    pod_failure_abort_reason: str
    """Human-readable explanation for a set ``pod_failure_abort_event``."""

    async def check_pods_healthy(self) -> None:
        """Verify all tracked worker pods are healthy before profiling starts."""
        ...

    def get_pod_summary(self) -> dict[str, str]:
        """Map worker-pod index to a human-readable status string."""
        ...
