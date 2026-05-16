# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.environment import Environment
from aiperf.common.protocols import AIPerfLifecycleProtocol

if TYPE_CHECKING:
    from aiperf.common.error_queue import ErrorQueue
    from aiperf.common.logging import LogQueue
    from aiperf.common.types import ServiceTypeT
    from aiperf.config import BenchmarkRun


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
        log_queue: LogQueue | None = None,
        error_queue: ErrorQueue | None = None,
    ): ...

    required_services: dict[ServiceTypeT, int]
    pod_failure_abort_event: asyncio.Event
    pod_failure_abort_reason: str

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None: ...

    async def run_services(self, service_types: dict[ServiceTypeT, int]) -> None: ...
    async def run_required_services(self) -> None: ...
    def notify_shutdown(self) -> None: ...
    def activate_heartbeat_monitoring(self) -> None: ...
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
        timeout_seconds: float = Environment.SERVICE.REGISTRATION_TIMEOUT,
    ) -> None: ...

    async def wait_for_api_subprocess(self) -> None:
        """Block until the API service runtime terminates (Kubernetes mode only)."""
        ...

    def activate_pod_monitoring(self) -> None:
        """Enable Kubernetes pod health monitoring during startup."""
        ...

    def get_pod_summary(self) -> dict[str, str]:
        """Get pod state summary for diagnostics (Kubernetes mode only).

        Returns a dict mapping pod_index to a human-readable status string.
        Empty dict in non-Kubernetes modes.
        """
        ...

    async def check_pods_healthy(self) -> None:
        """Verify all tracked pods are healthy before starting profiling."""
        ...
