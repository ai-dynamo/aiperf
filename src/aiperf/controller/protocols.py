# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.environment import Environment
from aiperf.common.protocols import AIPerfLifecycleProtocol

if TYPE_CHECKING:
    import multiprocessing

    from aiperf.common.config import ServiceConfig, UserConfig
    from aiperf.common.models import ServiceRunInfo
    from aiperf.common.types import ServiceTypeT


@runtime_checkable
class ServiceManagerProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for a service manager that manages the running of services using the specific ServiceRunType.
    Abstracts away the details of service deployment and management.
    see :class:`aiperf.controller.base_service_manager.BaseServiceManager` for more details.
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        service_config: ServiceConfig,
        user_config: UserConfig,
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
