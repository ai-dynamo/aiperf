# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
from abc import ABC, abstractmethod

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import ServiceType
from aiperf.common.models import ServiceRunInfo


class BaseServiceManager(ABC):
    """
    Base class for service managers. It provides a common interface for
    managing services and a way to look up service information by service ID.
    """

    def __init__(
        self, required_service_types: list[ServiceType], config: ServiceConfig
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.required_service_types = required_service_types
        self.config = config

        # Maps to track service information
        self.service_map: dict[ServiceType, list[ServiceRunInfo]] = {}

        # Create service ID map for component lookups
        self.service_id_map: dict[str, ServiceRunInfo] = {}

    @abstractmethod
    async def run_all_services(self) -> None:
        """Run all required services."""
        pass

    @abstractmethod
    async def shutdown_all_services(self) -> None:
        """Shutdown all managed services."""
        pass

    @abstractmethod
    async def kill_all_services(self) -> None:
        """Kill all managed services."""
        pass

    @abstractmethod
    async def wait_for_all_services_registration(
        self, stop_event: asyncio.Event, timeout_seconds: int = 30
    ) -> None:
        """Wait for all required services to be registered."""
        pass
