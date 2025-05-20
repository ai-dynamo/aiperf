#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import asyncio
import logging
from abc import ABC, abstractmethod

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import ServiceType
from aiperf.common.models.service_models import ServiceRunInfo


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

    def get(self, service_id: str) -> ServiceRunInfo | None:
        """
        Get the service run information by service ID.
        """
        return self.service_id_map.get(service_id)

    @abstractmethod
    async def initialize_all_services(self) -> None:
        """Initialize all required services."""
        pass

    @abstractmethod
    async def stop_all_services(self) -> None:
        """Stop all managed services."""
        pass

    @abstractmethod
    async def wait_for_all_services_registration(
        self, stop_event: asyncio.Event, timeout_seconds: int = 30
    ) -> None:
        """Wait for all required services to be registered."""
        pass

    @abstractmethod
    async def wait_for_all_services_start(self) -> None:
        """Wait for all required services to be started."""
        pass
