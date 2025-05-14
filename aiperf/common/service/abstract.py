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
from abc import ABC, abstractmethod

from aiperf.common.enums import (
    ClientType,
    ServiceType,
)


class AbstractBaseService(ABC):
    """Abstract base class for all services.

    This class provides the base foundation for which every service should provide.
    """

    @property
    @abstractmethod
    def required_clients(self) -> list[ClientType]:
        """The communication clients required by the service.

        This property should be implemented by derived classes to specify the
        communication clients that the service requires."""
        pass

    @property
    @abstractmethod
    def service_type(self) -> ServiceType:
        """The type of service.

        This property should be implemented by derived classes to specify the
        type of service."""
        pass

    @abstractmethod
    async def run(self) -> None:
        """Run the service.

        This method should be implemented by derived classes to run the main loop of the service.
        """
        pass

    @abstractmethod
    async def _initialize(self) -> None:
        """Initialize service-specific components.

        This method should be implemented by derived classes to set up any resources
        specific to that service.
        """

    @abstractmethod
    async def _on_start(self) -> None:
        """Start the service.

        This method should be implemented by derived classes to run any processes
        or components specific to that service.
        """

    @abstractmethod
    async def _on_stop(self) -> None:
        """Stop the service.

        This method should be implemented by derived classes to stop any processes
        or components specific to that service.
        """

    @abstractmethod
    async def _cleanup(self) -> None:
        """Clean up service-specific components.

        This method should be implemented by derived classes to free any resources
        allocated by the service.
        """
