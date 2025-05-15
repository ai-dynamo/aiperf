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

    This class provides the base foundation for which every service should provide. Some
    methods are required to be implemented by derived classes, while others are
    meant to be implemented by the base class.
    """

    @property
    @abstractmethod
    def required_clients(self) -> list[ClientType]:
        """The communication clients required by the service. If nothing is returned,
        the service will be responsible for creating its own clients.

        This property should be implemented by derived classes to specify the
        communication clients that the service requires."""
        pass

    @property
    @abstractmethod
    def service_type(self) -> ServiceType:
        """The type/name of the service.

        This property should be implemented by derived classes to specify the
        type/name of the service."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service.

        This method will be implemented by the base class.
        """
        pass

    @abstractmethod
    async def run(self) -> None:
        """Run the service. This method will be the primary entry point for the service
        and will be called by the bootstrap script. It should not return until the service
        is completely shutdown.

        This method will be implemented by the base class.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the service.

        This method will be implemented by the base class.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the service. It should be called after the service has been initialized
        and configured.

        This method will be implemented by the base class.
        """
        pass

    @abstractmethod
    async def _initialize(self) -> None:
        """Called by the base class when the service is initializing to allow the
        derived service to set up any resources specific to that service.
        """
        pass

    @abstractmethod
    async def _on_start(self) -> None:
        """Called by the base class when the service is started to allow the
        derived service to run any processes or components specific to that service.
        """
        pass

    @abstractmethod
    async def _on_stop(self) -> None:
        """Called by the base class when the service is stopping to allow the
        derived service to stop any processes or components specific to that service.
        """
        pass

    @abstractmethod
    async def _cleanup(self) -> None:
        """Called by the base class after the service is stopped to allow the
        derived service to free any resources allocated by the service.
        """
        pass
