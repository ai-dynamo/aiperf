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
from aiperf.common.enums.service import ServiceState
from aiperf.common.models.message import Message


class BaseServiceInterface(ABC):
    """Base interface for all services.

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
        # TODO: We can do this better by using a decorator to set the service type
        pass

    @abstractmethod
    async def set_state(self, state: ServiceState) -> None:
        """Set the state of the service.

        This method will be implemented by the base class, and extra
        functionality can be added by derived classes via the `@on_set_state`
        decorator.
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service.

        This method will be implemented by the base class.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the service. It should be called after the service has been initialized
        and configured.

        This method will be implemented by the base class, and extra
        functionality can be added by derived classes via the `@on_start`
        decorator.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the service.

        This method will be implemented by the base class, and extra
        functionality can be added by derived classes via the `@on_stop`
        decorator.
        """
        pass

    @abstractmethod
    async def configure(self, message: Message) -> None:
        """Configure the service with the given configuration.

        This method will be implemented by the base class, and extra
        functionality can be added by derived classes via the `@on_configure`
        decorator.
        """
        pass

    @abstractmethod
    async def run_forever(self) -> None:
        """Run the service. This method will be the primary entry point for the service
        and will be called by the bootstrap script. It should not return until the
        service is completely shutdown.

        This method will be implemented by the base class. Any additional
        functionality can be added by derived classes via the `@on_run`
        decorator.
        """
        pass

    @abstractmethod
    async def _forever_loop(self) -> None:
        """Run the service in a loop until the stop event is set. This method will be
        called by the `run` method to allow the service to run indefinitely.

        This method will be implemented by the base class, and is not expected to be
        overridden by derived classes.
        """
        pass
