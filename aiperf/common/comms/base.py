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
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

from aiperf.common.comms.client_enums import ClientType
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import CommunicationBackend, TopicType
from aiperf.common.exceptions import (
    CommunicationCreateError,
    CommunicationTypeAlreadyRegisteredError,
    CommunicationTypeUnknownError,
)
from aiperf.common.models import (
    Message,
    ZMQCommunicationConfig,
    ZMQTCPTransportConfig,
)

logger = logging.getLogger(__name__)


################################################################################
# Base Communication Class
################################################################################


class BaseCommunication(ABC):
    """Base class for specifying the base communication layer for AIPerf components."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize communication channels."""
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if communication channels are initialized.

        Returns:
            True if communication channels are initialized, False otherwise
        """
        pass

    @property
    @abstractmethod
    def is_shutdown(self) -> bool:
        """Check if communication channels are shutdown.

        Returns:
            True if communication channels are shutdown, False otherwise
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown communication channels."""
        pass

    @abstractmethod
    async def create_clients(self, *client_types: ClientType) -> None:
        """Create the communication clients.

        Args:
            *client_types: The client types to create
        """
        pass

    @abstractmethod
    async def publish(self, topic: TopicType, message: Message) -> None:
        """Publish a response to a topic.

        Args:
            topic: Topic to publish to
            message: Message to publish
        """
        pass

    @abstractmethod
    async def subscribe(
        self,
        topic: TopicType,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to a topic.

        Args:
            topic: Topic to subscribe to
            callback: Function to call when a response is received
        """
        pass

    @abstractmethod
    async def request(
        self,
        target: str,
        request_data: Message,
        timeout: float = 5.0,
    ) -> Message:
        """Send a request and wait for a response.

        Args:
            target: Target component to send request to
            request_data: Request data
            timeout: Timeout in seconds

        Returns:
            Response message if successful
        """
        pass

    @abstractmethod
    async def respond(self, target: str, response: Message) -> None:
        """Send a response to a request.

        Args:
            target: Target component to send response to
            response: Response message
        """
        pass

    @abstractmethod
    async def push(self, topic: TopicType, message: Message) -> None:
        """Push data to a target.

        Args:
            topic: Topic to push to
            message: Message to be pushed
        """
        pass

    @abstractmethod
    async def pull(
        self,
        topic: TopicType,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Pull data from a source.

        Args:
            topic: Topic to pull from
            callback: function to call when data is received.
        """
        pass


################################################################################
# Communication Factory
################################################################################


class CommunicationFactory:
    """Factory for creating communication instances. Provides a registry of communication types and
    methods for registering new communication types and creating communication instances from existing
    registered types.
    """

    # Registry of communication types
    _comm_registry: dict[CommunicationBackend | str, type[BaseCommunication]] = {}

    @classmethod
    def register(cls, comm_type: CommunicationBackend | str) -> Callable:
        """Register a new communication type.

        Args:
            comm_type: String representation of the communication type

        Returns:
            Decorator for the communication class

        Raises:
            CommunicationTypeAlreadyRegisteredError: If the communication type is already registered
        """

        def decorator(comm_cls: type[BaseCommunication]) -> type[BaseCommunication]:
            if comm_type in cls._comm_registry:
                raise CommunicationTypeAlreadyRegisteredError(
                    f"Communication type {comm_type} already registered"
                )
            cls._comm_registry[comm_type] = comm_cls
            return comm_cls

        return decorator

    @classmethod
    def create_communication(
        cls, service_config: ServiceConfig, **kwargs
    ) -> BaseCommunication:
        """Create a communication instance.

        Args:
            service_config: Service configuration containing the communication type
            **kwargs: Additional arguments for the communication class

        Returns:
            BaseCommunication instance

        Raises:
            CommunicationTypeUnknownError: If the communication type is not registered
            CommunicationCreateError: If there was an error creating the communication instance
        """
        if service_config.comm_backend not in cls._comm_registry:
            raise CommunicationTypeUnknownError(
                f"Unknown communication type: {service_config.comm_backend}"
            )

        try:
            comm_class = cls._comm_registry[service_config.comm_backend]
            config = kwargs.get("config") or ZMQCommunicationConfig(
                protocol_config=ZMQTCPTransportConfig()
            )
            kwargs["config"] = config

            return comm_class(**kwargs)
        except Exception as e:
            raise CommunicationCreateError from e
