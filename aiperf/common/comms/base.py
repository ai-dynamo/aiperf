# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

from aiperf.common.comms.client_enums import ClientType
from aiperf.common.enums import TopicType
from aiperf.common.models import Message

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
