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
from collections.abc import Callable, Coroutine
from typing import Any

from aiperf.common.enums import ClientType, TopicType
from aiperf.common.models.message_models import Message


class BaseCommunication(ABC):
    """Base class for communication between AIPerf components."""

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
        """Gracefully shutdown communication channels.

        Raises:
            Exception object if an exception occurred, or None if shutdown was successful
        """
        pass

    @abstractmethod
    async def create_clients(self, *client_types: ClientType) -> None:
        """Create the communication clients."""
        pass

    @abstractmethod
    async def publish(self, topic: TopicType, message: Message) -> None:
        """Publish a response to a topic.

        Args:
            topic: Topic to publish to
            message: Message to publish (must be a Pydantic model)

        Raises:
            Exception object if an exception occurred, or None if response was published successfully
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
            callback: Function to call when a response is received (receives Message object)
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
            request_data: Request data (must be a Message instance)
            timeout: Timeout in seconds

        Returns:
            Response message (Message instance) if successful, or Exception object if an exception occurred
        """
        pass

    @abstractmethod
    async def respond(self, target: str, response: Message) -> None:
        """Send a response to a request.

        Args:
            target: Target component to send response to
            response: Response message (must be a Message instance)

        Raises:
            Exception object if an exception occurred, or None if response was sent successfully
        """
        pass

    @abstractmethod
    async def push(self, topic: TopicType, message: Message) -> None:
        """Push data to a target.

        Args:
            topic: Topic to push to (must be a TopicType instance)
            message: Message to be pushed (must be a Message instance)

        Raises:
            Exception object if an exception occurred, or None if data was pushed successfully
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
            topic: Topic to pull from (must be a TopicType instance)
            callback: function to call when data is received. (receives Message object)

        Raises:
            Exception object if an exception occurred, or None if pull registration was successful.
        """
        pass
