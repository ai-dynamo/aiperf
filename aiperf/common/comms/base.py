# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from collections.abc import Callable, Coroutine
from typing import Any, Protocol

from aiperf.common.constants import DEFAULT_COMMS_REQUEST_TIMEOUT
from aiperf.common.enums import (
    CommunicationClientAddressType,
    CommunicationClientType,
    MessageType,
)
from aiperf.common.factories import FactoryMixin
from aiperf.common.messages import Message

logger = logging.getLogger(__name__)


################################################################################
# Base Communication Client Interfaces
################################################################################


class CommunicationClientProtocol(Protocol):
    """Base interface for specifying the base communication client for AIPerf components."""

    async def initialize(self) -> None:
        """Initialize communication channels."""
        ...

    async def shutdown(self) -> None:
        """Shutdown communication channels."""
        ...


class PushClientProtocol(CommunicationClientProtocol):
    """Interface for push clients."""

    async def push(self, message: Message) -> None:
        """Push data to a target. The message will be routed automatically
        based on the message type.

        Args:
            message_type: MessageType to push to
            message: Message to be pushed
        """
        ...


class PullClientProtocol(CommunicationClientProtocol):
    """Interface for pull clients."""

    async def register_pull_callback(
        self,
        message_type: MessageType,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
        max_concurrency: int | None = None,
    ) -> None:
        """Register a callback for a pull client. The callback will be called when
        a message is received for the given message type.

        Args:
            message_type: The message type to register the callback for
            callback: The callback to register
            max_concurrency: The maximum number of concurrent requests to allow.
        """
        ...


class RequestClientProtocol(CommunicationClientProtocol):
    """Interface for request clients."""

    async def request(
        self,
        message: Message,
        timeout: float = DEFAULT_COMMS_REQUEST_TIMEOUT,
    ) -> Message:
        """Send a request and wait for a response. The message will be routed automatically
        based on the message type.

        Args:
            message: Message to send (will be routed based on the message type)
            timeout: Timeout in seconds for the request.

        Returns:
            Response message if successful
        """
        ...

    async def request_async(
        self,
        message: Message,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Send a request and be notified when the response is received. The message will be routed automatically
        based on the message type.

        Args:
            message: Message to send (will be routed based on the message type)
            callback: Callback to be called when the response is received
        """
        ...


class ReplyClientProtocol(CommunicationClientProtocol):
    """Interface for reply clients."""

    def register_request_handler(
        self,
        service_id: str,
        message_type: MessageType,
        handler: Callable[[Message], Coroutine[Any, Any, Message | None]],
    ) -> None:
        """Register a request handler for a message type. The handler will be called when
        a request is received for the given message type.

        Args:
            service_id: The service ID to register the handler for
            message_type: The message type to register the handler for
            handler: The handler to register
        """
        ...


class SubClientProtocol(CommunicationClientProtocol):
    """Interface for subscribe clients."""

    async def subscribe(
        self,
        message_type: MessageType,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to a specific message type. The callback will be called when
        a message is received for the given message type."""
        ...


class PubClientProtocol(CommunicationClientProtocol):
    """Interface for publish clients."""

    async def publish(self, message: Message) -> None:
        """Publish a message. The message will be routed automatically based on the message type."""
        ...


################################################################################
# Communication Client Factory
################################################################################


class CommunicationClientFactory(
    FactoryMixin[CommunicationClientType, CommunicationClientProtocol]
):
    """Factory for registering and creating BaseCommunicationClient instances based on the specified client type.

    Example:
    ```python
        # Register a new client type
        @CommunicationClientFactory.register(ClientType.PUB)
        class ZMQPubClient(BaseZMQClient):
            pass

        # Create a new client instance
        client = CommunicationClientFactory.create_instance(
            ClientType.PUB,
            address=ClientAddressType.SERVICE_XSUB_FRONTEND,
            bind=False,
        )
    ```
    """


################################################################################
# Base Communication Interface
################################################################################


class CommunicationProtocol(Protocol):
    """Base class for specifying the base communication layer for AIPerf components."""

    async def initialize(self) -> None:
        """Initialize communication channels."""

    @property
    def is_initialized(self) -> bool:
        """Check if communication channels are initialized.

        Returns:
            True if communication channels are initialized, False otherwise
        """
        ...

    async def shutdown(self) -> None:
        """Gracefully shutdown communication channels."""
        ...

    def get_address(self, address_type: CommunicationClientAddressType | str) -> str:
        """Get the address for a given address type.

        Args:
            address_type: The type of address to get the address for, or the address itself.

        Returns:
            The address for the given address type, or the address itself if it is a string.
        """
        ...

    def create_client(
        self,
        client_type: CommunicationClientType,
        address: CommunicationClientAddressType | str,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> CommunicationClientProtocol:
        """Create a communication client for a given client type and address.

        Args:
            client_type: The type of client to create.
            address: The type of address to use when looking up in the communication config, or the address itself.
            bind: Whether to bind or connect the socket.
            socket_ops: Additional socket options to set.
        """
        ...
