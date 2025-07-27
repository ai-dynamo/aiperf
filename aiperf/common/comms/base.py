# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from aiperf.common.constants import DEFAULT_COMMS_REQUEST_TIMEOUT
from aiperf.common.enums import (
    CommAddress,
    CommunicationBackend,
    CommunicationClientType,
)
from aiperf.common.factories import FactoryMixin
from aiperf.common.messages import Message
from aiperf.common.types import MessageOutputT, MessageT, MessageTypeT

################################################################################
# Base Communication Client Interfaces
################################################################################


@runtime_checkable
class CommunicationClientProtocol(Protocol):
    """Base interface for specifying the base communication client for AIPerf components."""

    async def initialize(self) -> None:
        """Initialize communication channels."""
        ...

    async def shutdown(self) -> None:
        """Shutdown communication channels."""
        ...


class CommunicationClientProtocolFactory(
    FactoryMixin[CommunicationClientType, CommunicationClientProtocol]
):
    """Factory for registering CommunicationClientProtocol interfaces for dynamic client creation."""


@CommunicationClientProtocolFactory.register(CommunicationClientType.PUSH)
class PushClientProtocol(CommunicationClientProtocol):
    """Interface for push clients."""

    async def push(self, message: Message) -> None:
        """Push data to a target. The message will be routed automatically
        based on the message.message_type.

        Args:
            message: Message to be pushed
        """
        ...


@CommunicationClientProtocolFactory.register(CommunicationClientType.PULL)
class PullClientProtocol(CommunicationClientProtocol):
    """Interface for pull clients."""

    async def register_pull_callback(
        self,
        message_type: MessageTypeT,
        callback: Callable[[MessageT], Coroutine[Any, Any, None]],
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


@CommunicationClientProtocolFactory.register(CommunicationClientType.REQUEST)
class RequestClientProtocol(CommunicationClientProtocol):
    """Interface for request clients."""

    async def request(
        self,
        message: MessageT,  # type: ignore[type-arg]
        timeout: float = DEFAULT_COMMS_REQUEST_TIMEOUT,
    ) -> MessageOutputT:  # type: ignore[type-arg]
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
        message: MessageT,
        callback: Callable[[MessageOutputT], Coroutine[Any, Any, None]],
    ) -> None:
        """Send a request and be notified when the response is received. The message will be routed automatically
        based on the message type.

        Args:
            message: Message to send (will be routed based on the message type)
            callback: Callback to be called when the response is received
        """
        ...


@CommunicationClientProtocolFactory.register(CommunicationClientType.REPLY)
class ReplyClientProtocol(CommunicationClientProtocol):
    """Interface for reply clients."""

    def register_request_handler(
        self,
        service_id: str,
        message_type: MessageTypeT,
        handler: Callable[[MessageT], Coroutine[Any, Any, MessageOutputT | None]],
    ) -> None:
        """Register a request handler for a message type. The handler will be called when
        a request is received for the given message type.

        Args:
            service_id: The service ID to register the handler for
            message_type: The message type to register the handler for
            handler: The handler to register
        """
        ...


@CommunicationClientProtocolFactory.register(CommunicationClientType.SUB)
class SubClientProtocol(CommunicationClientProtocol):
    """Interface for subscribe clients."""

    async def subscribe(
        self,
        message_type: MessageTypeT,
        callback: Callable[[MessageT], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to a specific message type. The callback will be called when
        a message is received for the given message type."""
        ...

    async def subscribe_all(
        self,
        message_callback_map: dict[
            MessageTypeT,
            Callable[[Message], Any] | list[Callable[[Message], Any]],
        ],
    ) -> None:
        """Subscribe to all message types in the map. The callback(s) will be called when
        a message is received for the given message type.

        Args:
            message_callback_map: A map of message types to callbacks. The callbacks can be a single callback or a list of callbacks.
        """
        ...


@CommunicationClientProtocolFactory.register(CommunicationClientType.PUB)
class PubClientProtocol(CommunicationClientProtocol):
    """Interface for publish clients."""

    async def publish(self, message: MessageT) -> None:
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


ClientProtocolT = TypeVar("ClientProtocolT", bound=CommunicationClientProtocol)


def _create_specific_client(
    client_type: CommunicationClientType,
    client_class: type[ClientProtocolT],
) -> Callable[
    [
        "BaseCommunication",
        CommAddress | str,
        bool,
        dict | None,
    ],
    ClientProtocolT,
]:
    def _create_client(
        self: "BaseCommunication",
        address: CommAddress | str,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> ClientProtocolT:
        return cast(
            ClientProtocolT, self.create_client(client_type, address, bind, socket_ops)
        )

    _create_client.__name__ = f"create_{client_type.lower()}_client"
    _create_client.__doc__ = f"Create a {client_type.upper()} client"
    return _create_client


class BaseCommunication(ABC):
    """Base class for specifying the base communication layer for AIPerf components."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize communication channels."""

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if communication channels are initialized.

        Returns:
            True if communication channels are initialized, False otherwise
        """

    @property
    @abstractmethod
    def stop_requested(self) -> bool:
        """Check if the communication channels are being shutdown.

        Returns:
            True if the communication channels are being shutdown, False otherwise
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown communication channels."""

    @abstractmethod
    def get_address(self, address_type: CommAddress | str) -> str:
        """Get the address for a given address type.

        Args:
            address_type: The type of address to get the address for, or the address itself.

        Returns:
            The address for the given address type, or the address itself if it is a string.
        """

    @abstractmethod
    def create_client(
        self,
        client_type: CommunicationClientType,
        address: CommAddress | str,
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

    create_pub_client = _create_specific_client(
        CommunicationClientType.PUB, PubClientProtocol
    )
    create_sub_client = _create_specific_client(
        CommunicationClientType.SUB, SubClientProtocol
    )
    create_push_client = _create_specific_client(
        CommunicationClientType.PUSH, PushClientProtocol
    )
    create_pull_client = _create_specific_client(
        CommunicationClientType.PULL, PullClientProtocol
    )
    create_request_client = _create_specific_client(
        CommunicationClientType.REQUEST, RequestClientProtocol
    )
    create_reply_client = _create_specific_client(
        CommunicationClientType.REPLY, ReplyClientProtocol
    )


class CommunicationFactory(FactoryMixin[CommunicationBackend, BaseCommunication]):
    """Factory for registering and creating BaseCommunication instances based on the specified communication backend.
    See :class:`FactoryMixin` for more details.
    """
