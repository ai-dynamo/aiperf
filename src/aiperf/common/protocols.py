# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.environment import Environment

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable, Coroutine
    from typing import Any

    from aiperf.common.config import ServiceConfig, UserConfig
    from aiperf.common.enums import LifecycleState
    from aiperf.common.models import (
        MessageCallbackMapT,
        MessageOutputT,
        MessageT,
        MessageTypeT,
    )
    from aiperf.common.types import CommAddressType, ServiceTypeT
    from aiperf.plugin.enums import CommClientType


@runtime_checkable
class AIPerfLoggerProtocol(Protocol):
    """Protocol for AIPerf logger methods."""

    @property
    def is_trace_enabled(self) -> bool: ...
    @property
    def is_debug_enabled(self) -> bool: ...

    def __init__(self, logger_name: str | None = None, **kwargs) -> None: ...
    def log(
        self, level: int, message: str | Callable[..., str], *args, **kwargs
    ) -> None: ...
    def trace_or_debug(
        self,
        trace_msg: str | Callable[..., str],
        debug_msg: str | Callable[..., str],
    ) -> None: ...
    def trace(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def debug(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def info(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def notice(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def warning(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def success(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def error(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def exception(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def critical(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def is_enabled_for(self, level: int) -> bool: ...


@runtime_checkable
class TaskManagerProtocol(AIPerfLoggerProtocol, Protocol):
    """Protocol for TaskManager methods."""

    def execute_async(self, coro: Coroutine) -> asyncio.Task: ...

    async def cancel_all_tasks(self, timeout: float) -> None: ...

    async def wait_for_tasks(self) -> list[BaseException | None]: ...

    def start_background_task(
        self,
        method: Callable,
        interval: float | Callable[[TaskManagerProtocol], float] | None = None,
        immediate: bool = False,
        stop_on_error: bool = False,
    ) -> None: ...


@runtime_checkable
class AIPerfLifecycleProtocol(TaskManagerProtocol, Protocol):
    """Protocol for AIPerf lifecycle methods."""

    @property
    def was_initialized(self) -> bool: ...
    @property
    def was_started(self) -> bool: ...
    @property
    def was_stopped(self) -> bool: ...
    @property
    def is_running(self) -> bool: ...
    @property
    def stop_requested(self) -> bool: ...
    @stop_requested.setter
    def stop_requested(self, value: bool) -> None: ...

    initialized_event: asyncio.Event
    started_event: asyncio.Event
    stopped_event: asyncio.Event

    @property
    def state(self) -> LifecycleState: ...

    async def initialize(self) -> None: ...
    async def start(self) -> None: ...
    async def initialize_and_start(self) -> None: ...
    async def stop(self) -> None: ...


################################################################################
# Communication Client Protocols (sorted alphabetically)
################################################################################


@runtime_checkable
class CommunicationClientProtocol(AIPerfLifecycleProtocol, Protocol):
    def __init__(
        self,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
        **kwargs,
    ) -> None: ...


@runtime_checkable
class PubClientProtocol(CommunicationClientProtocol, Protocol):
    async def publish(self, message: MessageT) -> None: ...


@runtime_checkable
class PullClientProtocol(CommunicationClientProtocol, Protocol):
    def register_pull_callback(
        self,
        message_type: MessageTypeT,
        callback: Callable[[MessageT], Coroutine[Any, Any, None]],
    ) -> None: ...


@runtime_checkable
class PushClientProtocol(CommunicationClientProtocol, Protocol):
    async def push(self, message: MessageT) -> None: ...


@runtime_checkable
class ReplyClientProtocol(CommunicationClientProtocol, Protocol):
    def register_request_handler(
        self,
        service_id: str,
        message_type: MessageTypeT,
        handler: Callable[[MessageT], Coroutine[Any, Any, MessageOutputT | None]],
    ) -> None: ...


@runtime_checkable
class RequestClientProtocol(CommunicationClientProtocol, Protocol):
    async def request(
        self,
        message: MessageT,
        timeout: float = Environment.SERVICE.COMMS_REQUEST_TIMEOUT,
    ) -> MessageOutputT: ...

    async def request_async(
        self,
        message: MessageT,
        callback: Callable[[MessageOutputT], Coroutine[Any, Any, None]],
    ) -> None: ...


@runtime_checkable
class StreamingRouterClientProtocol(CommunicationClientProtocol, Protocol):
    """Protocol for ROUTER socket client with bidirectional streaming."""

    def register_receiver(
        self,
        handler: Callable[[str, MessageT], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Register handler for incoming messages from DEALER clients.

        Args:
            handler: Async function that takes (identity: str, message: Message)
        """
        ...

    async def send_to(self, identity: str, message: MessageT) -> None:
        """
        Send message to specific DEALER client by identity.

        Args:
            identity: The DEALER client's identity (routing key)
            message: The message to send
        """
        ...


@runtime_checkable
class StreamingDealerClientProtocol(CommunicationClientProtocol, Protocol):
    """Protocol for DEALER socket client with bidirectional streaming."""

    def register_receiver(
        self,
        handler: Callable[[MessageT], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Register handler for incoming messages from ROUTER.

        Args:
            handler: Async function that takes (message: Message)
        """
        ...

    async def send(self, message: MessageT) -> None:
        """
        Send message to ROUTER.

        Args:
            message: The message to send
        """
        ...


@runtime_checkable
class SubClientProtocol(CommunicationClientProtocol, Protocol):
    async def subscribe(
        self,
        message_type: MessageTypeT,
        callback: Callable[[MessageT], Coroutine[Any, Any, None]],
    ) -> None: ...

    async def subscribe_all(
        self,
        message_callback_map: MessageCallbackMapT,
    ) -> None: ...


################################################################################
# Communication Protocol (must come after the clients)
################################################################################


@runtime_checkable
class CommunicationProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for the base communication layer.
    see :class:`aiperf.common.comms.base_comms.BaseCommunication` for more details.
    """

    def get_address(self, address_type: CommAddressType) -> str: ...

    """Get the address for the given address type can be an enum value for lookup, or a string for direct use."""

    def create_client(
        self,
        client_type: CommClientType,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
        max_pull_concurrency: int | None = None,
        **kwargs: Any,
    ) -> CommunicationClientProtocol:
        """Create a client for the given client type and address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_pub_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> PubClientProtocol:
        """Create a PUB client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_sub_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> SubClientProtocol:
        """Create a SUB client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_push_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> PushClientProtocol:
        """Create a PUSH client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_pull_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
        max_pull_concurrency: int | None = None,
    ) -> PullClientProtocol:
        """Create a PULL client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_request_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> RequestClientProtocol:
        """Create a REQUEST client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_reply_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> ReplyClientProtocol:
        """Create a REPLY client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_streaming_router_client(
        self,
        address: CommAddressType,
        bind: bool = True,
        socket_ops: dict | None = None,
    ) -> StreamingRouterClientProtocol:
        """Create a STREAMING_ROUTER client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_streaming_dealer_client(
        self,
        address: CommAddressType,
        identity: str,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> StreamingDealerClientProtocol:
        """Create a STREAMING_DEALER client for the given address and identity, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...


################################################################################
# Service Protocols
################################################################################


@runtime_checkable
class MessageBusClientProtocol(PubClientProtocol, SubClientProtocol, Protocol):
    """A message bus client is a client that can publish and subscribe to messages
    on the event bus/message bus."""

    comms: CommunicationProtocol
    sub_client: SubClientProtocol
    pub_client: PubClientProtocol


@runtime_checkable
class ServiceProtocol(MessageBusClientProtocol, Protocol):
    """Protocol for a service. Essentially a MessageBusClientProtocol with a service_type and service_id attributes."""

    def __init__(
        self,
        user_config: UserConfig,
        service_config: ServiceConfig,
        service_id: str | None = None,
        **kwargs,
    ) -> None: ...

    service_type: ServiceTypeT
    service_id: str
