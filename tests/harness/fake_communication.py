# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-memory fake communication backend for component testing.

This module registers FakeCommunication and FakeProxy with their respective
factories using maximum priority, automatically replacing ZMQ implementations
when imported. No configuration changes are needed - simply importing this
module hot-swaps all communication infrastructure.

Bypasses ZMQ entirely, routing messages in-memory between clients at the same
address for fast, isolated testing without network or IPC overhead.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, ClassVar

from aiperf.common.base_comms import BaseCommunication
from aiperf.common.enums import CommAddress
from aiperf.common.hooks import on_stop
from aiperf.common.messages import TargetedServiceMessage
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.types import CommAddressType, MessageCallbackMapT, MessageTypeT
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    CommClientType,
    CommunicationBackend,
    PluginType,
    ZMQProxyType,
)
from aiperf.zmq.zmq_defaults import TOPIC_DELIMITER


@dataclass(frozen=True)
class CapturedPayload:
    """Captured payload from the communication layer."""

    client_type: CommClientType
    address: str
    payload: Any
    timestamp_ns: int
    topic: str | None = None
    sender_identity: str | None = None
    receiver_identity: str | None = None


# =============================================================================
# Base Fake Proxy
# =============================================================================


class FakeProxy(AIPerfLifecycleMixin):
    """No-op fake proxy replacing ZMQ proxies (test double: Fake).

    Since FakeCommunication routes messages directly between clients at the same
    address, proxies become unnecessary and this provides a minimal lifecycle stub.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(
        cls,
        *args,
        **kwargs,
    ) -> FakeProxy:
        return cls(*args, **kwargs)


# =============================================================================
# Fake Clients - Minimal implementations of protocol methods
# =============================================================================


class FakeCommunicationClient(AIPerfLifecycleMixin):
    """Fake communication client."""

    client_type: ClassVar[CommClientType]

    def __init__(
        self,
        address: str,
        identity: str,
        bus: FakeCommunicationBus,
        additional_bind_address: str | None = None,
    ) -> None:
        super().__init__(id=identity)
        self.bus = bus
        self.address = address
        self.identity = identity
        self.additional_bind_address = additional_bind_address

    def capture_sent_payload(
        self,
        /,
        payload: Any,
        *,
        receiver_identity: str | None = None,
        topic: str | None = None,
    ) -> None:
        self.bus.sent_payloads.append(
            CapturedPayload(
                payload=payload,
                topic=topic,
                receiver_identity=receiver_identity,
                client_type=self.client_type,
                address=self.address,
                timestamp_ns=time.perf_counter_ns(),
                sender_identity=self.identity,
            )
        )

    def capture_received_payload(
        self,
        /,
        payload: Any,
        *,
        sender_identity: str | None = None,
        topic: str | None = None,
    ) -> None:
        self.bus.received_payloads.append(
            CapturedPayload(
                payload=payload,
                topic=topic,
                sender_identity=sender_identity,
                client_type=self.client_type,
                address=self.address,
                timestamp_ns=time.perf_counter_ns(),
                receiver_identity=self.identity,
            )
        )


class FakeStreamingRouterClient(FakeCommunicationClient):
    """Fake ROUTER - receives from dealers, sends to specific dealer by identity."""

    client_type = CommClientType.STREAMING_ROUTER

    def __init__(
        self,
        address: str,
        identity: str,
        bus: FakeCommunicationBus,
        additional_bind_address: str | None = None,
    ) -> None:
        super().__init__(
            address, identity, bus, additional_bind_address=additional_bind_address
        )
        self.handler: Callable[[str, Any], Awaitable[None]] | None = None

    def register_receiver(
        self, handler: Callable[[str, Any], Coroutine[Any, Any, None]]
    ) -> None:
        if self.handler is not None:
            raise ValueError("Receiver handler already registered")
        self.handler = handler

    async def send_to(self, identity: str, message: Any) -> None:
        """Send to dealer - dynamically looks up dealer by identity."""
        self.capture_sent_payload(message, receiver_identity=identity)
        for comm in self.bus.communications:
            if dealer_client := comm.dealer_clients.get(identity):
                if dealer_client.handler:
                    dealer_client.capture_received_payload(
                        message, sender_identity=self.identity
                    )
                    await dealer_client.handler(message)
                else:
                    self.warning(f"No handler registered for dealer client {identity}")
                return


class FakeStreamingDealerClient(FakeCommunicationClient):
    """Fake DEALER - sends to router, receives from router."""

    client_type = CommClientType.STREAMING_DEALER

    def __init__(self, address: str, identity: str, bus: FakeCommunicationBus) -> None:
        super().__init__(address, identity, bus)
        self.handler: Callable[[Any], Awaitable[None]] | None = None

    def register_receiver(
        self, handler: Callable[[Any], Coroutine[Any, Any, None]]
    ) -> None:
        if self.handler is not None:
            raise ValueError("Receiver handler already registered")
        self.handler = handler

    async def send(self, message: Any) -> None:
        """Send to router - dynamically looks up routers at this address."""
        self.capture_sent_payload(message)
        for comm in self.bus.communications:
            for router_client in comm.router_clients.get(self.address, []):
                if router_client.handler:
                    router_client.capture_received_payload(
                        message, sender_identity=self.identity
                    )
                    await router_client.handler(self.identity, message)


class FakePubClient(FakeCommunicationClient):
    """Fake PUB - publishes to all subscribers at same address."""

    client_type = CommClientType.PUB

    def _determine_topic(self, message: Any) -> str:
        """Determine topic based on message type and targeting."""
        msg_type = getattr(message, "message_type", None)
        if isinstance(message, TargetedServiceMessage):
            if message.target_service_id:
                return f"{msg_type}{TOPIC_DELIMITER}{message.target_service_id}"
            if message.target_service_type:
                return f"{msg_type}{TOPIC_DELIMITER}{message.target_service_type}"
        return str(msg_type)

    async def publish(self, message: Any) -> None:
        """Publish to subscribers - dynamically looks up subs at this address."""
        topic = self._determine_topic(message)
        self.capture_sent_payload(message, topic=topic)
        # Make sure to copy the subscriptions list to avoid modifying it while iterating
        subscribers = []
        for comm in self.bus.communications:
            for sub_client in comm.sub_clients:
                if (
                    sub_client.address == self.address
                    and topic in sub_client.subscriptions
                ):
                    subscribers.append(sub_client)
        for subscriber in subscribers:
            subscriber.capture_received_payload(
                message, sender_identity=self.identity, topic=topic
            )
            callbacks = list(subscriber.subscriptions[topic])
            for callback in callbacks:
                await callback(message)


class FakeSubClient(FakeCommunicationClient):
    """Fake SUB - subscribes to message types."""

    client_type = CommClientType.SUB

    def __init__(self, address: str, identity: str, bus: FakeCommunicationBus) -> None:
        super().__init__(address, identity, bus)
        # Keyed by topic string (e.g., "MessageType.COMMAND" or "MessageType.COMMAND.service-id")
        self.subscriptions: dict[str, list[Callable]] = {}

    async def subscribe(
        self,
        message_type: MessageTypeT,
        callback: Callable[[Any], Coroutine[Any, Any, None]],
    ) -> None:
        topic = str(message_type)
        self.subscriptions.setdefault(topic, []).append(callback)

    async def subscribe_all(self, message_callback_map: MessageCallbackMapT) -> None:
        for msg_type, callbacks in message_callback_map.items():
            topic = str(msg_type)
            if isinstance(callbacks, list):
                self.subscriptions.setdefault(topic, []).extend(callbacks)
            else:
                self.subscriptions.setdefault(topic, []).append(callbacks)


class FakePushClient(FakeCommunicationClient):
    """Fake PUSH - pushes to pull clients at same address (round-robin)."""

    client_type = CommClientType.PUSH

    def __init__(self, address: str, identity: str, bus: FakeCommunicationBus) -> None:
        super().__init__(address, identity, bus)
        self.round_robin_index: int = 0

    async def push(self, message: Any) -> None:
        """Push to next pull client - dynamically looks up pulls at this address."""
        self.capture_sent_payload(message)
        # Collect all pull clients at this address
        pull_clients = [
            pull_client
            for comm in self.bus.communications
            for pull_client in comm.pull_clients
            if pull_client.address == self.address
        ]
        if not pull_clients:
            return
        pull_client = pull_clients[self.round_robin_index % len(pull_clients)]
        self.round_robin_index += 1
        msg_type = getattr(message, "message_type", None)
        if msg_type and (callback := pull_client.callbacks.get(msg_type)):
            pull_client.capture_received_payload(message, sender_identity=self.identity)
            await callback(message)


class FakePullClient(FakeCommunicationClient):
    """Fake PULL - receives from push clients (one callback per message type)."""

    client_type = CommClientType.PULL

    def __init__(
        self,
        address: str,
        identity: str,
        bus: FakeCommunicationBus,
        additional_bind_address: str | None = None,
    ) -> None:
        super().__init__(
            address, identity, bus, additional_bind_address=additional_bind_address
        )
        self.callbacks: dict[Any, Callable] = {}  # ONE callback per type

    def register_pull_callback(
        self,
        message_type: MessageTypeT,
        callback: Callable[[Any], Coroutine[Any, Any, None]],
    ) -> None:
        if message_type in self.callbacks:
            raise ValueError(
                f"Callback already registered for message type {message_type}"
            )
        self.callbacks[message_type] = callback


class FakeRequestClient(FakeCommunicationClient):
    """Fake REQUEST - sends request and waits for reply."""

    client_type = CommClientType.REQUEST

    def __init__(self, address: str, identity: str, bus: FakeCommunicationBus) -> None:
        super().__init__(address, identity, bus)

    async def request(self, message: Any, timeout: float = 30.0) -> Any:  # noqa: ARG002
        """Send request - dynamically looks up reply clients at this address."""
        self.capture_sent_payload(message)
        for comm in self.bus.communications:
            for reply_client in comm.reply_clients.get(self.address, []):
                reply_client.capture_received_payload(
                    message, sender_identity=self.identity
                )
                return await reply_client.handle_request(message)
        return None

    async def request_async(
        self,
        message: Any,
        callback: Callable[[Any], Coroutine[Any, Any, None]],
    ) -> None:
        """Send request and call callback with response."""
        response = await self.request(message)
        if response is not None:
            await callback(response)


class FakeReplyClient(FakeCommunicationClient):
    """Fake REPLY - handles requests and sends replies."""

    client_type = CommClientType.REPLY

    def __init__(self, address: str, identity: str, bus: FakeCommunicationBus) -> None:
        super().__init__(address, identity, bus)
        self.handlers: dict[
            MessageTypeT, tuple[str, Callable]
        ] = {}  # keyed by msg_type

    def register_request_handler(
        self,
        service_id: str,
        message_type: MessageTypeT,
        handler: Callable[[Any], Coroutine[Any, Any, Any]],
    ) -> None:
        if message_type in self.handlers:
            raise ValueError(
                f"Handler already registered for message type {message_type}"
            )
        self.handlers[message_type] = (service_id, handler)

    async def handle_request(self, message: Any) -> Any:
        """Handle incoming request by finding matching handler."""
        msg_type = getattr(message, "message_type", None)
        if msg_type and msg_type in self.handlers:
            _, handler = self.handlers[msg_type]
            return await handler(message)
        return None


# =============================================================================
# FakeCommunicationBus - Shared routing state
# =============================================================================


class FakeCommunicationBus:
    """Registry of FakeCommunication instances for cross-wiring.

    Each FakeCommunication owns its own clients. The bus tracks all
    instances so they can discover and wire to each other's clients.
    """

    def __init__(self) -> None:
        self.communications: list[FakeCommunication] = []
        self.sent_payloads: list[CapturedPayload] = []
        self.received_payloads: list[CapturedPayload] = []

    def register(self, comm: FakeCommunication) -> None:
        """Register a FakeCommunication instance with the bus."""
        if comm not in self.communications:
            self.communications.append(comm)

    def unregister(self, comm: FakeCommunication) -> None:
        """Unregister a FakeCommunication instance from the bus."""
        if comm in self.communications:
            self.communications.remove(comm)


# =============================================================================
# FakeCommunication - Factory-registered fake backend
# =============================================================================


class FakeCommunication(BaseCommunication):
    """In-memory communication backend replacing ZMQ (test double: Fake).

    Auto-wires clients at the same address:
    - Router ↔ Dealer
    - Pub → Sub
    - Push → Pull (round-robin)
    - Request ↔ Reply

    For multi-service testing, use a shared bus:
        FakeCommunication.set_shared_bus(FakeCommunicationBus())
    """

    # Class-level shared bus - all instances share the same bus
    shared_bus: FakeCommunicationBus = FakeCommunicationBus()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bus = FakeCommunication.shared_bus
        # Per-instance client collections
        self.router_clients: dict[str, list[FakeStreamingRouterClient]] = defaultdict(
            list
        )
        self.dealer_clients: dict[str, FakeStreamingDealerClient] = defaultdict(
            list
        )  # by identity
        self.pub_clients: dict[str, list[FakePubClient]] = defaultdict(list)
        self.sub_clients: list[FakeSubClient] = []
        self.push_clients: dict[str, list[FakePushClient]] = defaultdict(list)
        self.pull_clients: list[FakePullClient] = []
        self.request_clients: list[FakeRequestClient] = []
        self.reply_clients: dict[str, list[FakeReplyClient]] = defaultdict(list)
        # Client cache for deduplication on the same service like the real communication layer
        self.clients_cache: dict[
            tuple[CommClientType, str, bool], FakeCommunicationClient
        ] = {}
        # Register with bus for cross-wiring
        self.bus.register(self)
        self.warning(
            "*** Using FakeCommunication to bypass ZMQ. This is for component integration testing only. ***"
        )

    @classmethod
    def set_shared_bus(cls, bus: FakeCommunicationBus) -> None:
        """Set class-level shared bus for all new instances."""
        cls.shared_bus = bus

    @classmethod
    def clear_shared_bus(cls) -> None:
        """Clear shared bus (for test cleanup)."""
        cls.shared_bus = FakeCommunicationBus()

    def get_address(self, address_type: CommAddressType) -> str:
        """Return fake address string.

        Normalizes proxy addresses by stripping _frontend/_backend suffixes.
        This makes the proxy transparent - clients connecting to frontend and
        backend of the same proxy get wired together automatically.
        """
        if isinstance(address_type, CommAddress):
            addr = f"fake://{address_type.value}"
        else:
            addr = str(address_type)
        return self._normalize_proxy_address(addr)

    @staticmethod
    def _normalize_proxy_address(address: str) -> str:
        """Strip _frontend/_backend suffixes to make proxy transparent.

        Real ZMQ proxies forward messages between frontend and backend sockets.
        By normalizing addresses, clients at frontend and backend of the same
        proxy end up at the same address, so FakeCommunicationBus wires them
        together automatically.

        Example:
            fake://event_bus_proxy_frontend -> fake://event_bus_proxy
            fake://event_bus_proxy_backend  -> fake://event_bus_proxy
            ipc:///tmp/proxy_frontend.ipc   -> ipc:///tmp/proxy.ipc
        """
        for suffix in ("_frontend", "_backend", "_frontend.ipc", "_backend.ipc"):
            if address.endswith(suffix):
                return address[: -len(suffix)] + (
                    ".ipc" if suffix.endswith(".ipc") else ""
                )
        return address

    def create_client(
        self,
        client_type: CommClientType,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,  # noqa: ARG002
        max_pull_concurrency: int | None = None,  # noqa: ARG002
        **kwargs,
    ) -> FakeCommunicationClient:
        """Create fake client and auto-wire to counterparts."""
        addr = self.get_address(address)

        # Check cache first (matching ZMQ behavior)
        cache_key = (client_type, addr, bind)
        if cache_key in self.clients_cache:
            return self.clients_cache[cache_key]

        client: FakeCommunicationClient
        identity = kwargs.get(
            "identity", f"{client_type}-{len(self.clients_by_type(client_type))}"
        )
        additional_bind = kwargs.get("additional_bind_address")
        match client_type:
            case CommClientType.STREAMING_ROUTER:
                client = FakeStreamingRouterClient(
                    addr, identity, self.bus, additional_bind_address=additional_bind
                )
                self.router_clients[addr].append(client)

            case CommClientType.STREAMING_DEALER:
                client = FakeStreamingDealerClient(addr, identity, self.bus)
                self.dealer_clients[identity] = client

            case CommClientType.PUB:
                client = FakePubClient(addr, identity, self.bus)
                self.pub_clients[addr].append(client)

            case CommClientType.SUB:
                client = FakeSubClient(addr, identity, self.bus)
                self.sub_clients.append(client)

            case CommClientType.PUSH:
                client = FakePushClient(addr, identity, self.bus)
                self.push_clients[addr].append(client)

            case CommClientType.PULL:
                client = FakePullClient(
                    addr, identity, self.bus, additional_bind_address=additional_bind
                )
                self.pull_clients.append(client)

            case CommClientType.REQUEST:
                client = FakeRequestClient(addr, identity, self.bus)
                self.request_clients.append(client)

            case CommClientType.REPLY:
                client = FakeReplyClient(addr, identity, self.bus)
                self.reply_clients[addr].append(client)

            case _:
                raise ValueError(f"Unsupported client type: {client_type}")

        self.clients_cache[cache_key] = client
        # Note: Don't call attach_child_lifecycle - fake clients can be created
        # after the fake is started, and don't need lifecycle management
        return client

    def clients_by_type(
        self, client_type: CommClientType
    ) -> list[FakeCommunicationClient]:
        """Get all clients by type."""
        return [
            client
            for client in self.clients_cache.values()
            if client.client_type == client_type
        ]

    @on_stop
    async def _unregister_from_bus(self) -> None:
        """Unregister from bus and clear local collections."""
        self.bus.unregister(self)
        self.router_clients.clear()
        self.dealer_clients.clear()
        self.pub_clients.clear()
        self.sub_clients.clear()
        self.push_clients.clear()
        self.pull_clients.clear()
        self.request_clients.clear()
        self.reply_clients.clear()
        self.clients_cache.clear()


# =============================================================================
# Plugin Registration - Hot-swap production implementations when imported
# =============================================================================

# Register FakeProxy for all ZMQ proxy types at max priority
plugins.register(
    PluginType.ZMQ_PROXY,
    ZMQProxyType.XPUB_XSUB,
    FakeProxy,
    priority=sys.maxsize,
)
plugins.register(
    PluginType.ZMQ_PROXY,
    ZMQProxyType.DEALER_ROUTER,
    FakeProxy,
    priority=sys.maxsize,
)
plugins.register(
    PluginType.ZMQ_PROXY,
    ZMQProxyType.PUSH_PULL,
    FakeProxy,
    priority=sys.maxsize,
)

# Register FakeCommunication for ZMQ IPC backend at max priority
plugins.register(
    PluginType.COMMUNICATION,
    CommunicationBackend.ZMQ_IPC,
    FakeCommunication,
    priority=sys.maxsize,
)
