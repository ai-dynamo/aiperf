# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from collections.abc import Callable, Coroutine
from typing import Any

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import CommAddress
from aiperf.common.factories import CommunicationFactory
from aiperf.common.hooks import (
    AIPerfHook,
    MessageHookParams,
    on_init,
    on_start,
    on_stop,
    provides_hooks,
)
from aiperf.common.messages import Message
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.protocols import CommunicationProtocol
from aiperf.common.types import MessageCallbackMapT, MessageTypeT


@provides_hooks(AIPerfHook.ON_MESSAGE)
class MessageBusClientMixin(AIPerfLifecycleMixin, ABC):
    """Mixin to provide a message bus clients for AIPerf components, as well as
    a hook to handle messages @on_message."""

    def __init__(self, service_config: ServiceConfig, **kwargs) -> None:
        self.service_config = service_config
        self.comms: CommunicationProtocol = CommunicationFactory.create_instance(
            self.service_config.comm_backend,
            config=self.service_config.comm_config,
        )
        self.sub_client = self.comms.create_sub_client(
            CommAddress.EVENT_BUS_PROXY_BACKEND
        )
        self.pub_client = self.comms.create_pub_client(
            CommAddress.EVENT_BUS_PROXY_FRONTEND
        )
        super().__init__(
            comms=self.comms,
            service_config=self.service_config,
            pub_client=self.pub_client,
            sub_client=self.sub_client,
            **kwargs,
        )

    @on_init
    async def _init_message_bus(self) -> None:
        await self.comms.initialize()
        await self._setup_on_message_hooks()

    @on_start
    async def _start_message_bus(self) -> None:
        await self.comms.start()

    @on_stop
    async def _stop_message_bus(self) -> None:
        await self.comms.stop()

    async def subscribe(
        self,
        message_type: MessageTypeT,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to a specific message type. The callback will be called when
        a message is received for the given message type."""
        await self.sub_client.subscribe(message_type, callback)

    async def subscribe_all(
        self,
        message_callback_map: MessageCallbackMapT,
    ) -> None:
        """Subscribe to all message types in the map. The callback(s) will be called when
        a message is received for the given message type.

        Args:
            message_callback_map: A map of message types to callbacks. The callbacks can be a single callback or a list of callbacks.
        """
        await self.sub_client.subscribe_all(message_callback_map)

    async def publish(self, message: Message) -> None:
        """Publish a message. The message will be routed automatically based on the message type."""
        await self.pub_client.publish(message)

    async def _setup_on_message_hooks(self) -> None:
        """Send subscription requests for all @on_message hook decorators."""
        subscription_map: MessageCallbackMapT = {}
        for hook in self.get_hooks(AIPerfHook.ON_MESSAGE):
            if not isinstance(hook.params, MessageHookParams):
                raise ValueError(f"Invalid hook params: {hook.params}")
            for message_type in hook.params.message_types:
                subscription_map.setdefault(message_type, []).append(hook.func)

        await self.sub_client.subscribe_all(subscription_map)
