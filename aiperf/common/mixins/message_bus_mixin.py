# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from collections.abc import Callable, Coroutine
from typing import Any

from aiperf.common.config import ServiceConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import CommAddress
from aiperf.common.hooks import AIPerfHook, Hook, on_init, provides_hooks
from aiperf.common.messages import Message
from aiperf.common.mixins.communication_mixin import CommunicationMixin
from aiperf.common.protocols import MessageBusClientProtocol
from aiperf.common.types import MessageCallbackMapT, MessageTypeT


@provides_hooks(AIPerfHook.ON_MESSAGE)
@implements_protocol(MessageBusClientProtocol)
class MessageBusClientMixin(CommunicationMixin, ABC):
    """Mixin to provide message bus clients (pub and sub)for AIPerf components, as well as
    a hook to handle messages: @on_message."""

    def __init__(self, service_config: ServiceConfig, **kwargs) -> None:
        super().__init__(service_config=service_config, **kwargs)
        # NOTE: The communication base class will automatically manage the pub/sub clients' lifecycle.
        self.sub_client = self.comms.create_sub_client(
            CommAddress.EVENT_BUS_PROXY_BACKEND
        )
        self.pub_client = self.comms.create_pub_client(
            CommAddress.EVENT_BUS_PROXY_FRONTEND
        )

    @on_init
    async def _setup_on_message_hooks(self) -> None:
        """Send subscription requests for all @on_message hook decorators."""
        subscription_map: MessageCallbackMapT = {}

        def _add_to_subscription_map(hook: Hook, message_type: MessageTypeT) -> None:
            """
            This function is called for every message_type parameter of every @on_message hook.
            We use this to build a map of message types to callbacks, which is then used to call
            subscribe_all for efficiency
            """
            self.debug(
                lambda: f"Subscribing to message type: '{message_type}' for hook: {hook}"
            )
            subscription_map.setdefault(message_type, []).append(hook.func)

        # For each @on_message hook, add each message type to the subscription map.
        self.for_each_hook_param(
            AIPerfHook.ON_MESSAGE,
            self_obj=self,
            param_type=MessageTypeT,
            lambda_func=_add_to_subscription_map,
        )
        await self.sub_client.subscribe_all(subscription_map)

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
