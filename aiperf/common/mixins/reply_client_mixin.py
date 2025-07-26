# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import CommAddress
from aiperf.common.hooks import (
    AIPerfHook,
    Hook,
    on_init,
    provides_hooks,
)
from aiperf.common.mixins.communication_mixin import CommunicationMixin
from aiperf.common.types import MessageTypeT


@provides_hooks(AIPerfHook.ON_REQUEST)
class ReplyClientMixin(CommunicationMixin, ABC):
    """Mixin to provide a reply client for AIPerf components using a ReplyClient for the specified CommAddress.
    Add the @on_request decorator to specify a function that will be called when a request is received.

    NOTE: This currently only supports a single reply client per service, as that is our current use case.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        reply_client_address: CommAddress,
        reply_client_bind: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(service_config=service_config, **kwargs)
        # NOTE: The communication base class will automatically manage the reply client's lifecycle.
        self.reply_client = self.comms.create_reply_client(
            reply_client_address, bind=reply_client_bind
        )

    @on_init
    async def _setup_request_handler_hooks(self) -> None:
        """Configure the reply client to handle requests for all @request_handler hook decorators."""

        def _register_request_handler(hook: Hook, message_type: MessageTypeT) -> None:
            self.debug(
                lambda: f"Registering request handler for message type: {message_type} for hook: {hook}"
            )
            self.reply_client.register_request_handler(
                service_id=self.id,
                message_type=message_type,
                handler=hook.func,
            )

        # For each @on_request hook, register a request handler for each message type.
        self.for_each_hook_param(
            AIPerfHook.ON_REQUEST,
            self_obj=self,
            param_type=MessageTypeT,
            lambda_func=_register_request_handler,
        )
