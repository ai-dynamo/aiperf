# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from aiperf.common.enums import CommAddress
from aiperf.common.hooks import (
    AIPerfHook,
    MessageHookParams,
    on_init,
    provides_hooks,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin


@provides_hooks(AIPerfHook.REQUEST_HANDLER)
class ReplyClientMixin(MessageBusClientMixin, ABC):
    """Mixin to provide a reply client for AIPerf components using a ReplyClient for the specified CommAddress.
    Add the @request_handler hook to specify a function that will be called when a request is received.
    """

    # TODO: Once we have a better dependency injection system, we can improve the way this is done.
    #       Currently, this extends the MessageBusClientMixin, so it can get the comms instance from there.
    #       I did not want to create a new mixin for just the comms instance
    def __init__(
        self,
        reply_client_address: CommAddress,
        reply_client_bind: bool = False,
        **kwargs,
    ) -> None:
        self.reply_client_address = reply_client_address
        self.reply_client_bind = reply_client_bind
        super().__init__(
            reply_client_address=reply_client_address,
            **kwargs,
        )
        # NOTE: The reply client will be automatically managed by the self.comms instance.
        #       This is why we don't need to initialize or start/stop the reply client.
        self.reply_client = self.comms.create_reply_client(
            self.reply_client_address, bind=self.reply_client_bind
        )

    @on_init
    async def _setup_request_handler_hooks(self) -> None:
        """Configure the reply client to handle requests for all @request_handler hook decorators."""
        for hook in self.get_hooks(AIPerfHook.REQUEST_HANDLER):
            if not isinstance(hook.params, MessageHookParams):
                raise ValueError(
                    f"Invalid hook params: {hook.params}. Expected MessageHookParams but got {type(hook.params)}"
                )
            self.debug(lambda hook=hook: f"Registering request handler {hook!r}")
            for message_type in hook.params.message_types:
                self.reply_client.register_request_handler(
                    service_id=self.id,
                    message_type=message_type,
                    handler=hook.func,  # type: ignore
                )
