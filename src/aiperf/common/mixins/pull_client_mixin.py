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


@provides_hooks(AIPerfHook.ON_PULL_MESSAGE)
class PullClientMixin(CommunicationMixin, ABC):
    """Mixin to provide a pull client for AIPerf components using a PullClient for the specified CommAddress.
    Add the @on_pull_message decorator to specify a function that will be called when a pull is received.

    NOTE: This currently only supports a single pull client per service, as that is our current use case.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        pull_client_address: CommAddress,
        pull_client_bind: bool = False,
        max_pull_concurrency: int | None = None,
        pull_client_additional_bind_address: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(service_config=service_config, **kwargs)
        # NOTE: The communication base class will automatically manage the pull client's lifecycle.
        self.pull_client = self.comms.create_pull_client(
            pull_client_address,
            bind=pull_client_bind,
            max_pull_concurrency=max_pull_concurrency,
            additional_bind_address=pull_client_additional_bind_address,
        )

    @on_init
    async def _setup_pull_handler_hooks(self) -> None:
        """Configure the pull client to register callbacks for all @on_pull_message hook decorators."""

        def _register_pull_callback(hook: Hook, message_type: MessageTypeT) -> None:
            self.debug(
                lambda: f"Registering pull callback for message type: {message_type} for hook: {hook}"
            )
            self.pull_client.register_pull_callback(
                message_type=message_type,
                callback=hook.func,
            )

        # For each @on_pull_message hook, register a pull callback for each specified message type.
        self.for_each_hook_param(
            AIPerfHook.ON_PULL_MESSAGE,
            self_obj=self,
            param_type=MessageTypeT,
            lambda_func=_register_pull_callback,
        )
