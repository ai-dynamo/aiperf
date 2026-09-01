# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC

from aiperf.common.hooks import (
    AIPerfHook,
    provides_hooks,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin


@provides_hooks(AIPerfHook.ON_COMMAND)
class CommandHandlerMixin(MessageBusClientMixin, ABC):
    """Mixin that declares @on_command hook support for services.

    Command dispatch itself lives on the DEALER/ROUTER control channel:
    ``BaseComponentService._handle_control_command`` dispatches incoming
    ``Command`` structs to this service's ``@on_command`` hooks, and
    ``SystemController._dispatch_control_command`` does the same for commands
    arriving on the ROUTER.

    This mixin's only remaining role is to declare the ON_COMMAND hook type via
    ``@provides_hooks`` so hook discovery works across the class hierarchy.
    """

    def __init__(
        self,
        service_id: str,
        **kwargs,
    ) -> None:
        self.service_id = service_id

        super().__init__(
            **kwargs,
        )
