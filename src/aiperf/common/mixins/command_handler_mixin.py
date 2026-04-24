# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from abc import ABC
from typing import TYPE_CHECKING

if sys.version_info >= (3, 11):
    pass
else:
    pass

from aiperf.common.hooks import (
    AIPerfHook,
    provides_hooks,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


@provides_hooks(AIPerfHook.ON_COMMAND)
class CommandHandlerMixin(MessageBusClientMixin, ABC):
    """Mixin that declares @on_command hook support for services.

    Command dispatch is handled by the DEALER/ROUTER control channel:
    - BaseComponentService dispatches incoming Command structs to @on_command hooks
    - SystemController dispatches commands received on the ROUTER socket

    This mixin's role is to declare the ON_COMMAND hook type via @provides_hooks
    so that hook discovery works correctly in the class hierarchy.
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str,
        **kwargs,
    ) -> None:
        self.service_id = service_id

        super().__init__(
            run=run,
            **kwargs,
        )
