# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import signal
import uuid
from abc import ABC
from typing import ClassVar

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import CommandType, LifecycleState
from aiperf.common.exceptions import ServiceError
from aiperf.common.hooks import on_command
from aiperf.common.messages.command_messages import (
    CommandAcknowledgedResponse,
    CommandMessage,
)
from aiperf.common.mixins.command_handler_mixin import CommandHandlerMixin
from aiperf.common.mixins.process_health_mixin import ProcessHealthMixin
from aiperf.common.types import ServiceTypeT


class BaseService(CommandHandlerMixin, ProcessHealthMixin, ABC):
    """Base class for all AIPerf services, providing common functionality for
    communication, state management, and lifecycle operations.
    This class inherits from the MessageBusClientMixin, which provides the
    message bus client functionality.

    This class provides the foundation for implementing the various services of the
    AIPerf system. Some of the abstract methods are implemented here, while others
    are still required to be implemented by derived classes.
    """

    service_type: ClassVar[ServiceTypeT]
    """The type of service this class implements. This is set by the ServiceFactory.register decorator."""

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        self.service_config = service_config
        self.user_config = user_config
        self.service_id = service_id or f"{self.service_type}_{uuid.uuid4().hex[:8]}"
        super().__init__(
            service_id=self.service_id,
            id=self.service_id,
            service_config=self.service_config,
            user_config=self.user_config,
            **kwargs,
        )
        self.debug(
            lambda: f"__init__ {self.service_type} service (id: {self.service_id})"
        )
        self._set_process_title()

    def _set_process_title(self) -> None:
        try:
            import setproctitle

            setproctitle.setproctitle(f"aiperf {self.service_id}")
        except Exception:
            # setproctitle is not available on all platforms, so we ignore the error
            self.debug("Failed to set process title, ignoring")

    def _service_error(self, message: str) -> ServiceError:
        return ServiceError(
            message=message,
            service_type=self.service_type,
            service_id=self.service_id,
        )

    @on_command(CommandType.SHUTDOWN)
    async def _on_shutdown_command(self, message: CommandMessage) -> None:
        self.debug(f"Received shutdown command from {message.service_id}")
        # Send an acknowledged response back to the sender, because we won't be able to send it after we stop.
        await self.publish(
            CommandAcknowledgedResponse.from_command_message(message, self.service_id)
        )

        try:
            await self.stop()
        except Exception as e:
            self.exception(
                f"Failed to stop service {self} ({self.service_id}) after receiving shutdown command: {e}. Killing."
            )
            await self._kill()

    async def stop(self) -> None:
        """This overrides the base class stop method to handle the case where the service is already stopping.
        In this case, we need to kill the process to be safe."""
        if self.stop_requested:
            self.error(f"Attempted to stop {self} in state {self.state}. Killing.")
            await self._kill()
            return
        await super().stop()

    async def _kill(self) -> None:
        """Kill the lifecycle. This is used when the lifecycle is requested to stop, but is already in a stopping state.
        This is a last resort to ensure that the lifecycle is stopped.
        """
        await self._set_state(LifecycleState.FAILED)
        self.error(lambda: f"Killing {self}")
        self.stop_requested = True
        self.stopped_event.set()
        # TODO: This is a hack to ensure that the process is killed.
        #       We should find a better way to do this.
        os.kill(os.getpid(), signal.SIGKILL)
        raise asyncio.CancelledError(f"Killed {self}")
