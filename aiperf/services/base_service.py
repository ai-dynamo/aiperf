# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import signal
import uuid
from abc import ABC
from collections.abc import Iterable
from typing import Any, ClassVar

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommandResponseStatus,
    CommandType,
    LifecycleState,
    MessageType,
)
from aiperf.common.exceptions import ServiceError
from aiperf.common.hooks import (
    AIPerfHook,
    on_command,
    on_init,
    on_message,
    provides_hooks,
)
from aiperf.common.messages import CommandMessage, CommandResponseMessage
from aiperf.common.mixins import MessageBusClientMixin
from aiperf.common.models import ErrorDetails
from aiperf.common.protocols import ServiceProtocol
from aiperf.common.types import ServiceTypeT


@provides_hooks(AIPerfHook.ON_COMMAND)
@implements_protocol(ServiceProtocol)
class BaseService(MessageBusClientMixin, ABC):
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

    @on_init
    async def _subscribe_to_command_messages(self) -> None:
        """Subscribe to command messages for all services, specifically for the service type,
        and specific to our service id."""
        # NOTE: These subscriptions are in addition to the @on_message(MessageType.COMMAND) hook, but we need to
        #       have access to the service type and id, so we can't use the @on_message hook.
        subscription_map = {}
        await self.subscribe_all(subscription_map)

    def _create_command_response_message(
        self,
        message: CommandMessage,
        status: CommandResponseStatus,
        data: Any | None = None,
        error: ErrorDetails | None = None,
    ) -> CommandResponseMessage:
        """Create a command response message for the given command message. Fills in all the details based on the original command message."""
        return CommandResponseMessage(
            target_service_id=message.service_id,  # send it back to the sender
            service_id=self.service_id,
            command=message.command,
            command_id=message.command_id,
            status=status,
            data=data,
            error=error,
        )

    @on_message(
        lambda self: {
            MessageType.COMMAND,
            f"{MessageType.COMMAND}.{self.service_type}",
            f"{MessageType.COMMAND}.{self.service_id}",
        }
    )
    async def _process_command_message(self, message: CommandMessage) -> None:
        """Process a command message received from the controller, and forward it to the appropriate handler.
        Wait for the handler to complete and publish the response, or handle the error and publish the failure response.
        """
        self.debug(lambda: f"Received command message: {message.model_dump_json()}")

        # Go through the hooks and find the first one that matches the command type.
        # Currently, we only support a single handler per command type, so we break out of the loop after the first one.
        # TODO: Do we want/need to add support for multiple handlers per command type?
        for hook in self.get_hooks(AIPerfHook.ON_COMMAND):
            if isinstance(hook.params, Iterable) and message.command in hook.params:
                try:
                    response_data = await hook.func(message)
                    await self.publish(
                        self._create_command_response_message(
                            message,
                            CommandResponseStatus.SUCCESS,
                            response_data,
                        )
                    )
                except Exception as e:
                    self.exception(
                        f"Failed to handle command {message.command} with hook {hook}: {e}"
                    )
                    await self.publish(
                        self._create_command_response_message(
                            message,
                            CommandResponseStatus.FAILURE,
                            error=ErrorDetails.from_exception(e),
                        )
                    )

                # Break out of the loop after the first successful handler (only 1 handler per command)
                break

        # If we reach here, no handler was found for the command, so we publish an unhandled response.
        await self.publish(
            self._create_command_response_message(
                message,
                CommandResponseStatus.UNHANDLED,
            )
        )

    @on_command(CommandType.SHUTDOWN)
    async def _on_shutdown_command(self, message: CommandMessage) -> None:
        self.debug(f"Received shutdown command from {message.service_id}")
        # Send an acknowledged response back to the sender, because we won't be able to send it after we stop.
        await self.publish(
            self._create_command_response_message(
                message,
                CommandResponseStatus.ACKNOWLEDGED,
            )
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
