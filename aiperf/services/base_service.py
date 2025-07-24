# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import uuid
from abc import ABC
from typing import Any, ClassVar

from aiperf.common.config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums.command_enums import CommandResponseStatus, CommandType
from aiperf.common.enums.message_enums import MessageType
from aiperf.common.exceptions import (
    ServiceError,
)
from aiperf.common.hooks import (
    AIPerfHook,
    CommandHookParams,
    on_init,
    on_message,
    provides_hooks,
)
from aiperf.common.messages.command_messages import (
    CommandMessage,
    CommandResponseMessage,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.types import MessageCallbackMapT, ServiceTypeT


@provides_hooks(AIPerfHook.COMMAND_HANDLER)
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
        # NOTE: These subscriptions are in addition to the @on_message hook, but we need to
        #       have access to the service type and id, so we can't use the @on_message hook.
        subscription_map: MessageCallbackMapT = {
            f"{MessageType.COMMAND}.{self.service_type}": self._process_command_message,
            f"{MessageType.COMMAND}.{self.service_id}": self._process_command_message,
        }
        await self.subscribe_all(subscription_map)

    @on_message(MessageType.COMMAND)
    async def _process_command_message(self, message: CommandMessage) -> None:
        """Process a command message received from the controller, and forward it to the appropriate handler.
        Wait for the handler to complete and publish the response, or handle the error and publish the failure response.
        """
        self.debug(lambda: f"Received command message: {message.model_dump_json()}")

        if message.command == CommandType.SHUTDOWN:
            self.debug("Received shutdown command")
            await self.pub_client.publish(
                CommandResponseMessage(
                    service_id=self.service_id,
                    command=message.command,
                    command_id=message.command_id,
                    status=CommandResponseStatus.ACKNOWLEDGED,
                )
            )

        response_status = CommandResponseStatus.SUCCESS
        response_error: ErrorDetails | None = None
        response_data: Any | None = None

        for hook in self.get_hooks(AIPerfHook.COMMAND_HANDLER):
            if (
                isinstance(hook.params, CommandHookParams)
                and message.command in hook.params.command_types
            ):
                try:
                    response_data = await hook.func(message)
                except Exception as e:
                    self.exception(
                        f"Failed to handle command {message.command} with hook {hook}: {e}"
                    )
                    response_status = CommandResponseStatus.FAILURE
                    response_error = ErrorDetails.from_exception(e)

                # Break out of the loop after the first successful handler (only 1 handler per command)
                break

        await self.pub_client.publish(
            CommandResponseMessage(
                service_id=self.service_id,
                command=message.command,
                command_id=message.command_id,
                status=response_status,
                data=response_data,
                error=response_error,
            ),
        )
