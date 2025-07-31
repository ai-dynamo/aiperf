# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from abc import ABC
from collections.abc import Iterable

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import DEFAULT_COMMAND_RESPONSE_TIMEOUT
from aiperf.common.enums import MessageType
from aiperf.common.hooks import (
    AIPerfHook,
    Hook,
    on_message,
    provides_hooks,
)
from aiperf.common.messages import (
    CommandAcknowledgedResponse,
    CommandErrorResponse,
    CommandMessage,
    CommandResponse,
    CommandSuccessResponse,
    CommandUnhandledResponse,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import ErrorDetails


@provides_hooks(AIPerfHook.ON_COMMAND)
class CommandHandlerMixin(MessageBusClientMixin, ABC):
    """Mixin to provide command handling functionality to a service.

    This mixin is used by the BaseService class, and is not intended to be used directly.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str,
        **kwargs,
    ) -> None:
        self.service_config = service_config
        self.user_config = user_config
        self.service_id = service_id

        self._processed_command_ids: set[str] = set()
        self._single_response_futures: dict[str, asyncio.Future[CommandResponse]] = {}
        self._multi_response_futures: dict[
            str, dict[str, asyncio.Future[CommandResponse]]
        ] = {}

        super().__init__(
            service_config=self.service_config,
            user_config=self.user_config,
            **kwargs,
        )

    # NOTE: Keep in mind that subscriptions in ZMQ are prefix based wildcards, so the unique portion has to come first.
    @on_message(
        lambda self: {
            MessageType.COMMAND,
            f"{self.service_type}.{MessageType.COMMAND}",
            f"{self.service_id}.{MessageType.COMMAND}",
        }
    )
    async def _process_command_message(self, message: CommandMessage) -> None:
        """Process a command message received from the controller, and forward it to the appropriate handler.
        Wait for the handler to complete and publish the response, or handle the error and publish the failure response.
        """
        self.debug(lambda: f"Received command message: {message}")
        if message.command_id in self._processed_command_ids:
            self.debug(
                lambda: f"Received duplicate command message: {message}. Ignoring."
            )
            await self.publish(
                CommandAcknowledgedResponse.from_command_message(
                    message, self.service_id
                )
            )
            return

        self._processed_command_ids.add(message.command_id)

        if message.service_id == self.service_id:
            self.debug(
                lambda: f"Received broadcast command message from self: {message}. Ignoring."
            )
            return

        self.debug(lambda: f"Received command message: {message}")

        # Go through the hooks and find the first one that matches the command type.
        # Currently, we only support a single handler per command type, so we break out of the loop after the first one.
        # TODO: Do we want/need to add support for multiple handlers per command type?
        for hook in self.get_hooks(AIPerfHook.ON_COMMAND):
            if isinstance(hook.params, Iterable) and message.command in hook.params:
                await self._execute_command_hook(message, hook)
                # Only one handler per command type, so return after the first handler.
                return

        # If we reach here, no handler was found for the command, so we publish an unhandled response.
        await self.publish(
            CommandUnhandledResponse.from_command_message(message, self.service_id)
        )

    # NOTE: Keep in mind that subscriptions in ZMQ are prefix based wildcards, so the unique portion has to come first.
    @on_message(
        lambda self: {
            f"{self.service_id}.{MessageType.COMMAND_RESPONSE}",
        }
    )
    async def _process_command_response_message(self, message: CommandResponse) -> None:
        self.debug(lambda: f"Received command response message: {message}")
        if message.command_id in self._single_response_futures:
            self._single_response_futures[message.command_id].set_result(message)
        elif message.command_id in self._multi_response_futures:
            self._multi_response_futures[message.command_id][
                message.service_id
            ].set_result(message)

    async def send_command_and_wait_for_response(
        self, message: CommandMessage, timeout: float = DEFAULT_COMMAND_RESPONSE_TIMEOUT
    ) -> CommandResponse | ErrorDetails:
        """Send a single command message to a single service and wait for the response."""
        future = asyncio.Future[CommandResponse]()
        self._single_response_futures[message.command_id] = future
        await self.publish(message)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as e:
            return ErrorDetails.from_exception(e)
        finally:
            del self._single_response_futures[message.command_id]

    async def send_command_and_wait_for_all_responses(
        self,
        command: CommandMessage,
        service_ids: list[str],
        timeout: float = DEFAULT_COMMAND_RESPONSE_TIMEOUT,
    ) -> list[CommandResponse | ErrorDetails]:
        """Send a single command message to multiple services and wait for the responses from all services."""
        self._multi_response_futures[command.command_id] = {
            service_id: asyncio.Future[CommandResponse]() for service_id in service_ids
        }
        await self.publish(command)
        try:
            return await asyncio.wait_for(
                asyncio.gather(
                    *[
                        future
                        for future in self._multi_response_futures[
                            command.command_id
                        ].values()
                    ]
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as e:
            return [ErrorDetails.from_exception(e) for _ in range(len(service_ids))]
        finally:
            del self._multi_response_futures[command.command_id]

    async def _execute_command_hook(self, message: CommandMessage, hook: Hook) -> None:
        """Execute a command hook."""
        try:
            response = await hook.func(message)
            if response is None:
                # If there is no data to send back, just send an acknowledged response.
                await self.publish(
                    CommandAcknowledgedResponse.from_command_message(
                        message, self.service_id
                    )
                )
                return

            await self.publish(
                CommandSuccessResponse.from_command_message(
                    message, self.service_id, response
                )
            )
        except Exception as e:
            self.exception(
                f"Failed to handle command {message.command} with hook {hook}: {e}"
            )
            await self.publish(
                CommandErrorResponse.from_command_message(
                    message,
                    self.service_id,
                    ErrorDetails.from_exception(e),
                )
            )
