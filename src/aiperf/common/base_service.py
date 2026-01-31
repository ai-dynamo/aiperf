# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
import signal
import uuid
from abc import ABC

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import CommandType, LifecycleState
from aiperf.common.exceptions import ServiceError
from aiperf.common.hooks import on_command
from aiperf.common.messages import CommandMessage
from aiperf.common.messages.command_messages import CommandAcknowledgedResponse
from aiperf.common.mixins import CommandHandlerMixin
from aiperf.common.mixins.process_health_mixin import ProcessHealthMixin
from aiperf.plugin.enums import ServiceType


class BaseService(CommandHandlerMixin, ProcessHealthMixin, ABC):
    """Base class for all AIPerf services, providing common functionality for
    communication, state management, and lifecycle operations.
    This class inherits from the MessageBusClientMixin, which provides the
    message bus client functionality.

    This class provides the foundation for implementing the various services of the
    AIPerf system. Some of the abstract methods are implemented here, while others
    are still required to be implemented by derived classes.
    """

    _service_type_cache: ServiceType | None = None
    """Cached service type (class-level)."""

    @classmethod
    def get_service_type(cls) -> ServiceType:
        """The type of service this class implements.

        This is derived from _registered_name which is set when the class is
        loaded via plugins. Falls back to reverse lookup if needed.
        """
        # Check class-level cache first
        if cls._service_type_cache is not None:
            return cls._service_type_cache

        # Try _registered_name (set when loaded via plugins.get())
        registered_name = getattr(cls, "_registered_name", None)
        if not registered_name:
            # Fallback: reverse lookup in the registry for direct instantiation
            from aiperf.plugin import plugins
            from aiperf.plugin.enums import PluginType

            registered_name = plugins.find_registered_name(PluginType.SERVICE, cls)

        if registered_name:
            cls._service_type_cache = ServiceType(registered_name)
            return cls._service_type_cache

        raise AttributeError(
            f"Cannot determine service_type for {cls.__name__}. "
            f"Class must be registered in plugins.yaml or loaded via plugins."
        )

    @property
    def service_type(self) -> ServiceType:
        return self.get_service_type()

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
            if self.service_type != ServiceType.SYSTEM_CONTROLLER:
                self.error(f"Attempted to stop {self} in state {self.state}. Ignoring.")
                return
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
        # TODO: Publish a ServiceFailedMessage to the message bus to notify the system controller that the service has failed.
        self.stop_requested = True
        self.stopped_event.set()
        # TODO: This is a hack to ensure that the process is killed.
        #       We should find a better way to do this.
        os.kill(os.getpid(), signal.SIGKILL)
        raise asyncio.CancelledError(f"Killed {self}")
