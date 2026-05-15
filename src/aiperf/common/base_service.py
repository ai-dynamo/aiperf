# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
import signal
import uuid
from abc import ABC
from typing import TYPE_CHECKING

from aiperf.common.enums import CommandType, LifecycleState
from aiperf.common.exceptions import ServiceError
from aiperf.common.hooks import on_command
from aiperf.common.messages import CommandMessage
from aiperf.common.messages.command_messages import CommandAcknowledgedResponse
from aiperf.common.messages.service_messages import BaseServiceErrorMessage
from aiperf.common.mixins import CommandHandlerMixin
from aiperf.common.mixins.health_server_mixin import HealthServerMixin
from aiperf.common.mixins.process_health_mixin import ProcessHealthMixin
from aiperf.common.models.error_models import ErrorDetails
from aiperf.plugin.enums import ServiceType

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class BaseService(HealthServerMixin, CommandHandlerMixin, ProcessHealthMixin, ABC):
    """Base class for all AIPerf services, providing common functionality for
    communication, state management, and lifecycle operations.

    Composes ``HealthServerMixin``, ``CommandHandlerMixin`` (which transitively
    pulls in ``MessageBusClientMixin``), and ``ProcessHealthMixin``. Concrete
    services subclass ``BaseComponentService`` instead; this class is reserved
    for the SystemController.
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
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        self.run = run
        self.service_id = service_id or f"{self.service_type}_{uuid.uuid4().hex[:8]}"
        super().__init__(
            service_id=self.service_id,
            id=self.service_id,
            run=self.run,
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
        """Override stop to short-circuit when a stop is already in flight.

        When ``stop_requested`` is already set, the SystemController force-kills
        (this path is the SystemController's last-resort cleanup); other
        service types log and ignore the duplicate request.
        """
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
        # Notify the system controller that this service has failed before we
        # SIGKILL ourselves. Best-effort: comms may already be torn down, in
        # which case the publish will fail and we just log and continue.
        try:
            await self.publish(
                BaseServiceErrorMessage(
                    service_id=self.service_id,
                    error=ErrorDetails(
                        message=f"Service {self.service_id} entered FAILED state and is being killed",
                    ),
                )
            )
        except Exception as publish_error:
            self.debug(
                lambda e=publish_error: f"Failed to publish BaseServiceErrorMessage during _kill (comms may already be down): {e!r}"
            )
        self.stop_requested = True
        self.stopped_event.set()
        # SIGKILL is the only reliable way to terminate the process when a
        # graceful stop has already failed; the lifecycle task may be wedged
        # inside a C extension (zmq, uvloop, orjson) where CancelledError
        # cannot interrupt. Replace this only if we add a robust abort path
        # for blocked extension calls.
        os.kill(os.getpid(), signal.SIGKILL)
        raise asyncio.CancelledError(f"Killed {self}")
