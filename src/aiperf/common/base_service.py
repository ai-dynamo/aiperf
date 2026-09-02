# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import uuid
from abc import ABC
from typing import TYPE_CHECKING

import zmq

from aiperf.common.constants import IS_WINDOWS
from aiperf.common.control_structs import Command, CommandAck
from aiperf.common.enums import CommandType, LifecycleState
from aiperf.common.exceptions import NotInitializedError, ServiceError
from aiperf.common.hooks import on_command
from aiperf.common.messages.service_messages import BaseServiceErrorMessage
from aiperf.common.mixins import CommandHandlerMixin
from aiperf.common.mixins.health_server_mixin import HealthServerMixin
from aiperf.common.mixins.process_health_mixin import ProcessHealthMixin
from aiperf.common.models.error_models import ErrorDetails
from aiperf.plugin.enums import ServiceType

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


def _force_exit_process(is_windows: bool) -> None:
    """Platform-conditional unconditional process exit. Extracted as a
    module-level helper so the platform branch is unit-testable without
    standing up a full ``BaseService`` instance.

    POSIX uses ``SIGKILL`` — uncatchable, exits the process immediately.
    Windows has no ``SIGKILL``, and ``SIGTERM`` is NOT a substitute:
    ``bootstrap.py`` installs ``SIG_IGN`` for SIGTERM in every child process
    (to prevent C-extension teardown SIGSEGVs), so ``os.kill(pid, SIGTERM)``
    would hit the child's own ignore-handler and be a no-op. Use
    ``os._exit`` on Windows to bypass the signal layer entirely.

    Neither branch returns; the function is effectively ``NoReturn`` at
    runtime. Annotated as ``-> None`` because Python's static-analysis
    ``NoReturn`` would force every caller into unreachable-code warnings.
    """
    if is_windows:
        os._exit(1)
    else:
        os.kill(os.getpid(), signal.SIGKILL)


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

    def _defers_broadcast_shutdown(self) -> bool:
        """Return whether this service ignores the controller's SHUTDOWN broadcast.

        Override this instead of redefining ``_on_shutdown_command``. A second
        ``@on_command(CommandType.SHUTDOWN)`` on a subclass is silently
        unreachable -- hook registration walks ``reversed(__mro__)`` and the
        dispatcher stops at the first match, so the base copy always wins. The
        API service's Kubernetes carve-out was written that way and never ran
        once between being written and 2026-08-29, when a live cluster run
        showed the API exiting five seconds after its benchmark.

        A service that defers is still acked, so the controller sees the command
        was received; it just does not stop. It must then be retired by some
        other route (the API's is ``POST /api/shutdown``).
        """
        return False

    @on_command(CommandType.SHUTDOWN)
    async def _on_shutdown_command(self, message: Command) -> None:
        """The single SHUTDOWN handler for every service.

        Deliberately defined only here. Hook registration walks
        ``reversed(__mro__)`` and the dispatcher stops at the first match, so a
        second copy on a subclass would be unreachable while still looking
        maintained -- which is exactly how this hardening previously ended up on
        a dead copy in ``BaseComponentService``. Subclasses that need to opt out
        of stopping override :meth:`_defers_broadcast_shutdown` instead.
        """
        self.debug("Received shutdown command")

        if self._defers_broadcast_shutdown():
            # Return without acking by hand: the dispatcher's success path sends
            # exactly one CommandAck for a handler that returns None. The manual
            # ack below exists only because stop() closes the DEALER before the
            # dispatcher could send it, and that does not apply here.
            self.info(
                f"{self.service_type} is ignoring the broadcast shutdown; it is "
                "retired through its own endpoint instead."
            )
            return

        # Ack before stopping: after stop() the control client is closed and the
        # dispatcher's post-return response would never reach the controller.
        # SystemController derives from BaseService directly and has no control
        # client of its own, so the attribute may legitimately be absent.
        #
        # Best effort even when present: a concurrent teardown (a failure path
        # already running comms.stop(), or a second SHUTDOWN) can close the
        # DEALER out from under this send, and a service that cannot ack must
        # still stop. Letting that raise would abort this handler before
        # stop() and leave the service to be SIGKILLed after the grace period.
        control_client = getattr(self, "control_client", None)
        if control_client is not None:
            with contextlib.suppress(zmq.ZMQError, NotInitializedError):
                await control_client.send(
                    CommandAck(cid=message.cid, cmd=message.cmd, sid=self.service_id)
                )

        try:
            await self.stop()
        except Exception as e:
            self.exception(
                f"Failed to stop service {self} ({self.service_id}) after receiving shutdown command: {e}. Killing."
            )
            await self._kill()
        # The ack above is this command's only response. Cancelling stops the
        # dispatcher before its success path sends a second one on a DEALER that
        # stop() has already closed.
        raise asyncio.CancelledError()

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
        # Graceful stop has already failed; the lifecycle task may be wedged
        # inside a C extension (zmq, uvloop, orjson) where CancelledError
        # cannot interrupt. ``_force_exit_process`` handles the platform
        # branch (SIGKILL on POSIX, ``os._exit`` on Windows — see the helper
        # for why SIGTERM is not a substitute there).
        _force_exit_process(IS_WINDOWS)
        # Unreachable: ``_force_exit_process`` terminates the process. Kept
        # so the static-analysis return type stays ``NoReturn``-shaped and
        # any future refactor that softens the kill path still surfaces a
        # CancelledError to awaiting callers.
        raise asyncio.CancelledError(f"Killed {self}")
