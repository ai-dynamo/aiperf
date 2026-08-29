# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from typing import TYPE_CHECKING, ClassVar

import zmq

from aiperf.common.base_service import BaseService
from aiperf.common.control_structs import (
    Heartbeat,
    Registration,
    RegistrationAck,
    ServiceBoundMessage,
    StatusUpdate,
)
from aiperf.common.enums import CommAddress, CommandType, LifecycleState
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import (
    background_task,
    on_command,
    on_init,
    on_start,
    on_state_change,
    on_stop,
)
from aiperf.common.messages import CommandMessage

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class BaseComponentService(BaseService):
    """Base class for all Component services.

    This class provides a common interface for all Component services in the AIPerf
    framework such as the Timing Manager, Dataset Manager, etc.

    It extends the BaseService by adding a streaming DEALER client to the
    SystemController's control ROUTER, over which registration, heartbeats and
    lifecycle status updates flow. Commands still ride the pub/sub bus.
    """

    # Capability tags advertised to the SystemController at registration. Result
    # producers override this (e.g. ``make_result_producer_capability("telemetry")``)
    # so the controller can join their result on shutdown.
    extra_capabilities: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        api_port: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )
        # Explicit port override supplied by the launcher (the K8s service
        # manager assigns ports per-pod); None means "fall back to config /
        # environment defaults".
        self._api_port = api_port

        self._registration_ack_event: asyncio.Event | None = None
        self._pending_registration_rid: str | None = None
        self._registration_complete = False
        self._early_heartbeat_task: asyncio.Task | None = None

        # Routed through self.comms so FakeCommunication can substitute an
        # in-process dealer instead of binding a real socket on fake://.
        self.control_client = self.comms.create_streaming_dealer_client(
            address=CommAddress.CONTROL,
            identity=self.id,
            bind=False,
            decode_type=ServiceBoundMessage,
        )

    # -------------------------------------------------------------------------
    # Lifecycle: control channel DEALER
    # -------------------------------------------------------------------------

    def _uses_controller_control_channel(self) -> bool:
        """Return whether this service should talk to the controller's control ROUTER."""
        return True

    @on_init
    async def _init_control_client(self) -> None:
        """Attach the receiver before the DEALER's receive loop is running.

        Deliberately does NOT initialize or start the client: unlike the
        controller's ROUTER (created with ``attach_lifecycle=False``), the DEALER
        is a lifecycle child of ``self.comms``, which initializes it in the
        ``_initialize_children`` hook that runs ahead of this one and starts it
        in ``_start_children`` before any of this class's ``@on_start`` hooks.
        Driving it a second time here would raise ``InvalidStateError``.
        """
        if not self._uses_controller_control_channel():
            return
        self.control_client.register_receiver(self._handle_control_command)

    @on_stop
    async def _stop_control_client(self) -> None:
        """Cancel the early-heartbeat loop before the DEALER socket goes away.

        The socket itself is closed by ``comms.stop()``, which owns it, and every
        ``@background_task`` is cancelled by ``AIPerfLifecycleMixin._stop_all_tasks``.
        Neither covers ``_early_heartbeat_task``: it is a bare ``asyncio.create_task``
        that the task manager never learned about, so nothing else would ever
        cancel it.
        """
        if not self._uses_controller_control_channel():
            return
        early_task = self._early_heartbeat_task
        if early_task is not None and not early_task.done():
            early_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                try:
                    await early_task
                except Exception as e:  # noqa: BLE001 - shutdown boundary; a bug here must be visible, not fatal
                    # Discarding this would hide a genuine defect in the loop
                    # behind a clean shutdown.
                    self.debug(f"Early heartbeat task failed on shutdown: {e!r}")
        self._early_heartbeat_task = None

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def _make_registration(self) -> Registration:
        """Build a Registration struct for this service.

        Includes Kubernetes pod metadata (pod_name, pod_index) when running in a
        K8s pod, populated from environment variables.
        """
        return Registration(
            sid=self.service_id,
            rid=uuid.uuid4().hex,
            stype=str(self.service_type),
            state=str(self.state),
            pod_name=os.environ.get("HOSTNAME"),
            pod_index=os.environ.get("AIPERF_POD_INDEX"),
            capabilities=tuple(self.extra_capabilities),
        )

    @on_start
    async def _register_service_on_start(self) -> None:
        """Register with the SystemController over the control channel.

        Runs after ``MessageBusClientMixin._wait_for_successful_probe`` (that
        hook is declared further up the MRO, and hooks run base-class-first), so
        by the time a service registers, both planes are proven: the probe
        proves the event bus, the registration ack proves the control channel.
        """
        if not self._uses_controller_control_channel():
            return
        await self._register_until_ack(
            send_interval=Environment.SERVICE.REGISTRATION_INTERVAL,
            overall_timeout=Environment.SERVICE.REGISTRATION_TIMEOUT,
            initial_warning_threshold=5.0,
            warning_interval=10.0,
        )

        # Start sending keepalive heartbeats immediately after registration. The
        # regular heartbeat background task does not fire until one full
        # interval has elapsed, but the ZMQ TCP connection can be dropped by K8s
        # networking during the STARTING phase (configuration wait), so this
        # keeps the connection alive until the background task takes over.
        self._early_heartbeat_task = asyncio.create_task(self._early_heartbeat_loop())

    async def _register_until_ack(
        self,
        *,
        send_interval: float,
        overall_timeout: float,
        initial_warning_threshold: float,
        warning_interval: float,
    ) -> None:
        """Fire Registration requests at ``send_interval`` until acked.

        Uses an asyncio.Event set by ``_handle_control_command`` when a matching
        RegistrationAck arrives, so we resolve within milliseconds of the
        controller's response instead of waiting out a request timeout.
        """
        self._registration_ack_event = asyncio.Event()
        attempt_count = 0
        elapsed_time = 0.0
        next_warning_time = initial_warning_threshold

        try:
            while not self.stop_requested:
                attempt_count += 1
                registration = self._make_registration()
                # Track the rid of the in-flight attempt so the ack handler can
                # ignore stale acks from prior attempts, which would otherwise
                # unblock the wait on the current attempt without it actually
                # having been acknowledged.
                self._pending_registration_rid = registration.rid
                self._registration_ack_event.clear()
                await self.control_client.send(registration)

                try:
                    await asyncio.wait_for(
                        self._registration_ack_event.wait(),
                        timeout=send_interval,
                    )
                    if attempt_count > 2:
                        self.info(
                            f"Registration for {self.id} succeeded after {attempt_count} attempts "
                            f"({elapsed_time:.1f}s)"
                        )
                    self._registration_complete = True
                    return
                except TimeoutError:
                    elapsed_time += send_interval

                    if elapsed_time >= next_warning_time:
                        self.warning(
                            f"Registration for {self.id} still waiting after {elapsed_time:.1f}s "
                            f"({attempt_count} attempts). Controller may not be ready yet."
                        )
                        next_warning_time += warning_interval

                    if elapsed_time >= overall_timeout:
                        raise TimeoutError(
                            f"Registration for {self.id} timed out after {elapsed_time:.1f}s "
                            f"({attempt_count} attempts)"
                        ) from None
        finally:
            self._registration_ack_event = None
            self._pending_registration_rid = None

    # -------------------------------------------------------------------------
    # Heartbeat & status
    # -------------------------------------------------------------------------

    async def _early_heartbeat_loop(self) -> None:
        """Send heartbeats during the STARTING phase to keep the DEALER alive.

        Runs from registration until the regular ``_heartbeat_task`` takes over.
        """
        try:
            while not self.stop_requested:
                await self.control_client.send(
                    Heartbeat(
                        sid=self.service_id,
                        stype=str(self.service_type),
                        state=str(self.state),
                    )
                )
                await asyncio.sleep(Environment.SERVICE.HEARTBEAT_INTERVAL)
        except asyncio.CancelledError:
            raise
        except (zmq.ZMQError, NotInitializedError) as e:
            # The socket is torn down by comms.stop(), which runs in
            # ``_stop_children`` -- ahead of ``_stop_all_tasks``, so this loop is
            # still schedulable after its socket is gone. ``BaseZMQClient._check_initialized``
            # signals that as NotInitializedError (or CancelledError, which must
            # keep propagating), and libzmq itself as ZMQError. Both are expected
            # at shutdown; swallow so the service exits cleanly.
            self.debug(f"Early heartbeat loop saw a closed socket at shutdown: {e!r}")

    @background_task(interval=Environment.SERVICE.HEARTBEAT_INTERVAL, immediate=False)
    async def _heartbeat_task(self) -> None:
        """Send a heartbeat to the system controller over the control channel.

        Background tasks are started by a hook that runs before registration, so
        this must stay silent until the controller knows about this service --
        otherwise every service logs an "unknown service" warning on the
        controller during startup.
        """
        if not self._uses_controller_control_channel():
            return
        if not self._registration_complete or self.stop_requested:
            return
        # The early loop exists only to bridge the gap until this task fires.
        early_task = self._early_heartbeat_task
        if early_task is not None and not early_task.done():
            early_task.cancel()
        try:
            await self.control_client.send(
                Heartbeat(
                    sid=self.service_id,
                    stype=str(self.service_type),
                    state=str(self.state),
                )
            )
        except (zmq.ZMQError, NotInitializedError) as e:
            # The ``stop_requested`` early-return above does NOT cover this: the
            # socket is closed by comms.stop() in ``_stop_children``, which runs
            # before ``_stop_all_tasks`` cancels this task, so a tick can pass
            # that guard and still find the socket gone. CancelledError is
            # deliberately not caught -- it must reach the task runner.
            self.debug(f"Heartbeat saw a closed socket at shutdown: {e!r}")

    @on_state_change
    async def _on_state_change(
        self, old_state: LifecycleState, new_state: LifecycleState
    ) -> None:
        """Report the new lifecycle state to the system controller.

        The controller stamps its own receipt time; the wire carries no
        timestamp, so a clock-skewed pod cannot backdate itself into staleness.
        """
        if not self._uses_controller_control_channel():
            return
        if self.stop_requested:
            return
        if not self.comms.was_initialized:
            return
        await self.control_client.send(
            StatusUpdate(
                sid=self.service_id,
                stype=str(self.service_type),
                state=str(new_state),
            )
        )

    # -------------------------------------------------------------------------
    # Inbound control channel
    # -------------------------------------------------------------------------

    async def _handle_control_command(self, message: ServiceBoundMessage) -> None:
        """Handle messages arriving from the controller on the DEALER.

        Only RegistrationAck is actionable for now; command dispatch still rides
        the pub/sub bus, so anything else is logged and dropped.
        """
        if isinstance(message, RegistrationAck):
            if (
                self._registration_ack_event is not None
                and self._pending_registration_rid is not None
                and message.rid == self._pending_registration_rid
            ):
                self._registration_ack_event.set()
            return
        self.debug(
            lambda: f"Dropping unexpected control channel message: {type(message).__name__}"
        )

    @on_command(CommandType.SHUTDOWN)
    async def _on_shutdown_command(self, message: CommandMessage) -> None:
        self.debug(f"Received shutdown command: {message}, {self.service_id}")
        try:
            await self.stop()
        except Exception as e:
            self.warning(
                f"Failed to stop service {self} ({self.service_id}) after receiving shutdown command: {e}. Killing."
            )
            await self._kill()
        raise asyncio.CancelledError()
