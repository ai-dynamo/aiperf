# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import os
import traceback
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

import zmq

from aiperf.common.base_service import BaseService
from aiperf.common.control_structs import (
    Command,
    CommandAck,
    CommandErr,
    CommandOk,
    CommandResponse,
    CommandUnhandled,
    Heartbeat,
    Registration,
    RegistrationAck,
    ReRegisterRequest,
    ServiceBoundMessage,
    StatusUpdate,
    encode_command_payload,
)
from aiperf.common.enums import CommAddress, LifecycleState
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import (
    AIPerfHook,
    Hook,
    background_task,
    on_init,
    on_start,
    on_state_change,
    on_stop,
)
from aiperf.common.messages import HeartbeatMessage

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class BaseComponentService(BaseService):
    """Base class for all Component services.

    This class provides a common interface for all Component services in the AIPerf
    framework such as the Timing Manager, Dataset Manager, etc.

    It extends the BaseService by adding a streaming DEALER client to the
    SystemController's control ROUTER, over which registration, heartbeats,
    lifecycle status updates and command request-reply all flow.
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

        # rid -> the waiting handshake's event. Keyed rather than held in a
        # single attribute because a controller nudge can start a second
        # handshake while the startup one is still in flight; one shared slot
        # lets the later handshake strand the earlier one's wait forever.
        self._pending_registrations: dict[str, asyncio.Event] = {}
        self._registration_complete = False
        self._reregistration_requested = False
        self._reregistration_task: asyncio.Task | None = None
        self._early_heartbeat_task: asyncio.Task | None = None
        self._control_state_seq = 0
        """Monotonic per-service counter stamped on every Heartbeat/StatusUpdate
        sent on the control channel. Never reset -- it is the ordering
        authority ``ServiceRegistry.update_service`` uses instead of
        wall-clock time, which the controller stamps at receipt and so cannot
        detect real out-of-order delivery."""

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
            try:
                await early_task
            except asyncio.CancelledError:
                # Only swallow the cancellation we asked for. If the task did
                # not end up cancelled, this CancelledError is our own caller
                # being cancelled and must propagate.
                if not early_task.cancelled():
                    raise
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
        ack_event = asyncio.Event()
        pending_rid: str | None = None
        attempt_count = 0
        elapsed_time = 0.0
        next_warning_time = initial_warning_threshold

        try:
            while not self.stop_requested:
                attempt_count += 1
                registration = self._make_registration()
                # Only the newest attempt's rid stays registered, so a late ack
                # for a prior attempt finds no entry and cannot unblock this
                # wait without the current attempt actually being acked.
                if pending_rid is not None:
                    self._pending_registrations.pop(pending_rid, None)
                pending_rid = registration.rid
                self._pending_registrations[pending_rid] = ack_event
                ack_event.clear()
                await self.control_client.send(registration)

                try:
                    await asyncio.wait_for(
                        ack_event.wait(),
                        timeout=send_interval,
                    )
                    if attempt_count > 2:
                        self.info(
                            f"Registration for {self.id} succeeded after {attempt_count} attempts "
                            f"({elapsed_time:.1f}s)"
                        )
                    self._registration_complete = True
                    self._reregistration_requested = False
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
            if pending_rid is not None:
                self._pending_registrations.pop(pending_rid, None)

    def _reregister_after_controller_nudge(self) -> None:
        """Re-run the registration handshake on a controller ``ReRegisterRequest``.

        Fires when a controller ROUTER restart comes back up with an empty
        ``ServiceRegistry`` while this service survived and kept
        heartbeating. Runs as a background task: ``_handle_control_command``
        must not block the DEALER receive loop for the whole handshake, and
        ``_register_until_ack`` retries on its own schedule until acked.

        Guarded by the in-flight task so a burst of nudges (e.g. one per
        Heartbeat and one per StatusUpdate while the controller still has no
        record of this service) does not stack concurrent handshakes. A later
        heartbeat restarts a handshake that exhausted its bounded timeout.
        """
        self._reregistration_requested = True
        task = self._reregistration_task
        if task is not None and not task.done():
            return
        # The handshake started by ``_register_service_on_start`` is awaited
        # directly, so it is not tracked by ``_reregistration_task``. A pending
        # entry is the only evidence that it -- or any other handshake -- is
        # still in flight, and one already in flight will register this service
        # anyway. ``_reregistration_requested`` stays set, so the heartbeat path
        # restarts the handshake if that one fails. This is a dedupe, not a
        # correctness guard: acks are correlated per rid, so an overlapping
        # handshake that slips past it still resolves on its own ack.
        if self._pending_registrations:
            return
        self._registration_complete = False
        self._reregistration_task = self.execute_async(
            self._register_until_ack(
                send_interval=Environment.SERVICE.REGISTRATION_INTERVAL,
                overall_timeout=Environment.SERVICE.REGISTRATION_TIMEOUT,
                initial_warning_threshold=5.0,
                warning_interval=10.0,
            )
        )

    # -------------------------------------------------------------------------
    # Heartbeat & status
    # -------------------------------------------------------------------------

    def _next_control_state_seq(self) -> int:
        """Mint the next sequence number for a Heartbeat/StatusUpdate send.

        Shared across both message types so the controller can order them
        against each other, not just within their own type.
        """
        self._control_state_seq += 1
        return self._control_state_seq

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
                        seq=self._next_control_state_seq(),
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
        """Emit this service's two heartbeats: one per transport, per consumer.

        The two sends are NOT redundant and neither may be "cleaned up" as a
        duplicate of the other. They have different consumers, different
        transports, and different failure consequences:

        - ``Heartbeat`` on the DEALER control channel is the SystemController's
          liveness watchdog. Missing it gets the service declared unhealthy and
          reaped by the controller.
        - ``HeartbeatMessage`` on the pub bus is the *credit router's* worker
          liveness clock. ``TimingManager._on_heartbeat`` feeds it to
          ``StickyCreditRouter.note_worker_heartbeat``, and
          ``WorkerLoad.last_heartbeat_ns`` has no other source once the value
          seeded at registration ages out. Missing it makes every worker look
          stale to ``evict_stale_workers`` after ``WORKER.STALE_TIME *
          WORKER.ROUTER_STALE_EVICTION_MULTIPLIER`` and fails any run longer
          than that window with ``Fatal worker loss: worker_unavailable``.

        The bus heartbeat is deliberately not gated on the control channel or on
        registration: the router's clock is independent of the controller's
        registration handshake. The control-channel heartbeat is gated, because
        background tasks start before registration and an early send makes the
        controller log an "unknown service" warning for every service at
        startup.
        """
        if self.stop_requested:
            return

        try:
            await self.publish(
                HeartbeatMessage(
                    service_id=self.service_id,
                    service_type=self.service_type,
                    state=self.state,
                )
            )
        except (zmq.ZMQError, NotInitializedError) as e:
            self.debug(f"Bus heartbeat saw a closed socket at shutdown: {e!r}")

        if not self._uses_controller_control_channel():
            return
        if not self._registration_complete:
            if self._reregistration_requested:
                self._reregister_after_controller_nudge()
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
                    seq=self._next_control_state_seq(),
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
                seq=self._next_control_state_seq(),
            )
        )

    # -------------------------------------------------------------------------
    # Outbound control channel
    # -------------------------------------------------------------------------

    async def send_command_to_controller(
        self,
        cmd: str,
        payload: bytes = b"",
        timeout: float = Environment.SERVICE.COMMAND_RESPONSE_TIMEOUT,
    ) -> CommandResponse:
        """Send a Command to the SystemController and await its response.

        Reuses the existing DEALER/ROUTER control channel -- no new socket and
        no new bind point. The controller's ``_dispatch_control_command``
        already routes inbound ``Command`` structs to its ``@on_command``
        hooks; this helper is the service-side initiator.

        The ROUTER is the only path between two non-controller services, so a
        command aimed at a *peer* must be re-fanned by a controller-side
        handler rather than sent directly (see the controller's
        ``PROFILE_COMPLETE`` relay).

        Raises:
            TimeoutError: If the controller does not answer within ``timeout``.
        """
        command = Command(cid=uuid.uuid4().hex, cmd=cmd, payload=payload)
        return await self.control_client.request(command, timeout=timeout)

    # -------------------------------------------------------------------------
    # Inbound control channel
    # -------------------------------------------------------------------------

    async def _handle_control_command(self, message: ServiceBoundMessage) -> None:
        """Handle messages arriving from the controller on the DEALER.

        RegistrationAck resolves the in-flight registration; ReRegisterRequest
        re-runs the registration handshake from scratch; Command structs are
        dispatched to this service's ``@on_command`` hooks. Command responses
        never reach here -- the DEALER receive loop resolves them against
        ``_pending_requests`` by ``cid`` first.
        """
        if isinstance(message, RegistrationAck):
            ack_event = self._pending_registrations.get(message.rid)
            if ack_event is not None:
                ack_event.set()
            return
        if isinstance(message, ReRegisterRequest):
            self._reregister_after_controller_nudge()
            return
        if not isinstance(message, Command):
            self.debug(
                lambda: f"Dropping unexpected control channel message: {type(message).__name__}"
            )
            return

        # One handler per command type: the response we send back is the hook's
        # result, and a second hook would have no way to answer.
        for hook in self.get_hooks(AIPerfHook.ON_COMMAND):
            resolved = hook.resolve_params(self)
            if isinstance(resolved, Iterable) and message.cmd in resolved:
                await self._execute_control_command(message, hook)
                return

        # CommandUnhandled rather than CommandAck: "no handler" is a failure to
        # callers like the controller's artifact-finalization barrier, and an
        # ack would report it as success.
        self.debug(lambda: f"No handler for command {message.cmd}")
        await self.control_client.send(
            CommandUnhandled(cid=message.cid, cmd=message.cmd, sid=self.service_id)
        )

    async def _execute_control_command(self, message: Command, hook: Hook) -> None:
        """Run an @on_command hook and send its response over the DEALER."""
        try:
            result = await hook.func(message)
        except asyncio.CancelledError:
            # The SHUTDOWN handler acks itself and then cancels: the service is
            # going away and cannot answer afterwards. Sending a response here
            # would duplicate that ack.
            raise
        except Exception as e:  # noqa: BLE001 - dispatcher must surface handler errors over the control channel
            self.error(f"Failed to handle command {message.cmd}: {e}")
            # The error report is itself best effort: a handler that failed
            # *because* the service is tearing down will find the DEALER already
            # closed by comms.stop(), and letting that second failure escape
            # would replace a logged handler error with an unhandled task
            # exception. CancelledError still propagates.
            with contextlib.suppress(zmq.ZMQError, NotInitializedError):
                await self.control_client.send(
                    CommandErr(
                        cid=message.cid,
                        cmd=message.cmd,
                        sid=self.service_id,
                        error=str(e),
                        traceback=traceback.format_exc(),
                    )
                )
            return

        if result is None:
            await self.control_client.send(
                CommandAck(cid=message.cid, cmd=message.cmd, sid=self.service_id)
            )
        else:
            await self.control_client.send(
                CommandOk(
                    cid=message.cid,
                    cmd=message.cmd,
                    sid=self.service_id,
                    payload=encode_command_payload(result),
                )
            )
