# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import os
import traceback
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.base_service import BaseService
from aiperf.common.control_structs import (
    Command,
    CommandAck,
    CommandErr,
    CommandOk,
    Heartbeat,
    MemoryReport,
    Registration,
    RegistrationAck,
    ServiceBoundMessage,
    StatusUpdate,
)
from aiperf.common.enums import CommAddress, CommandType, LifecycleState
from aiperf.common.environment import Environment
from aiperf.common.hooks import (
    AIPerfHook,
    Hook,
    background_task,
    on_init,
    on_state_change,
    on_stop,
)
from aiperf.common.memory_tracker import MemoryPhase

if TYPE_CHECKING:
    from aiperf.common.memory_tracker import MemoryReading
    from aiperf.config import BenchmarkRun


class BaseComponentService(BaseService):
    """Base class for all Component services.

    This class provides a common interface for all Component services in the AIPerf
    framework such as the Timing Manager, Dataset Manager, etc.

    It extends the BaseService by adding:
    - A streaming DEALER client for direct DEALER/ROUTER communication with the
      SystemController (registration, heartbeat, status updates) using msgspec/msgpack.
    - Heartbeat and registration functionality.
    - Memory tracking at init/stop/post-config.

    The control DEALER lives outside the comms lifecycle so it is available
    before comms.start() -- registration needs it during connection probes.
    """

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
        self._api_port = api_port
        self._registration_ack_event: asyncio.Event | None = None
        self._pending_registration_rid: str | None = None

        from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient

        control_address = self.comms.get_address(CommAddress.CONTROL)
        self.control_client = ZMQStreamingDealerClient(
            address=control_address,
            identity=self.id,
            bind=False,
            decode_type=ServiceBoundMessage,
        )

    # -------------------------------------------------------------------------
    # Lifecycle: control channel DEALER
    # -------------------------------------------------------------------------

    @on_init
    async def _init_control_client(self) -> None:
        """Initialize and start the DEALER control client independently of comms."""
        if not self._uses_controller_control_channel():
            return
        self.control_client.register_receiver(self._handle_control_command)
        await self.control_client.initialize()
        await self.control_client.start()

    @on_stop
    async def _stop_control_client(self) -> None:
        """Stop the DEALER control client.

        Cancels background tasks that use the control channel (heartbeats,
        early-heartbeat) before closing the socket so they don't attempt to
        send on a closed client. Idempotent with _stop_all_tasks.
        """
        if not self._uses_controller_control_channel():
            return
        early_task = getattr(self, "_early_heartbeat_task", None)
        if early_task is not None and not early_task.done():
            early_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await early_task
        await self.cancel_all_tasks()
        await self.control_client.stop()

    # -------------------------------------------------------------------------
    # Registration & connection probes
    # -------------------------------------------------------------------------

    def _uses_controller_control_channel(self) -> bool:
        """Return whether this service should use the controller control channel."""
        return True

    def _make_registration(self) -> Registration:
        """Build a Registration struct for this service.

        Includes Kubernetes pod metadata (pod_name, pod_index) when running
        in a K8s pod, populated from environment variables.
        """
        return Registration(
            sid=self.service_id,
            rid=uuid.uuid4().hex,
            stype=str(self.service_type),
            state=str(self.state),
            pod_name=os.environ.get("HOSTNAME"),
            pod_index=os.environ.get("AIPERF_POD_INDEX"),
        )

    async def _run_connection_probes(self) -> None:
        """DEALER/ROUTER registration then PUB/SUB self-echo.

        Registration doubles as the connection probe: a successful round-trip
        proves the controller is reachable. Registration is idempotent.
        After probes succeed, publish the startup memory captured at init.
        """
        if not self._uses_controller_control_channel():
            await super()._run_connection_probes()
            return

        await self._register_until_ack(
            send_interval=Environment.SERVICE.REGISTRATION_INTERVAL,
            overall_timeout=Environment.SERVICE.REGISTRATION_TIMEOUT,
            initial_warning_threshold=5.0,
            warning_interval=10.0,
        )

        # Start sending keepalive heartbeats immediately after registration.
        # Background tasks don't start until RUNNING state, but the ZMQ TCP
        # connection can be dropped by K8s networking during the STARTING phase
        # (configuration wait). This early heartbeat keeps the connection alive
        # until the regular heartbeat background task takes over.
        self._early_heartbeat_task = asyncio.create_task(self._early_heartbeat_loop())

        await super()._run_connection_probes()

        # Send the startup memory reading captured during @on_init.
        reading = getattr(self, "_startup_memory_reading", None)
        if reading is None:
            return
        await self._send_memory_report(MemoryPhase.STARTUP, reading)

    async def _register_until_ack(
        self,
        *,
        send_interval: float,
        overall_timeout: float,
        initial_warning_threshold: float,
        warning_interval: float,
    ) -> None:
        """Fire-and-forget Registration requests at ``send_interval`` until ack.

        Uses an asyncio.Event set by _handle_control_command when a
        RegistrationAck arrives, so we resolve within milliseconds of the
        controller's response instead of waiting for a request timeout.
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
                # ignore stale acks from prior attempts (which would otherwise
                # unblock wait_for on the current attempt without it actually
                # being acknowledged).
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
                    return
                except asyncio.TimeoutError:
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
        """Send heartbeats during STARTING phase to keep ZMQ connections alive.

        Runs from registration until background tasks start (RUNNING state).
        The regular ``_heartbeat_task`` takes over once the service is running.
        """
        if not self._uses_controller_control_channel():
            return
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
            pass
        except Exception:
            pass

    @background_task(interval=Environment.SERVICE.HEARTBEAT_INTERVAL, immediate=True)
    async def _heartbeat_task(self) -> None:
        """Send a heartbeat to the system controller over the DEALER/ROUTER channel."""
        if not self._uses_controller_control_channel():
            return
        # Cancel the early heartbeat loop — the regular one takes over now
        early_task = getattr(self, "_early_heartbeat_task", None)
        if early_task and not early_task.done():
            early_task.cancel()
        await self.control_client.send(
            Heartbeat(
                sid=self.service_id,
                stype=str(self.service_type),
                state=str(self.state),
            )
        )

    @on_state_change
    async def _on_state_change(
        self, old_state: LifecycleState, new_state: LifecycleState
    ) -> None:
        """Send state change to the system controller over the DEALER/ROUTER channel."""
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
    # Memory tracking
    # -------------------------------------------------------------------------

    async def _send_memory_report(
        self,
        phase: MemoryPhase,
        reading: MemoryReading | None = None,
    ) -> None:
        """Send own memory stats to the controller via the control channel."""
        if not self._uses_controller_control_channel():
            return
        if reading is None:
            from aiperf.common.memory_tracker import MemorySnapshot

            snap = MemorySnapshot(
                pid=os.getpid(), label=self.service_id, group=str(self.service_type)
            )
            reading = snap.capture(phase)
        if reading is None:
            return
        await self.control_client.send(
            MemoryReport(
                sid=self.service_id,
                stype=str(self.service_type),
                pid=os.getpid(),
                phase=str(phase),
                pss_bytes=reading.pss,
                rss_bytes=reading.rss,
                uss_bytes=reading.uss,
                shared_bytes=reading.shared,
            )
        )

    @on_init
    async def _capture_memory_at_init(self) -> None:
        """Capture own memory reading as early as possible."""
        from aiperf.common.memory_tracker import read_memory_self

        self._startup_memory_reading = read_memory_self()

    @on_stop
    async def _report_memory_at_stop(self) -> None:
        """Report own memory before shutdown."""
        await self._send_memory_report(MemoryPhase.SHUTDOWN)

    # -------------------------------------------------------------------------
    # Command dispatch (DEALER/ROUTER control channel)
    # -------------------------------------------------------------------------

    async def _handle_control_command(self, message: Any) -> None:
        """Handle incoming messages from the controller via DEALER.

        Only Command structs are dispatched to @on_command hooks. Other message
        types (RegistrationAck, CommandAck/Ok/Err) are handled by the
        _pending_requests matching in the DEALER receive loop.
        """
        if isinstance(message, RegistrationAck):
            if (
                self._registration_ack_event is not None
                and self._pending_registration_rid is not None
                and message.rid == self._pending_registration_rid
            ):
                self._registration_ack_event.set()
            return
        if not isinstance(message, Command):
            self.warning(
                f"Unexpected message type on control channel: {type(message).__name__}"
            )
            return

        for hook in self.get_hooks(AIPerfHook.ON_COMMAND):
            resolved = hook.resolve_params(self)
            if isinstance(resolved, Iterable) and message.cmd in resolved:
                await self._execute_control_command(message, hook)
                return

        self.debug(f"No handler for command {message.cmd}, sending ack")
        await self.control_client.send(CommandAck(cid=message.cid, sid=self.service_id))

    async def _execute_control_command(self, message: Command, hook: Hook) -> None:
        """Execute an @on_command hook and send the response via DEALER."""
        try:
            result = await hook.func(message)
            if result is None:
                await self.control_client.send(
                    CommandAck(cid=message.cid, sid=self.service_id)
                )
            else:
                payload = self._serialize_command_result(result)
                await self.control_client.send(
                    CommandOk(cid=message.cid, sid=self.service_id, payload=payload)
                )
        except Exception as e:
            tb = traceback.format_exc()
            self.error(f"Failed to handle command {message.cmd}: {e}")
            await self.control_client.send(
                CommandErr(
                    cid=message.cid,
                    sid=self.service_id,
                    error=str(e),
                    traceback=tb,
                )
            )

        if message.cmd == CommandType.PROFILE_CONFIGURE:
            await self._send_memory_report(MemoryPhase.POST_CONFIG)

    @staticmethod
    def _serialize_command_result(result: Any) -> bytes:
        """Serialize a command handler result to bytes for CommandOk payload."""
        if isinstance(result, bytes):
            return result
        from pydantic import BaseModel

        if isinstance(result, BaseModel):
            return result.model_dump_json().encode()
        return orjson.dumps(result)
