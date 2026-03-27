# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from abc import ABC
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from aiperf.common.enums import CommAddress, MessageType
from aiperf.common.environment import Environment
from aiperf.common.hooks import (
    AIPerfHook,
    Hook,
    on_init,
    on_start,
    provides_hooks,
)
from aiperf.common.messages import Message
from aiperf.common.messages.service_messages import ConnectionProbeMessage
from aiperf.common.mixins.communication_mixin import CommunicationMixin
from aiperf.common.types import MessageCallbackMapT, MessageTypeT
from aiperf.common.utils import yield_to_event_loop

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


@provides_hooks(AIPerfHook.ON_MESSAGE)
class MessageBusClientMixin(CommunicationMixin, ABC):
    """Mixin to provide message bus clients (pub and sub) for AIPerf components,
    as well as a hook to handle messages: @on_message.

    For components that also need DEALER/ROUTER communication with the controller
    (registration, heartbeats, status), use BaseComponentService which adds a
    streaming DEALER client directly.
    """

    def __init__(self, run: BenchmarkRun, **kwargs) -> None:
        super().__init__(run=run, **kwargs)
        # NOTE: The communication base class will automatically manage the pub/sub clients' lifecycle.
        self.sub_client = self.comms.create_sub_client(
            CommAddress.EVENT_BUS_PROXY_BACKEND
        )
        self.pub_client = self.comms.create_pub_client(
            CommAddress.EVENT_BUS_PROXY_FRONTEND
        )
        self._connection_probe_event = asyncio.Event()

    @on_init
    async def _setup_on_message_hooks(self) -> None:
        """Send subscription requests for all @on_message hook decorators."""
        subscription_map: MessageCallbackMapT = {}

        def _add_to_subscription_map(hook: Hook, message_type: MessageTypeT) -> None:
            self.debug(
                lambda: f"Adding subscription for message type: '{message_type}' for hook: {hook}"
            )
            subscription_map.setdefault(message_type, []).append(hook.func)

        self.for_each_hook_param(
            AIPerfHook.ON_MESSAGE,
            self_obj=self,
            param_type=MessageTypeT,
            lambda_func=_add_to_subscription_map,
        )
        self.debug(lambda: f"Subscribing to {len(subscription_map)} topics")
        await self.sub_client.subscribe_all(subscription_map)

        await self.sub_client.subscribe(
            MessageType.CONNECTION_PROBE,
            self._process_connection_probe_message,
        )

    @on_start
    async def _wait_for_successful_probe(self) -> None:
        """Verify connectivity to the message bus via connection probes.

        Delegates to _run_connection_probes which subclasses can override
        to prepend additional phases (e.g. DEALER/ROUTER control channel).
        """
        self.debug(lambda: f"Waiting for connection probe message for {self.id}")
        await self._run_connection_probes()

    async def _run_connection_probes(self) -> None:
        """PUB/SUB self-echo probe that mitigates ZMQ's "slow joiner" problem.

        When a SUB socket connects and subscribes, there is a brief window before
        the subscription propagates to the XPUB proxy. Messages published during
        this window are silently dropped. This probe loop publishes a
        ConnectionProbeMessage and waits for it to arrive back via the
        PUB -> XSUB -> XPUB -> SUB path, retrying until the round-trip succeeds.
        A successful echo proves that subscriptions are active and the service
        will not miss broadcast messages.

        If probes fail beyond the reconnect threshold, PUB/SUB sockets are
        recreated to recover from lost subscriptions. Reconnect intervals use
        exponential backoff (capped at 2x the base interval) to avoid
        unnecessary disruption under transient load.

        Subclasses override this to prepend additional probe phases.
        """
        attempt_count = 0
        reconnect_count = 0
        elapsed_time = 0.0
        initial_warning_threshold = 5.0
        warning_interval = 10.0
        next_warning_time = initial_warning_threshold
        probe_interval = Environment.SERVICE.CONNECTION_PROBE_INTERVAL
        overall_timeout = Environment.SERVICE.CONNECTION_PROBE_TIMEOUT
        reconnect_base = Environment.SERVICE.CONNECTION_PROBE_RECONNECT_INTERVAL
        next_reconnect_time = reconnect_base
        reconnect_backoff = reconnect_base

        while not self.stop_requested:
            attempt_count += 1
            try:
                await asyncio.wait_for(
                    self._probe_and_wait_for_response(),
                    timeout=probe_interval,
                )
                if attempt_count > 2:
                    self.info(
                        f"Connection probe for {self.id} succeeded after {attempt_count} attempts "
                        f"({elapsed_time:.1f}s, {reconnect_count} reconnect(s))"
                    )
                return
            except asyncio.TimeoutError:
                elapsed_time = attempt_count * probe_interval

                if elapsed_time >= next_reconnect_time:
                    reconnect_count += 1
                    self.warning(
                        f"Recreating PUB/SUB sockets for {self.id} after {elapsed_time:.1f}s "
                        f"of failed probes (reconnect #{reconnect_count})"
                    )
                    await self._reconnect_message_bus()
                    next_reconnect_time += reconnect_backoff
                    reconnect_backoff = min(reconnect_backoff * 2, reconnect_base * 2)

                if elapsed_time >= next_warning_time:
                    self.warning(
                        f"Connection probe for {self.id} still waiting after {elapsed_time:.1f}s "
                        f"({attempt_count} attempts, {reconnect_count} reconnect(s)). "
                        f"Check that ZMQ message bus is running "
                        f"and accessible at pub={self.pub_client.address} "
                        f"sub={self.sub_client.address}. Will timeout after {overall_timeout}s."
                    )
                    next_warning_time += warning_interval

                if elapsed_time >= overall_timeout:
                    raise TimeoutError(
                        f"Connection probe for {self.id} timed out after {elapsed_time:.1f}s "
                        f"({attempt_count} attempts, {reconnect_count} reconnect(s)). "
                        f"Addresses: pub={self.pub_client.address} sub={self.sub_client.address}"
                    ) from None

                self.debug(
                    "Timeout waiting for connection probe message, sending another probe"
                )
                await yield_to_event_loop()

    async def _reconnect_message_bus(self) -> None:
        """Recreate PUB/SUB sockets and resubscribe to recover from broken connections."""
        try:
            await self.pub_client._recreate_socket()
            await self.sub_client._recreate_socket()
            await self.sub_client._resubscribe_all()
        except Exception as e:
            self.warning(
                f"Failed to reconnect message bus for {self.id}, "
                f"will retry on next probe cycle: {e!r}"
            )

    async def _process_connection_probe_message(
        self, message: ConnectionProbeMessage
    ) -> None:
        """Process a connection probe message."""
        if message.service_id != self.id:
            return
        self.debug(lambda: f"Received connection probe message: {message}")
        self._connection_probe_event.set()

    async def _probe_and_wait_for_response(self) -> None:
        """Wait for a connection probe message."""
        self._connection_probe_event.clear()
        await self.publish(ConnectionProbeMessage(service_id=self.id))
        await self._connection_probe_event.wait()

    async def subscribe(
        self,
        message_type: MessageTypeT,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to a specific message type."""
        await self.sub_client.subscribe(message_type, callback)

    async def subscribe_all(
        self,
        message_callback_map: MessageCallbackMapT,
    ) -> None:
        """Subscribe to all message types in the map."""
        await self.sub_client.subscribe_all(message_callback_map)

    async def publish(self, message: Message) -> None:
        """Publish a message. The message will be routed automatically based on the message type."""
        await self.pub_client.publish(message)
