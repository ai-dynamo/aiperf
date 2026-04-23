# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-channel dispatch for the SystemController.

Routes incoming :class:`ControllerBoundMessage` variants (Registration,
Heartbeat, StatusUpdate, MemoryReport, Telemetry/ServerMetrics status,
Command) to per-variant handler methods that mutate SystemController state.
"""

from __future__ import annotations

import time

from msgspec import Struct

from aiperf.common.control_structs import (
    Command as ControlCommand,
)
from aiperf.common.control_structs import (
    CommandAck,
    CommandErr,
    CommandOk,
    ControllerBoundMessage,
    Heartbeat,
    MemoryReport,
    Registration,
    RegistrationAck,
    ServerMetricsStatus,
    StatusUpdate,
    TelemetryStatus,
)
from aiperf.common.enums import LifecycleState
from aiperf.common.memory_tracker import MemoryPhase, MemoryReading
from aiperf.common.service_registry import ServiceRegistry


class SystemControllerDispatchMixin:
    """Incoming control-channel message dispatch for :class:`SystemController`."""

    async def _handle_control_message(
        self, identity: str, message: ControllerBoundMessage
    ) -> Struct | None:
        """Dispatch control channel messages from child services.

        Returns a Struct response for request-reply patterns (Registration, Command).
        Returns None for fire-and-forget messages (Heartbeat, StatusUpdate, etc.).
        """
        match message:
            case Registration():
                return self._on_registration(message)
            case Heartbeat():
                return self._on_heartbeat(message)
            case StatusUpdate():
                return self._on_status_update(message)
            case MemoryReport():
                return self._on_memory_report(message)
            case TelemetryStatus():
                return await self._on_telemetry_status(message)
            case ServerMetricsStatus():
                return await self._on_server_metrics_status(message)
            case ControlCommand():
                return await self._dispatch_control_command(identity, message)
            case CommandAck() | CommandOk() | CommandErr():
                # Responses to pending requests are handled by _pending_requests
                # matching in the ROUTER receive loop. If we get here, it's
                # an unexpected response.
                self.debug(
                    f"Unexpected command response from {identity}: {type(message).__name__}"
                )
                return None

    def _on_registration(self, message: Registration) -> RegistrationAck:
        """Handle a Registration from a service; return an ACK."""
        self.debug(
            lambda: f"Received registration from {message.stype} service: {message.sid}"
        )
        already_configuring = message.sid in self._configuring_ids
        ServiceRegistry.register(
            service_id=message.sid,
            service_type=message.stype,
            first_seen_ns=time.time_ns(),
            state=LifecycleState(message.state),
            pod_name=message.pod_name,
            pod_index=message.pod_index,
        )
        self._record_declared_capacity(message, already_configuring)
        if self._auto_configure and not already_configuring:
            self._configure_scheduler.execute_async(
                self._configure_single_service(message.sid)
            )
        return RegistrationAck(rid=message.rid)

    def _record_declared_capacity(
        self, message: Registration, already_configuring: bool
    ) -> None:
        """Capture a pod's declared capacity and warn on topology mismatch."""
        if (
            already_configuring
            or message.declared_worker_capacity is None
            or message.declared_record_processor_capacity is None
        ):
            return
        self._declared_group_capacities[message.sid] = (
            message.declared_worker_capacity,
            message.declared_record_processor_capacity,
        )
        self.info(
            f"Pod '{message.sid}' reports capacity: "
            f"{message.declared_worker_capacity} workers, "
            f"{message.declared_record_processor_capacity} record processors"
        )
        if self._k8s_topology is not None and (
            message.declared_worker_capacity != self._k8s_topology.workers_per_pod
            or message.declared_record_processor_capacity
            != self._k8s_topology.record_processors_per_pod
        ):
            self.warning(
                f"Pod '{message.sid}' reported unexpected capacity "
                f"({message.declared_worker_capacity} workers, "
                f"{message.declared_record_processor_capacity} record processors). "
                f"Expected {self._k8s_topology.workers_per_pod} workers and "
                f"{self._k8s_topology.record_processors_per_pod} "
                "record processors per pod."
            )

    def _on_heartbeat(self, message: Heartbeat) -> None:
        """Handle a Heartbeat from a service; update registry last-seen."""
        self.debug(
            lambda msg=message: f"Received heartbeat from {msg.stype} service: {msg.sid}"
        )
        ServiceRegistry.update_service(
            service_id=message.sid,
            service_type=message.stype,
            last_seen_ns=time.time_ns(),
            state=LifecycleState(message.state),
        )
        return None

    def _on_status_update(self, message: StatusUpdate) -> None:
        """Handle a StatusUpdate from a service; update registry last-seen."""
        self.debug(
            lambda msg=message: f"Received status from {msg.stype} service: {msg.sid}"
        )
        ServiceRegistry.update_service(
            service_id=message.sid,
            service_type=message.stype,
            last_seen_ns=time.time_ns(),
            state=LifecycleState(message.state),
        )
        return None

    def _on_memory_report(self, message: MemoryReport) -> None:
        """Record a subprocess memory-tracker sample."""
        self._memory_tracker.record(
            label=message.sid,
            group=message.stype,
            pid=message.pid,
            phase=MemoryPhase(message.phase),
            reading=MemoryReading(
                pss=message.pss_bytes,
                rss=message.rss_bytes,
                uss=message.uss_bytes,
                shared=message.shared_bytes,
            ),
        )
        return None

    async def _on_telemetry_status(self, message: TelemetryStatus) -> None:
        """Record GPU telemetry endpoint status and possibly trigger shutdown."""
        self._telemetry_endpoints_configured = list(message.endpoints_configured)
        self._telemetry_endpoints_reachable = list(message.endpoints_reachable)
        self._should_wait_for_telemetry = message.enabled
        if not message.enabled:
            reason_msg = f" - {message.reason}" if message.reason else ""
            self.info(f"GPU telemetry disabled{reason_msg}")
            ServiceRegistry.forget(message.sid)
        else:
            self.info(
                f"GPU telemetry enabled - {len(message.endpoints_reachable)}/{len(message.endpoints_configured)} endpoint(s) reachable"
            )
        await self._check_and_trigger_shutdown()
        return None

    async def _on_server_metrics_status(self, message: ServerMetricsStatus) -> None:
        """Record server-metrics endpoint status and possibly trigger shutdown."""
        self._server_metrics_endpoints_configured = list(message.endpoints_configured)
        self._server_metrics_endpoints_reachable = list(message.endpoints_reachable)
        self._should_wait_for_server_metrics = message.enabled
        if not message.enabled:
            reason_msg = f" - {message.reason}" if message.reason else ""
            self.info(f"Server metrics disabled{reason_msg}")
            ServiceRegistry.forget(message.sid)
        else:
            self.info(
                f"Server metrics enabled - {len(message.endpoints_reachable)}/{len(message.endpoints_configured)} endpoint(s) reachable."
            )
            unreachable = set(message.endpoints_configured) - set(
                message.endpoints_reachable
            )
            if unreachable:
                self.warning(f"Unreachable endpoints: {', '.join(unreachable)}")
        await self._check_and_trigger_shutdown()
        return None
