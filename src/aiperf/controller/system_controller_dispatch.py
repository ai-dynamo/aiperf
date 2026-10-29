# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-channel dispatch for the SystemController.

Routes incoming :class:`ControllerBoundMessage` variants (Registration,
Heartbeat, StatusUpdate, Command, command responses) to per-variant handler
methods that mutate SystemController state.

The mixin is deliberately transport-agnostic: it is handed an already-decoded
struct plus the originating DEALER identity, so it can be driven directly from
a unit test without a socket.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import zmq
from msgspec import Struct

from aiperf.common.control_structs import (
    Command as ControlCommand,
)
from aiperf.common.control_structs import (
    CommandAck,
    CommandErr,
    CommandOk,
    CommandUnhandled,
    ControllerBoundMessage,
    Heartbeat,
    Registration,
    RegistrationAck,
    ReRegisterRequest,
    StatusUpdate,
)
from aiperf.common.enums import (
    LifecycleState,
    SystemState,
    parse_result_producer_capability,
)
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.models import ErrorDetails, ErrorDetailsCount, ServerMetricsResults
from aiperf.common.models.error_models import ExitErrorInfo
from aiperf.common.service_registry import ServiceRegistry, liveness_clock_ns
from aiperf.plugin.enums import ServiceType

if TYPE_CHECKING:
    from aiperf.common.models import ServiceRunInfo


class SystemControllerDispatchMixin:
    """Incoming control-channel message dispatch for :class:`SystemController`."""

    async def _handle_control_message(
        self, identity: str, message: ControllerBoundMessage
    ) -> Struct | None:
        """Dispatch control-channel messages from child services.

        Returns a Struct response for request-reply patterns (Registration).
        Returns None for fire-and-forget messages (Heartbeat, StatusUpdate).
        """
        match message:
            case Registration():
                return self._on_registration(message)
            case Heartbeat():
                return self._on_heartbeat(message)
            case StatusUpdate():
                return self._on_status_update(message)
            case ControlCommand():
                return await self._dispatch_control_command(identity, message)
            case CommandAck() | CommandOk() | CommandErr() | CommandUnhandled():
                # Responses to pending requests are resolved by ``cid`` matching
                # in the ROUTER receive loop before the handler is reached. If we
                # get here, nothing was waiting for this response.
                self.debug(
                    lambda: f"Unexpected command response from {identity}: "
                    f"{type(message).__name__}"
                )
                return None
        return None

    def _on_registration(self, message: Registration) -> RegistrationAck:
        self.debug(
            lambda: f"Processing registration from {message.stype} with ID: {message.sid}"
        )
        service_type = ServiceType(message.stype)
        state = LifecycleState(message.state)
        self._track_service_instance(message, service_type)

        prior_info = ServiceRegistry.get_service(message.sid)
        was_registered = ServiceRegistry.is_registered(message.sid)
        is_replacement = self._is_replacement_worker_group_registration(
            message, prior_info, was_registered
        )
        ServiceRegistry.register(
            service_id=message.sid,
            service_type=service_type,
            first_seen_ns=liveness_clock_ns(),
            state=state,
            pod_name=message.pod_name,
            pod_index=message.pod_index,
        )
        service_info = ServiceRegistry.get_service(message.sid)
        if service_info is None:
            raise RuntimeError(
                f"Service registry lost registration for '{message.sid}'"
            )

        previous = self.service_manager.service_id_map.get(message.sid)
        if previous is not None and previous.service_type != service_type:
            self.service_manager.service_map[previous.service_type] = [
                info
                for info in self.service_manager.service_map.get(
                    previous.service_type, []
                )
                if info.service_id != message.sid
            ]
        self.service_manager.service_id_map[message.sid] = service_info
        if service_type == ServiceType.WORKER:
            self._registered_worker_ids.add(message.sid)
        services = self.service_manager.service_map.setdefault(service_type, [])
        for index, existing in enumerate(services):
            if existing.service_id == message.sid:
                services[index] = service_info
                break
        else:
            services.append(service_info)

        # Join every result domain this service advertises into the shutdown
        # barrier. A telemetry/server-metrics producer may later report it is
        # disabled via its status message, which unregisters the domain again.
        for capability in message.capabilities:
            domain = parse_result_producer_capability(capability)
            if (
                domain is not None
                and message.sid not in self._retired_result_producer_ids
            ):
                self._result_join_coordinator.register(domain, message.sid)

        try:
            type_name = ServiceType(service_type).name.title().replace("_", " ")
        except (TypeError, ValueError):
            type_name = service_type
        self.info(lambda: f"Registered {type_name} (id: '{message.sid}')")

        if (
            is_replacement
            and self._system_state != SystemState.INITIALIZING
            and message.sid not in self._replacement_configuring_ids
        ):
            self._replacement_configuring_ids.add(message.sid)
            self.execute_async(self._configure_replacement_worker_group(message.sid))

        return RegistrationAck(rid=message.rid)

    def _track_service_instance(
        self, message: Registration, service_type: ServiceType
    ) -> None:
        """Distinguish reconnects from restarts that lost optional telemetry."""
        previous_instance = self._service_instance_ids.get(message.sid)
        if (
            service_type == ServiceType.SERVER_METRICS_MANAGER
            and previous_instance is not None
            and message.instance_id is not None
            and previous_instance != message.instance_id
            and self._system_state != SystemState.INITIALIZING
        ):
            self._retire_restarted_server_metrics(message.sid)
        if message.instance_id is not None:
            self._service_instance_ids[message.sid] = message.instance_id
        if message.sid not in self._retired_result_producer_ids:
            self._reaped_service_ids.discard(message.sid)

    def _retire_restarted_server_metrics(self, service_id: str) -> None:
        """A restarted optional sidecar cannot recover its predecessor's samples."""
        if service_id in self._retired_result_producer_ids:
            return
        reason = "server metrics container restarted during the benchmark"
        self._retired_result_producer_ids.add(service_id)
        self._reaped_service_ids.add(service_id)
        self._result_join_coordinator.evict_service(service_id, reason)
        if service_id in self._result_join_coordinator.evicted:
            error = ErrorDetails(
                type="ServerMetricsRestarted",
                message="The server metrics container restarted and its telemetry is incomplete. Inference request processing and final export continue.",
            )
            self.warning(error.message)
            self._exit_errors.append(
                ExitErrorInfo(
                    error_details=error,
                    operation="server_metrics_restarted",
                    service_id=service_id,
                )
            )
            self._record_missing_server_metrics(error)
        self.execute_async(self._check_and_trigger_shutdown())

    def _record_missing_server_metrics(self, error: ErrorDetails) -> None:
        """Keep optional telemetry failures visible in the primary inference export."""
        if self._server_metrics_results is None:
            self._server_metrics_results = ServerMetricsResults(
                benchmark_id=self.run.benchmark_id,
                start_ns=0,
                end_ns=0,
            )
        self._server_metrics_results.error_summary.append(
            ErrorDetailsCount(error_details=error, count=1)
        )
        self._server_metrics_result_arrived.set()

    def _is_replacement_worker_group_registration(
        self,
        message: Registration,
        prior_info: ServiceRunInfo | None,
        was_registered: bool,
    ) -> bool:
        """Return whether a JobSet replacement reused a worker-group ID."""
        if (
            ServiceType(message.stype) != ServiceType.WORKER_GROUP_MANAGER
            or prior_info is None
        ):
            return False
        if prior_info.state == LifecycleState.FAILED:
            return True
        return (
            was_registered
            and prior_info.pod_name is not None
            and message.pod_name is not None
            and prior_info.pod_name != message.pod_name
        )

    def _on_heartbeat(self, message: Heartbeat) -> None:
        """Record a heartbeat.

        The last-seen timestamp is stamped here rather than taken from the
        wire: the sender's clock may lag under Kubernetes, and
        ``get_stale_services`` compares against this process's monotonic
        clock (``liveness_clock_ns``). The registry is also the sole writer of
        ``last_seen_ns``/``state`` -- ``service_id_map`` holds the very
        ``ServiceRunInfo`` it owns, so writing here would defeat the ordering
        guard in ``update_service``.
        """
        if message.sid not in self.service_manager.service_id_map:
            self.warning(
                f"Received heartbeat from unknown service: '{message.sid}' ('{message.stype}')"
            )
            self.execute_async(self._request_reregistration(message.sid))
            return
        ServiceRegistry.update_service(
            message.sid,
            last_seen_ns=liveness_clock_ns(),
            state=LifecycleState(message.state),
            seq=message.seq,
        )

    def _on_status_update(self, message: StatusUpdate) -> None:
        """Record a lifecycle state change.

        Same stamping and ownership rules as ``_on_heartbeat``. The asymmetry in
        log level for an unknown sender is deliberate and pre-existing: a status
        update legitimately races ahead of registration, a heartbeat does not.
        """
        if message.sid not in self.service_manager.service_id_map:
            self.debug(
                lambda: f"Received status update from un-registered service: {message.sid} ({message.stype})"
            )
            self.execute_async(self._request_reregistration(message.sid))
            return
        ServiceRegistry.update_service(
            message.sid,
            last_seen_ns=liveness_clock_ns(),
            state=LifecycleState(message.state),
            seq=message.seq,
        )

    async def _request_reregistration(self, sid: str) -> None:
        """Nudge a service the registry does not recognize to re-register.

        Fires when the controller's ``ServiceRegistry``/``service_id_map``
        come up empty while a service survives -- e.g. the controller process
        restarted. Without this nudge the service would keep heartbeating
        into a controller that never re-adds it, wedging the run at whatever
        barrier counts registered services.
        """
        with contextlib.suppress(zmq.ZMQError, NotInitializedError):
            await self.control_router.send_to(sid, ReRegisterRequest(sid=sid))
