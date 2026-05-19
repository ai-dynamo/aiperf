# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`aiperf.controller.system_controller_dispatch`.

Focuses on:
- _handle_control_message variant routing (Registration / Heartbeat / StatusUpdate
  / MemoryReport / TelemetryStatus / ServerMetricsStatus / Command / response stragglers)
- Per-handler state mutations on SystemController
- ServiceRegistry side effects (register / update / forget)
- _record_declared_capacity topology mismatch warnings
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.control_structs import (
    Command as ControlCommand,
)
from aiperf.common.control_structs import (
    CommandAck,
    CommandErr,
    CommandOk,
    Heartbeat,
    MemoryReport,
    Registration,
    RegistrationAck,
    ServerMetricsStatus,
    StatusUpdate,
    TelemetryStatus,
)
from aiperf.common.enums import LifecycleState, SystemState
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller.system_controller import SystemController

# ============================================================
# Registration
# ============================================================


class TestOnRegistration:
    """Registration messages register the service and return RegistrationAck."""

    async def test_returns_registration_ack_with_matching_rid(
        self, system_controller: SystemController
    ) -> None:
        msg = Registration(
            sid="svc1", rid="rid-abc", stype="timing_manager", state="running"
        )
        result = await system_controller._handle_control_message("id_0", msg)

        assert isinstance(result, RegistrationAck)
        assert result.rid == "rid-abc"
        assert ServiceRegistry.is_registered("svc1")

    async def test_already_configuring_skips_capacity_record(
        self, system_controller: SystemController
    ) -> None:
        system_controller._configuring_ids.add("svc1")
        msg = Registration(
            sid="svc1",
            rid="r",
            stype="worker_group_manager",
            state="running",
            num_workers=4,
            num_record_processors=1,
        )
        await system_controller._handle_control_message("id_0", msg)

        assert "svc1" not in system_controller._declared_group_capacities

    async def test_capacity_recorded_for_new_pod(
        self, system_controller: SystemController
    ) -> None:
        msg = Registration(
            sid="wpm0",
            rid="r",
            stype="worker_group_manager",
            state="running",
            num_workers=4,
            num_record_processors=2,
        )
        await system_controller._handle_control_message("id_0", msg)

        assert system_controller._declared_group_capacities["wpm0"] == (4, 2)

    async def test_no_capacity_fields_no_record(
        self, system_controller: SystemController
    ) -> None:
        msg = Registration(sid="tm0", rid="r", stype="timing_manager", state="running")
        await system_controller._handle_control_message("id_0", msg)

        assert "tm0" not in system_controller._declared_group_capacities

    async def test_capacity_mismatch_warns(
        self, system_controller: SystemController
    ) -> None:
        topology = MagicMock()
        topology.workers_per_pod = 4
        topology.record_processors_per_pod = 1
        system_controller._k8s_topology = topology

        msg = Registration(
            sid="wpm0",
            rid="r",
            stype="worker_group_manager",
            state="running",
            num_workers=3,
            num_record_processors=2,
        )
        with patch.object(system_controller, "warning") as mock_warn:
            await system_controller._handle_control_message("id_0", msg)

        mock_warn.assert_called_once()
        assert "wpm0" in mock_warn.call_args[0][0]

    async def test_capacity_match_no_warn(
        self, system_controller: SystemController
    ) -> None:
        topology = MagicMock()
        topology.workers_per_pod = 4
        topology.record_processors_per_pod = 1
        system_controller._k8s_topology = topology

        msg = Registration(
            sid="wpm0",
            rid="r",
            stype="worker_group_manager",
            state="running",
            num_workers=4,
            num_record_processors=1,
        )
        with patch.object(system_controller, "warning") as mock_warn:
            await system_controller._handle_control_message("id_0", msg)

        mock_warn.assert_not_called()

    async def test_auto_configure_schedules_configure(
        self, system_controller: SystemController
    ) -> None:
        scheduler = MagicMock()
        system_controller._auto_configure = True
        system_controller._configure_scheduler = scheduler
        system_controller._configure_single_service = MagicMock(  # type: ignore[method-assign]
            return_value="coro"
        )

        msg = Registration(sid="svcX", rid="r", stype="timing_manager", state="running")
        await system_controller._handle_control_message("id_0", msg)

        scheduler.execute_async.assert_called_once_with("coro")
        system_controller._configure_single_service.assert_called_once_with("svcX")

    async def test_auto_configure_skipped_when_already_configuring(
        self, system_controller: SystemController
    ) -> None:
        scheduler = MagicMock()
        system_controller._auto_configure = True
        system_controller._configure_scheduler = scheduler
        system_controller._configuring_ids.add("svc-existing")

        msg = Registration(
            sid="svc-existing", rid="r", stype="timing_manager", state="running"
        )
        await system_controller._handle_control_message("id_0", msg)

        scheduler.execute_async.assert_not_called()

    @pytest.mark.parametrize("replacement_pod_name", ["old-pod", "new-pod"])
    async def test_replacement_worker_group_reconfigures_during_profiling(
        self, system_controller: SystemController, replacement_pod_name: str
    ) -> None:
        scheduler = MagicMock()
        system_controller._auto_configure = False
        system_controller._system_state = SystemState.PROFILING
        system_controller._configure_scheduler = scheduler
        system_controller._configured_ids.add("worker_group_manager_0")
        system_controller._configure_single_service = MagicMock(  # type: ignore[method-assign]
            return_value="coro"
        )
        ServiceRegistry.register(
            service_id="worker_group_manager_0",
            service_type="worker_group_manager",
            first_seen_ns=1,
            state=LifecycleState.RUNNING,
            pod_name="old-pod",
            pod_index="0",
        )

        msg = Registration(
            sid="worker_group_manager_0",
            rid="r",
            stype="worker_group_manager",
            state="running",
            pod_name=replacement_pod_name,
            pod_index="0",
        )
        await system_controller._handle_control_message("id_0", msg)

        scheduler.execute_async.assert_called_once_with("coro")
        system_controller._configure_single_service.assert_called_once_with(
            "worker_group_manager_0"
        )
        assert "worker_group_manager_0" not in system_controller._configured_ids
        assert (
            ServiceRegistry.get_service("worker_group_manager_0").pod_name
            == replacement_pod_name
        )


# ============================================================
# Heartbeat / StatusUpdate
# ============================================================


class TestOnHeartbeatAndStatusUpdate:
    """Heartbeat/StatusUpdate update last-seen on already-registered services."""

    @pytest.mark.parametrize(
        "msg_factory",
        [
            param(
                lambda: Heartbeat(sid="svc1", stype="timing_manager", state="running"),
                id="heartbeat",
            ),
            param(
                lambda: StatusUpdate(
                    sid="svc1", stype="timing_manager", state="stopping"
                ),
                id="status-update",
            ),
        ],
    )  # fmt: skip
    async def test_returns_none_and_updates_registry(
        self, system_controller: SystemController, msg_factory
    ) -> None:
        # Pre-register so update_service is non-noop.
        ServiceRegistry.register(
            service_id="svc1",
            service_type="timing_manager",
            first_seen_ns=1,
            state=LifecycleState.RUNNING,
        )

        result = await system_controller._handle_control_message("id", msg_factory())

        assert result is None
        info = ServiceRegistry.get_service("svc1")
        assert info is not None
        assert info.last_seen_ns is not None and info.last_seen_ns > 1

    async def test_heartbeat_for_unregistered_service_is_noop(
        self, system_controller: SystemController
    ) -> None:
        msg = Heartbeat(sid="ghost", stype="timing_manager", state="running")
        result = await system_controller._handle_control_message("id", msg)

        assert result is None
        assert ServiceRegistry.get_service("ghost") is None


# ============================================================
# MemoryReport
# ============================================================


class TestOnMemoryReport:
    """Memory reports flow through to the controller's MemoryTracker."""

    async def test_records_reading_on_tracker(
        self, system_controller: SystemController
    ) -> None:
        tracker = MagicMock()
        system_controller._memory_tracker = tracker

        msg = MemoryReport(
            sid="svc1",
            stype="worker",
            pid=1234,
            phase="startup",
            pss_bytes=100,
            rss_bytes=200,
            uss_bytes=150,
            shared_bytes=50,
        )
        result = await system_controller._handle_control_message("id", msg)

        assert result is None
        tracker.record.assert_called_once()
        kwargs = tracker.record.call_args.kwargs
        assert kwargs["label"] == "svc1"
        assert kwargs["group"] == "worker"
        assert kwargs["pid"] == 1234
        assert kwargs["reading"].pss == 100
        assert kwargs["reading"].rss == 200


# ============================================================
# TelemetryStatus
# ============================================================


class TestOnTelemetryStatus:
    """TelemetryStatus updates controller state and triggers shutdown check."""

    async def test_enabled_records_endpoints_and_triggers_check(
        self, system_controller: SystemController
    ) -> None:
        system_controller._check_and_trigger_shutdown = AsyncMock()  # type: ignore[method-assign]

        msg = TelemetryStatus(
            sid="tm",
            enabled=True,
            endpoints_configured=("e1", "e2"),
            endpoints_reachable=("e1",),
        )
        result = await system_controller._handle_control_message("id", msg)

        assert result is None
        assert system_controller._telemetry_endpoints_configured == ["e1", "e2"]
        assert system_controller._telemetry_endpoints_reachable == ["e1"]
        assert system_controller._should_wait_for_telemetry is True
        system_controller._check_and_trigger_shutdown.assert_awaited_once()

    async def test_disabled_forgets_service_in_registry(
        self, system_controller: SystemController
    ) -> None:
        ServiceRegistry.register(
            service_id="tm",
            service_type="telemetry_manager",
            first_seen_ns=1,
            state=LifecycleState.RUNNING,
        )
        system_controller._check_and_trigger_shutdown = AsyncMock()  # type: ignore[method-assign]

        msg = TelemetryStatus(sid="tm", enabled=False, reason="no DCGM exporter")
        await system_controller._handle_control_message("id", msg)

        assert system_controller._should_wait_for_telemetry is False
        assert not ServiceRegistry.is_registered("tm")


# ============================================================
# ServerMetricsStatus
# ============================================================


class TestOnServerMetricsStatus:
    """ServerMetricsStatus updates state, warns about unreachable endpoints."""

    async def test_enabled_with_unreachable_warns(
        self, system_controller: SystemController
    ) -> None:
        system_controller._check_and_trigger_shutdown = AsyncMock()  # type: ignore[method-assign]

        msg = ServerMetricsStatus(
            sid="sm",
            enabled=True,
            endpoints_configured=("e1", "e2", "e3"),
            endpoints_reachable=("e1",),
        )
        with patch.object(system_controller, "warning") as mock_warn:
            await system_controller._handle_control_message("id", msg)

        assert system_controller._should_wait_for_server_metrics is True
        # Should warn about the two unreachable endpoints.
        mock_warn.assert_called_once()
        warn_msg = mock_warn.call_args[0][0]
        assert "e2" in warn_msg or "e3" in warn_msg

    async def test_enabled_all_reachable_no_warn(
        self, system_controller: SystemController
    ) -> None:
        system_controller._check_and_trigger_shutdown = AsyncMock()  # type: ignore[method-assign]

        msg = ServerMetricsStatus(
            sid="sm",
            enabled=True,
            endpoints_configured=("e1",),
            endpoints_reachable=("e1",),
        )
        with patch.object(system_controller, "warning") as mock_warn:
            await system_controller._handle_control_message("id", msg)

        mock_warn.assert_not_called()

    async def test_disabled_forgets_service(
        self, system_controller: SystemController
    ) -> None:
        ServiceRegistry.register(
            service_id="sm",
            service_type="server_metrics_manager",
            first_seen_ns=1,
            state=LifecycleState.RUNNING,
        )
        system_controller._check_and_trigger_shutdown = AsyncMock()  # type: ignore[method-assign]

        msg = ServerMetricsStatus(sid="sm", enabled=False)
        await system_controller._handle_control_message("id", msg)

        assert system_controller._should_wait_for_server_metrics is False
        assert not ServiceRegistry.is_registered("sm")


# ============================================================
# Command + response stragglers
# ============================================================


class TestOnCommandAndResponseStragglers:
    """Commands forwarded to dispatch; stray responses logged but not crashing."""

    async def test_command_forwarded_to_dispatch_helper(
        self, system_controller: SystemController
    ) -> None:
        ack = CommandAck(cid="c1", sid=system_controller.service_id)
        system_controller._dispatch_control_command = AsyncMock(  # type: ignore[method-assign]
            return_value=ack
        )

        cmd = ControlCommand(cid="c1", cmd="DO")
        result = await system_controller._handle_control_message("origin", cmd)

        assert result is ack
        system_controller._dispatch_control_command.assert_awaited_once_with(
            "origin", cmd
        )

    @pytest.mark.parametrize(
        "msg",
        [
            param(CommandAck(cid="x", sid="s"), id="straggler-ack"),
            param(CommandOk(cid="x", sid="s", payload=b""), id="straggler-ok"),
            param(CommandErr(cid="x", sid="s", error="e"), id="straggler-err"),
        ],
    )  # fmt: skip
    async def test_response_stragglers_return_none_and_log(
        self, system_controller: SystemController, msg
    ) -> None:
        with patch.object(system_controller, "debug") as mock_debug:
            result = await system_controller._handle_control_message("origin", msg)

        assert result is None
        # debug() is called with the unexpected-response notice.
        assert mock_debug.called
