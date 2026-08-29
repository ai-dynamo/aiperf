# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-channel dispatch on the SystemController.

Every behavior asserted here was previously carried by the
``@on_command(REGISTER_SERVICE)`` / ``@on_message(HEARTBEAT|STATUS)`` pub-sub
handlers; the transport changed, the semantics must not.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from aiperf.common.control_structs import (
    Heartbeat,
    Registration,
    RegistrationAck,
    StatusUpdate,
)
from aiperf.common.enums import (
    LifecycleState,
    ServiceRegistrationStatus,
    SystemState,
    make_result_producer_capability,
    parse_result_producer_capability,
)
from aiperf.common.models import ServiceRunInfo
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType


def _registration(
    sid: str = "svc-1",
    *,
    rid: str = "r-1",
    stype: ServiceType = ServiceType.WORKER,
    state: LifecycleState = LifecycleState.RUNNING,
    pod_name: str | None = None,
    capabilities: tuple[str, ...] = (),
) -> Registration:
    return Registration(
        sid=sid,
        rid=rid,
        stype=str(stype),
        state=str(state),
        pod_name=pod_name,
        capabilities=capabilities,
    )


@pytest.fixture
def controller(system_controller: SystemController) -> SystemController:
    """The shared controller fixture with empty, real service-manager maps."""
    system_controller.service_manager.service_id_map = {}
    system_controller.service_manager.service_map = {}
    return system_controller


@pytest.mark.asyncio
async def test_registration_returns_ack_with_matching_rid(
    controller: SystemController,
) -> None:
    ack = await controller._handle_control_message("svc-1", _registration())
    assert ack == RegistrationAck(rid="r-1")
    assert ServiceRegistry.is_registered("svc-1")
    assert "svc-1" in controller.service_manager.service_id_map
    assert [
        info.service_id
        for info in controller.service_manager.service_map[ServiceType.WORKER]
    ] == ["svc-1"]


@pytest.mark.asyncio
async def test_registration_joins_capabilities_into_result_barrier(
    controller: SystemController,
) -> None:
    """Behavior 4: capabilities -> result-join barrier."""
    await controller._handle_control_message(
        "svc-1",
        _registration(
            stype=ServiceType.GPU_TELEMETRY_MANAGER,
            capabilities=("result_producer:telemetry",),
        ),
    )
    assert "telemetry" in controller._result_join_coordinator.pending_domains


@pytest.mark.asyncio
async def test_registration_ignores_unprefixed_capability(
    controller: SystemController,
) -> None:
    """A capability that is not a result-producer tag must not join the barrier."""
    await controller._handle_control_message(
        "svc-1", _registration(capabilities=("telemetry",))
    )
    assert controller._result_join_coordinator.pending_domains == ()


@pytest.mark.asyncio
async def test_registration_clears_reaped_service_id(
    controller: SystemController,
) -> None:
    """Behavior 2: a service alive again must rejoin command fan-out."""
    controller._reaped_service_ids.add("svc-1")
    await controller._handle_control_message("svc-1", _registration())
    assert "svc-1" not in controller._reaped_service_ids


@pytest.mark.asyncio
async def test_registration_rekeys_service_map_on_service_type_change(
    controller: SystemController,
) -> None:
    """Behavior 3: a changed service type must not leave a stale peer entry."""
    await controller._handle_control_message(
        "svc-1", _registration(stype=ServiceType.WORKER)
    )
    await controller._handle_control_message(
        "svc-1", _registration(rid="r-2", stype=ServiceType.RECORD_PROCESSOR)
    )

    assert controller.service_manager.service_map[ServiceType.WORKER] == []
    assert [
        info.service_id
        for info in controller.service_manager.service_map[ServiceType.RECORD_PROCESSOR]
    ] == ["svc-1"]
    assert controller.service_manager.service_id_map[
        "svc-1"
    ] is ServiceRegistry.get_service("svc-1")


@pytest.mark.asyncio
async def test_registration_raises_when_registry_loses_the_registration(
    controller: SystemController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behavior 6: silently continuing here would publish a half-registered service."""
    monkeypatch.setattr(ServiceRegistry, "get_service", lambda _sid: None)
    with pytest.raises(RuntimeError, match="lost registration"):
        await controller._handle_control_message("svc-1", _registration())


@pytest.mark.asyncio
async def test_failed_worker_group_reregistration_triggers_configure(
    controller: SystemController,
) -> None:
    """Behavior 1 (FAILED branch) + behavior 5 + behavior 7."""
    controller._system_state = SystemState.PROFILING
    controller._configure_replacement_worker_group = AsyncMock()
    message = _registration(
        sid="wgm-0", stype=ServiceType.WORKER_GROUP_MANAGER, pod_name="pod-a"
    )
    await controller._handle_control_message("wgm-0", message)
    ServiceRegistry.fail_service("wgm-0", ServiceType.WORKER_GROUP_MANAGER, fatal=False)

    await controller._handle_control_message("wgm-0", message)
    await asyncio.sleep(0)

    assert "wgm-0" in controller._replacement_configuring_ids
    controller._configure_replacement_worker_group.assert_called_once_with("wgm-0")


@pytest.mark.asyncio
async def test_worker_group_reregistration_from_new_pod_triggers_configure(
    controller: SystemController,
) -> None:
    """Behavior 1 (changed pod_name branch)."""
    controller._system_state = SystemState.PROFILING
    controller._configure_replacement_worker_group = AsyncMock()
    await controller._handle_control_message(
        "wgm-0",
        _registration(
            sid="wgm-0", stype=ServiceType.WORKER_GROUP_MANAGER, pod_name="pod-a"
        ),
    )
    await controller._handle_control_message(
        "wgm-0",
        _registration(
            sid="wgm-0",
            rid="r-2",
            stype=ServiceType.WORKER_GROUP_MANAGER,
            pod_name="pod-b",
        ),
    )
    await asyncio.sleep(0)

    controller._configure_replacement_worker_group.assert_called_once_with("wgm-0")


@pytest.mark.asyncio
async def test_replacement_configure_is_skipped_while_initializing(
    controller: SystemController,
) -> None:
    """Behavior 7: a replacement seen before startup finishes is configured by
    the normal bulk-configure pass, not by the per-service replacement path."""
    controller._system_state = SystemState.INITIALIZING
    controller._configure_replacement_worker_group = AsyncMock()
    await controller._handle_control_message(
        "wgm-0",
        _registration(
            sid="wgm-0", stype=ServiceType.WORKER_GROUP_MANAGER, pod_name="pod-a"
        ),
    )
    await controller._handle_control_message(
        "wgm-0",
        _registration(
            sid="wgm-0",
            rid="r-2",
            stype=ServiceType.WORKER_GROUP_MANAGER,
            pod_name="pod-b",
        ),
    )
    await asyncio.sleep(0)

    controller._configure_replacement_worker_group.assert_not_called()
    assert "wgm-0" not in controller._replacement_configuring_ids


@pytest.mark.asyncio
async def test_replacement_configure_is_not_started_twice(
    controller: SystemController,
) -> None:
    """Behavior 5: the in-flight guard survives a second replacement signal."""
    controller._system_state = SystemState.PROFILING
    controller._configure_replacement_worker_group = AsyncMock()
    for index, pod in enumerate(("pod-a", "pod-b", "pod-c")):
        await controller._handle_control_message(
            "wgm-0",
            _registration(
                sid="wgm-0",
                rid=f"r-{index}",
                stype=ServiceType.WORKER_GROUP_MANAGER,
                pod_name=pod,
            ),
        )
    await asyncio.sleep(0)

    controller._configure_replacement_worker_group.assert_called_once_with("wgm-0")


@pytest.mark.asyncio
async def test_heartbeat_stamps_last_seen_from_controller_clock(
    controller: SystemController,
) -> None:
    """Invariant I1: the wire carries no timestamp; the controller stamps."""
    await controller._handle_control_message("svc-1", _registration())
    before = time.time_ns()
    result = await controller._handle_control_message(
        "svc-1",
        Heartbeat(
            sid="svc-1",
            stype=str(ServiceType.WORKER),
            state=str(LifecycleState.RUNNING),
        ),
    )
    after = time.time_ns()

    assert result is None, "heartbeat is fire-and-forget"
    info = ServiceRegistry.get_service("svc-1")
    assert before <= info.last_seen_ns <= after
    assert not any(
        field in Heartbeat.__struct_fields__
        for field in ("ts", "request_ns", "sent_ns")
    ), "Heartbeat must not carry a sender-stamped timestamp"


@pytest.mark.asyncio
async def test_heartbeat_from_unknown_service_is_ignored(
    controller: SystemController,
) -> None:
    result = await controller._handle_control_message(
        "ghost",
        Heartbeat(
            sid="ghost",
            stype=str(ServiceType.WORKER),
            state=str(LifecycleState.RUNNING),
        ),
    )
    assert result is None
    assert ServiceRegistry.get_service("ghost") is None


@pytest.mark.asyncio
async def test_status_update_records_new_state(controller: SystemController) -> None:
    await controller._handle_control_message("svc-1", _registration())
    result = await controller._handle_control_message(
        "svc-1",
        StatusUpdate(
            sid="svc-1",
            stype=str(ServiceType.WORKER),
            state=str(LifecycleState.STOPPING),
        ),
    )
    assert result is None
    assert ServiceRegistry.get_service("svc-1").state == LifecycleState.STOPPING


@pytest.mark.asyncio
async def test_status_update_does_not_write_through_service_id_map(
    controller: SystemController,
) -> None:
    """Invariant I2: the registry is the sole writer of state/last_seen_ns.

    ``service_id_map`` holds the very ``ServiceRunInfo`` the registry owns, so a
    handler that wrote to it directly would pre-satisfy the ordering guard in
    ``update_service`` and let a stale update through.
    """
    await controller._handle_control_message("svc-1", _registration())
    info = controller.service_manager.service_id_map["svc-1"]
    assert info is ServiceRegistry.get_service("svc-1")

    await controller._handle_control_message(
        "svc-1",
        StatusUpdate(
            sid="svc-1",
            stype=str(ServiceType.WORKER),
            state=str(LifecycleState.STOPPING),
        ),
    )
    newest_ns = info.last_seen_ns

    # An update the transport delivered late must be dropped whole.
    ServiceRegistry.update_service(
        "svc-1", ServiceType.WORKER, newest_ns - 1, LifecycleState.RUNNING
    )
    assert info.state == LifecycleState.STOPPING
    assert info.last_seen_ns == newest_ns


@pytest.mark.asyncio
async def test_status_update_from_unknown_service_is_ignored(
    controller: SystemController,
) -> None:
    result = await controller._handle_control_message(
        "ghost",
        StatusUpdate(
            sid="ghost",
            stype=str(ServiceType.WORKER),
            state=str(LifecycleState.RUNNING),
        ),
    )
    assert result is None
    assert ServiceRegistry.get_service("ghost") is None


def test_result_producer_capability_prefix_is_pinned() -> None:
    """The capability wire format is stringly-typed and unenforced.

    ``_on_registration`` is its first consumer, so pin the exact prefix: a
    silent change would drop producers out of the shutdown barrier with no
    error anywhere.
    """
    assert make_result_producer_capability("telemetry") == "result_producer:telemetry"
    assert parse_result_producer_capability("result_producer:telemetry") == "telemetry"
    assert parse_result_producer_capability("telemetry") is None
    assert parse_result_producer_capability("result_producer:") is None


def test_is_replacement_worker_group_registration_ignores_other_types(
    controller: SystemController,
) -> None:
    """Only worker-group managers get the replacement treatment."""
    prior = ServiceRunInfo(
        service_id="svc-1",
        service_type=ServiceType.WORKER,
        registration_status=ServiceRegistrationStatus.REGISTERED,
        first_seen_ns=1,
        last_seen_ns=1,
        state=LifecycleState.FAILED,
    )
    assert not controller._is_replacement_worker_group_registration(
        _registration(stype=ServiceType.WORKER, pod_name="pod-b"), prior, True
    )


def test_control_router_is_not_lifecycle_managed_by_comms(benchmark_run) -> None:
    """Invariant: comms must not own the control ROUTER.

    ``FakeCommunication`` never attaches child lifecycles at all, so only a
    real ZMQ communication layer can catch a regression in this wiring.
    """
    from types import SimpleNamespace

    from aiperf.zmq.zmq_comms import ZMQIPCCommunication

    comms = ZMQIPCCommunication(config=benchmark_run.comm_config)
    stub = SimpleNamespace(comms=comms, run=benchmark_run)
    SystemController._init_control_router(stub)

    assert stub.control_router is not None
    assert stub.control_router not in comms._children
