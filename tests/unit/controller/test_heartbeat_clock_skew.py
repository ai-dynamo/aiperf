# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Heartbeat/status timestamps are stamped by the controller, not the sender.

Under Kubernetes the sender and the controller are different machines, and
``ServiceRegistry.get_stale_services`` compares against the *controller's*
clock. A sender-stamped timestamp would therefore let a pod whose clock lags be
reaped the moment it reports in.

These cases were written against the pub/sub ``HeartbeatMessage`` /
``StatusMessage`` handlers, whose ``request_ns`` field the controller had to
deliberately ignore. The control-channel structs go one better and carry no
timestamp at all, so the invariant is now structural -- but the *consequence*
(nothing goes stale, the registry's ordering guard still holds) is what these
tests guard, and it is not covered by the struct-shape assertion in
``test_system_controller_dispatch.py``.
"""

import time

import pytest

from aiperf.common.control_structs import Heartbeat, Registration, StatusUpdate
from aiperf.common.enums import LifecycleState
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType

SERVICE_ID = "worker_group_manager_0"


def _registration() -> Registration:
    return Registration(
        sid=SERVICE_ID,
        rid="r-1",
        stype=str(ServiceType.WORKER_MANAGER),
        state=str(LifecycleState.RUNNING),
        pod_name="worker-pod-0",
        pod_index="0",
    )


def _heartbeat(state: LifecycleState) -> Heartbeat:
    return Heartbeat(
        sid=SERVICE_ID,
        stype=str(ServiceType.WORKER_MANAGER),
        state=str(state),
    )


async def _register(system_controller: SystemController) -> None:
    system_controller.service_manager.service_id_map = {}
    system_controller.service_manager.service_map = {}
    await system_controller._handle_control_message(SERVICE_ID, _registration())


@pytest.mark.asyncio
async def test_heartbeat_stamps_last_seen_from_the_controller_clock(
    system_controller: SystemController,
) -> None:
    """``last_seen_ns`` comes from the receiving side, so a lagging sender is safe."""
    await _register(system_controller)
    before_ns = time.time_ns()

    await system_controller._handle_control_message(
        SERVICE_ID, _heartbeat(LifecycleState.RUNNING)
    )

    info = ServiceRegistry.get_service(SERVICE_ID)
    assert info is not None
    assert info.last_seen_ns >= before_ns
    assert ServiceRegistry.get_stale_services(threshold_sec=10.0) == []


@pytest.mark.asyncio
async def test_out_of_order_heartbeat_does_not_move_state_backwards(
    system_controller: SystemController,
) -> None:
    """The registry's ordering guard must actually reject a stale update."""
    await _register(system_controller)

    await system_controller._handle_control_message(
        SERVICE_ID, _heartbeat(LifecycleState.STOPPING)
    )
    newest_ns = ServiceRegistry.get_service(SERVICE_ID).last_seen_ns

    # Delivered late by the transport, carrying an older view of the service.
    ServiceRegistry.update_service(
        SERVICE_ID,
        ServiceType.WORKER_MANAGER,
        newest_ns - 1,
        LifecycleState.RUNNING,
    )

    info = ServiceRegistry.get_service(SERVICE_ID)
    assert info.state == LifecycleState.STOPPING
    assert info.last_seen_ns == newest_ns


@pytest.mark.asyncio
async def test_same_tick_update_still_applies_the_newer_state(
    system_controller: SystemController,
) -> None:
    """A same-tick update is a clock collision, not an out-of-order delivery.

    Both callers stamp ``last_seen_ns`` on receipt from the controller's own
    clock, so equal timestamps mean two messages landed within one tick --
    ``time.time_ns()`` is ~15.6ms granular on Windows, coarser than a startup
    state sequence. Rejecting those would silently drop the newer state.
    """
    await _register(system_controller)

    await system_controller._handle_control_message(
        SERVICE_ID, _heartbeat(LifecycleState.INITIALIZED)
    )
    tick_ns = ServiceRegistry.get_service(SERVICE_ID).last_seen_ns

    ServiceRegistry.update_service(
        SERVICE_ID,
        ServiceType.WORKER_MANAGER,
        tick_ns,
        LifecycleState.RUNNING,
    )

    info = ServiceRegistry.get_service(SERVICE_ID)
    assert info.state == LifecycleState.RUNNING
    assert info.last_seen_ns == tick_ns


@pytest.mark.asyncio
async def test_status_update_stamps_last_seen_from_the_controller_clock(
    system_controller: SystemController,
) -> None:
    """``_on_status_update`` shares the heartbeat handler's shape."""
    await _register(system_controller)
    before_ns = time.time_ns()

    await system_controller._handle_control_message(
        SERVICE_ID,
        StatusUpdate(
            sid=SERVICE_ID,
            stype=str(ServiceType.WORKER_MANAGER),
            state=str(LifecycleState.RUNNING),
        ),
    )

    info = ServiceRegistry.get_service(SERVICE_ID)
    assert info is not None
    assert info.last_seen_ns >= before_ns
    assert ServiceRegistry.get_stale_services(threshold_sec=10.0) == []


def test_control_structs_carry_no_sender_timestamp() -> None:
    """The structural half of the invariant: there is no field to trust.

    The pub/sub predecessors carried ``request_ns``, and the controller had to
    remember to ignore it. Re-adding such a field would reintroduce the
    reaping bug even with the handlers unchanged.
    """
    for struct in (Heartbeat, StatusUpdate, Registration):
        assert not any(
            field in struct.__struct_fields__
            for field in ("ts", "request_ns", "sent_ns")
        )
