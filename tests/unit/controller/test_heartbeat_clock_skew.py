# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Heartbeat/status timestamps are stamped by the controller, not the sender.

``request_ns`` is filled by ``default_factory=time.time_ns`` inside the
*sender's* process, while ``ServiceRegistry.get_stale_services`` compares
against the *controller's* clock. Under Kubernetes the two clocks are
different machines, so a skewed sender would otherwise be reaped instantly.
"""

import time

import pytest

from aiperf.common.enums import LifecycleState
from aiperf.common.messages import (
    HeartbeatMessage,
    RegisterServiceCommand,
    StatusMessage,
)
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType

SKEW_NS = 20 * 1_000_000_000


def _registration() -> RegisterServiceCommand:
    return RegisterServiceCommand(
        service_id="worker_group_manager_0",
        service_type=ServiceType.WORKER_GROUP_MANAGER,
        state=LifecycleState.RUNNING,
        pod_name="worker-pod-0",
        pod_index="0",
    )


async def _register(system_controller: SystemController) -> None:
    system_controller.service_manager.service_id_map = {}
    system_controller.service_manager.service_map = {}
    await system_controller._handle_register_service_command(_registration())


@pytest.mark.asyncio
async def test_heartbeat_from_skewed_sender_clock_does_not_backdate_last_seen(
    system_controller: SystemController,
) -> None:
    """A sender whose clock lags must not be made instantly stale."""
    await _register(system_controller)
    before_ns = time.time_ns()

    await system_controller._process_heartbeat_message(
        HeartbeatMessage(
            service_id="worker_group_manager_0",
            service_type=ServiceType.WORKER_GROUP_MANAGER,
            state=LifecycleState.RUNNING,
            request_ns=before_ns - SKEW_NS,
        )
    )

    info = ServiceRegistry.get_service("worker_group_manager_0")
    assert info is not None
    assert info.last_seen_ns >= before_ns
    assert ServiceRegistry.get_stale_services(threshold_sec=10.0) == []


@pytest.mark.asyncio
async def test_out_of_order_heartbeat_does_not_move_state_backwards(
    system_controller: SystemController,
) -> None:
    """The registry's ordering guard must actually reject a stale update."""
    await _register(system_controller)

    await system_controller._process_heartbeat_message(
        HeartbeatMessage(
            service_id="worker_group_manager_0",
            service_type=ServiceType.WORKER_GROUP_MANAGER,
            state=LifecycleState.STOPPING,
            request_ns=time.time_ns(),
        )
    )
    newest_ns = ServiceRegistry.get_service("worker_group_manager_0").last_seen_ns

    # Delivered late by the transport, carrying an older view of the service.
    ServiceRegistry.update_service(
        "worker_group_manager_0",
        ServiceType.WORKER_GROUP_MANAGER,
        newest_ns - 1,
        LifecycleState.RUNNING,
    )

    info = ServiceRegistry.get_service("worker_group_manager_0")
    assert info.state == LifecycleState.STOPPING
    assert info.last_seen_ns == newest_ns


@pytest.mark.asyncio
async def test_status_from_skewed_sender_clock_does_not_backdate_last_seen(
    system_controller: SystemController,
) -> None:
    """``_process_status_message`` shares the heartbeat handler's shape."""
    await _register(system_controller)
    before_ns = time.time_ns()

    await system_controller._process_status_message(
        StatusMessage(
            service_id="worker_group_manager_0",
            service_type=ServiceType.WORKER_GROUP_MANAGER,
            state=LifecycleState.RUNNING,
            request_ns=before_ns - SKEW_NS,
        )
    )

    info = ServiceRegistry.get_service("worker_group_manager_0")
    assert info is not None
    assert info.last_seen_ns >= before_ns
    assert ServiceRegistry.get_stale_services(threshold_sec=10.0) == []
