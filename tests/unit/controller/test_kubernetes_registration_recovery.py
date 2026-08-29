# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubernetes registrations populate tracking and recover replacement pods."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.control_structs import CommandAck
from aiperf.common.enums import CommandType, LifecycleState, SystemState
from aiperf.common.messages import HeartbeatMessage, RegisterServiceCommand
from aiperf.common.service_registry import ServiceRegistry
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType


def _registration(pod_name: str) -> RegisterServiceCommand:
    return RegisterServiceCommand(
        service_id="worker_group_manager_0",
        service_type=ServiceType.WORKER_GROUP_MANAGER,
        state=LifecycleState.RUNNING,
        pod_name=pod_name,
        pod_index="0",
    )


@pytest.mark.asyncio
async def test_registration_populates_registry_with_pod_identity(
    system_controller: SystemController,
) -> None:
    system_controller.service_manager.service_id_map = {}
    system_controller.service_manager.service_map = {}

    await system_controller._handle_register_service_command(
        _registration("worker-pod-old")
    )

    info = ServiceRegistry.get_service("worker_group_manager_0")
    assert info is not None
    assert info.pod_name == "worker-pod-old"
    assert info.pod_index == "0"
    assert ServiceRegistry.is_registered("worker_group_manager_0")


@pytest.mark.asyncio
async def test_duplicate_registration_does_not_duplicate_service_map(
    system_controller: SystemController,
) -> None:
    system_controller.service_manager.service_id_map = {}
    system_controller.service_manager.service_map = {}
    message = _registration("worker-pod-old")

    await system_controller._handle_register_service_command(message)
    await system_controller._handle_register_service_command(message)

    services = system_controller.service_manager.service_map[
        ServiceType.WORKER_GROUP_MANAGER
    ]
    assert [info.service_id for info in services] == ["worker_group_manager_0"]


@pytest.mark.asyncio
async def test_failed_worker_group_replacement_is_reconfigured(
    system_controller: SystemController,
) -> None:
    system_controller.service_manager.service_id_map = {}
    system_controller.service_manager.service_map = {}
    await system_controller._handle_register_service_command(
        _registration("worker-pod-old")
    )
    ServiceRegistry.fail_service(
        "worker_group_manager_0",
        ServiceType.WORKER_GROUP_MANAGER,
        fatal=False,
    )
    system_controller._system_state = SystemState.CONFIGURING
    system_controller._send_control_command = AsyncMock(
        return_value=CommandAck(
            cid="c-1",
            cmd=CommandType.PROFILE_CONFIGURE,
            sid="worker_group_manager_0",
        )
    )
    scheduled = []

    with patch.object(
        system_controller,
        "execute_async",
        side_effect=lambda coroutine: scheduled.append(coroutine) or MagicMock(),
    ):
        await system_controller._handle_register_service_command(
            _registration("worker-pod-new")
        )

    assert len(scheduled) == 1
    await scheduled[0]
    identity, cmd = system_controller._send_control_command.await_args.args
    assert identity == "worker_group_manager_0"
    assert cmd == CommandType.PROFILE_CONFIGURE
    assert ServiceRegistry.is_registered("worker_group_manager_0")


@pytest.mark.asyncio
async def test_process_heartbeat_message_stamps_registry_timestamp_on_receipt(
    system_controller: SystemController,
) -> None:
    """``last_seen_ns`` comes from the controller's clock, never the sender's.

    ``request_ns`` is stamped in the sender's process on a different machine
    under Kubernetes, so adopting it would make a skewed pod look instantly
    stale. See ``tests/unit/controller/test_heartbeat_clock_skew.py``.
    """
    system_controller.service_manager.service_id_map = {}
    system_controller.service_manager.service_map = {}
    await system_controller._handle_register_service_command(
        _registration("worker-pod-old")
    )
    before_ns = time.time_ns()

    await system_controller._process_heartbeat_message(
        HeartbeatMessage(
            service_id="worker_group_manager_0",
            service_type=ServiceType.WORKER_GROUP_MANAGER,
            state=LifecycleState.RUNNING,
            request_ns=12345,
        )
    )

    info = ServiceRegistry.get_service("worker_group_manager_0")
    assert info is not None
    assert info.last_seen_ns != 12345
    assert info.last_seen_ns >= before_ns
