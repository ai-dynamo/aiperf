# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service side of the DEALER/ROUTER control channel."""

import asyncio

import pytest

from aiperf.common.control_structs import (
    Registration,
    RegistrationAck,
    StatusUpdate,
)


@pytest.mark.asyncio
async def test_register_until_ack_resolves_on_matching_rid(component_service) -> None:
    sent: list[Registration] = []

    async def fake_send(struct) -> None:
        sent.append(struct)
        component_service._registration_ack_event.set()

    component_service.control_client.send = fake_send
    await component_service._register_until_ack(
        send_interval=0.05,
        overall_timeout=1.0,
        initial_warning_threshold=5.0,
        warning_interval=10.0,
    )
    assert len(sent) == 1
    assert sent[0].sid == component_service.service_id
    assert component_service._registration_complete


@pytest.mark.asyncio
async def test_register_until_ack_ignores_stale_ack_from_prior_attempt(
    component_service,
) -> None:
    """A late ack for attempt 1 must not unblock attempt 2."""
    sent: list[Registration] = []
    second_attempt_sent = asyncio.Event()

    async def fake_send(struct) -> None:
        sent.append(struct)
        if len(sent) == 2:
            second_attempt_sent.set()

    component_service.control_client.send = fake_send
    task = asyncio.create_task(
        component_service._register_until_ack(
            send_interval=0.05,
            overall_timeout=5.0,
            initial_warning_threshold=60.0,
            warning_interval=60.0,
        )
    )
    await asyncio.wait_for(second_attempt_sent.wait(), timeout=5.0)
    assert sent[1].rid != sent[0].rid, "each attempt must mint a fresh rid"

    # The ack for attempt 1, delivered after attempt 2 went out.
    await component_service._handle_control_command(RegistrationAck(rid=sent[0].rid))
    # Give the registration coroutine every chance to wake on a satisfied event
    # before concluding it is still blocked.
    for _ in range(5):
        await asyncio.sleep(0)
    assert not task.done(), "a stale ack must not satisfy the current attempt"

    await component_service._handle_control_command(
        RegistrationAck(rid=component_service._pending_registration_rid)
    )
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_register_until_ack_times_out_without_ack(component_service) -> None:
    async def fake_send(struct) -> None:
        return None

    component_service.control_client.send = fake_send
    with pytest.raises(TimeoutError, match="timed out"):
        await component_service._register_until_ack(
            send_interval=0.02,
            overall_timeout=0.06,
            initial_warning_threshold=5.0,
            warning_interval=10.0,
        )
    assert not component_service._registration_complete
    assert component_service._pending_registration_rid is None


@pytest.mark.asyncio
async def test_registration_advertises_extra_capabilities(component_service) -> None:
    """Capabilities ride the registration so the controller can join results."""
    type(component_service).extra_capabilities = ("result_producer:telemetry",)
    try:
        registration = component_service._make_registration()
    finally:
        type(component_service).extra_capabilities = ()
    assert registration.capabilities == ("result_producer:telemetry",)


def test_registration_carries_no_timestamp() -> None:
    """Invariant I1: a sender-stamped time would let clock skew reap a live pod."""
    assert "ts" not in Registration.__struct_fields__
    assert "request_ns" not in Registration.__struct_fields__
    assert "request_ns" not in StatusUpdate.__struct_fields__


@pytest.mark.asyncio
async def test_handle_control_command_drops_unknown_message(component_service) -> None:
    """Command dispatch is not on the control channel yet; nothing may raise."""
    component_service._registration_ack_event = asyncio.Event()
    component_service._pending_registration_rid = "r-1"

    await component_service._handle_control_command(
        StatusUpdate(sid="x", stype="worker", state="running")
    )
    assert not component_service._registration_ack_event.is_set()


@pytest.mark.asyncio
async def test_state_change_sends_status_update_on_control_channel(
    component_service,
) -> None:
    from aiperf.common.enums import LifecycleState

    sent: list[StatusUpdate] = []

    async def fake_send(struct) -> None:
        sent.append(struct)

    component_service.control_client.send = fake_send
    component_service.comms.initialized_event.set()
    await component_service._on_state_change(
        LifecycleState.STARTING, LifecycleState.RUNNING
    )

    assert len(sent) == 1
    assert isinstance(sent[0], StatusUpdate)
    assert sent[0].sid == component_service.service_id
    assert sent[0].state == str(LifecycleState.RUNNING)


@pytest.mark.asyncio
async def test_state_change_is_silent_after_stop_requested(component_service) -> None:
    from aiperf.common.enums import LifecycleState

    sent: list[StatusUpdate] = []

    async def fake_send(struct) -> None:
        sent.append(struct)

    component_service.control_client.send = fake_send
    component_service.comms.initialized_event.set()
    component_service.stop_requested = True
    await component_service._on_state_change(
        LifecycleState.RUNNING, LifecycleState.STOPPING
    )
    assert sent == []


@pytest.mark.asyncio
async def test_heartbeat_task_is_silent_before_registration(component_service) -> None:
    """The background-task hook runs before registration; a heartbeat sent then
    would only produce an "unknown service" warning on the controller."""
    sent: list[object] = []

    async def fake_send(struct) -> None:
        sent.append(struct)

    component_service.control_client.send = fake_send
    await component_service._heartbeat_task()
    assert sent == []

    component_service._registration_complete = True
    await component_service._heartbeat_task()
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_heartbeat_survives_a_socket_closed_underneath_it(
    component_service,
) -> None:
    """comms.stop() closes the DEALER before the task manager cancels this task.

    ``_stop_children`` runs ahead of ``_stop_all_tasks``, so a tick can clear the
    ``stop_requested`` guard and still reach a dead socket.
    ``BaseZMQClient._check_initialized`` reports that as ``NotInitializedError``,
    which must not escape to the background-task runner as a spurious shutdown
    exception.
    """
    from aiperf.common.exceptions import NotInitializedError

    async def dead_socket_send(struct) -> None:
        raise NotInitializedError("Socket not initialized or closed")

    component_service.control_client.send = dead_socket_send
    component_service._registration_complete = True

    await component_service._heartbeat_task()


@pytest.mark.asyncio
async def test_early_heartbeat_loop_survives_a_socket_closed_underneath_it(
    component_service,
) -> None:
    from aiperf.common.exceptions import NotInitializedError

    async def dead_socket_send(struct) -> None:
        raise NotInitializedError("Socket not initialized or closed")

    component_service.control_client.send = dead_socket_send

    await component_service._early_heartbeat_loop()


@pytest.mark.asyncio
async def test_heartbeat_lets_cancellation_propagate(component_service) -> None:
    """CancelledError is how ``_check_initialized`` reports a stopped socket and
    how the task manager unwinds this task; swallowing it would wedge shutdown."""

    async def cancelled_send(struct) -> None:
        raise asyncio.CancelledError("Socket was stopped")

    component_service.control_client.send = cancelled_send
    component_service._registration_complete = True

    with pytest.raises(asyncio.CancelledError):
        await component_service._heartbeat_task()
