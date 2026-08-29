# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service side of the DEALER/ROUTER control channel."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest
import zmq

from aiperf.common.control_structs import (
    Command,
    CommandAck,
    CommandErr,
    CommandOk,
    CommandUnhandled,
    Registration,
    RegistrationAck,
    StatusUpdate,
)
from aiperf.common.enums import CommandType
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import AIPerfHook


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


async def _noop() -> None:
    return None


@pytest.mark.asyncio
async def test_unmatched_command_returns_command_unhandled(component_service) -> None:
    """Invariant I7: 'no handler' must stay distinguishable from 'acked'."""
    sent: list = []
    component_service.control_client.send = lambda s: sent.append(s) or _noop()
    await component_service._handle_control_command(
        Command(cid="c-1", cmd="a_command_with_no_handler")
    )
    assert isinstance(sent[0], CommandUnhandled)
    assert sent[0].cid == "c-1"
    assert sent[0].cmd == "a_command_with_no_handler"
    assert sent[0].sid == component_service.service_id


@pytest.mark.asyncio
async def test_handler_returning_none_sends_ack_carrying_cmd_and_sid(
    component_service,
) -> None:
    """Invariant I8."""
    sent: list = []
    component_service.control_client.send = lambda s: sent.append(s) or _noop()

    async def handler(message):
        return None

    hook = SimpleNamespace(func=handler)
    await component_service._execute_control_command(
        Command(cid="c-1", cmd=CommandType.FINALIZE_ARTIFACTS), hook
    )
    assert sent[0] == CommandAck(
        cid="c-1",
        cmd=CommandType.FINALIZE_ARTIFACTS,
        sid=component_service.service_id,
    )


@pytest.mark.asyncio
async def test_handler_returning_value_sends_command_ok_with_payload(
    component_service,
) -> None:
    sent: list = []
    component_service.control_client.send = lambda s: sent.append(s) or _noop()

    async def handler(message):
        return {"a": 1}

    hook = SimpleNamespace(func=handler)
    await component_service._execute_control_command(
        Command(cid="c-1", cmd=CommandType.GET_POD_STATES), hook
    )
    assert isinstance(sent[0], CommandOk)
    assert sent[0].cmd == CommandType.GET_POD_STATES
    assert orjson.loads(sent[0].payload) == {"a": 1}


@pytest.mark.asyncio
async def test_handler_raising_sends_command_err_with_traceback(
    component_service,
) -> None:
    sent: list = []
    component_service.control_client.send = lambda s: sent.append(s) or _noop()

    async def handler(message):
        raise ValueError("boom")

    hook = SimpleNamespace(func=handler)
    await component_service._execute_control_command(
        Command(cid="c-1", cmd=CommandType.PROFILE_START), hook
    )
    assert isinstance(sent[0], CommandErr)
    assert sent[0].cmd == CommandType.PROFILE_START
    assert sent[0].error == "boom"
    assert "ValueError" in sent[0].traceback


@pytest.mark.asyncio
async def test_shutdown_handler_cancellation_does_not_send_a_second_response(
    component_service,
) -> None:
    """The SHUTDOWN handler acks then raises CancelledError; the dispatcher must
    re-raise without also sending a response."""
    sent: list = []
    component_service.control_client.send = lambda s: sent.append(s) or _noop()

    async def handler(message):
        await component_service.control_client.send(
            CommandAck(
                cid=message.cid, cmd=message.cmd, sid=component_service.service_id
            )
        )
        raise asyncio.CancelledError()

    hook = SimpleNamespace(func=handler)
    with pytest.raises(asyncio.CancelledError):
        await component_service._execute_control_command(
            Command(cid="c-1", cmd=CommandType.SHUTDOWN), hook
        )
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_command_is_routed_to_the_matching_on_command_hook(
    component_service,
) -> None:
    """End-to-end through _handle_control_command, not just _execute_*."""
    sent: list = []
    component_service.control_client.send = lambda s: sent.append(s) or _noop()
    seen: list = []

    async def handler(message):
        seen.append(message)

    for hook in component_service.get_hooks(AIPerfHook.ON_COMMAND):
        if CommandType.SHUTDOWN in (hook.resolve_params(component_service) or ()):
            hook.func = handler
            break
    else:
        pytest.fail("component service has no SHUTDOWN @on_command hook")

    await component_service._handle_control_command(
        Command(cid="c-1", cmd=CommandType.SHUTDOWN)
    )
    assert [m.cid for m in seen] == ["c-1"]
    assert isinstance(sent[0], CommandAck)


@pytest.mark.asyncio
async def test_error_response_send_survives_a_closed_control_client(
    component_service,
) -> None:
    """Teardown race: a handler that fails *because* the service is stopping
    finds the DEALER already closed by comms.stop(). The second failure must
    not replace the logged handler error with an unhandled task exception."""

    async def dead_send(_struct) -> None:
        raise NotInitializedError("control client is not initialized")

    component_service.control_client.send = dead_send

    async def handler(message):
        raise ValueError("boom")

    hook = SimpleNamespace(func=handler)
    await component_service._execute_control_command(
        Command(cid="c-1", cmd=CommandType.PROFILE_START), hook
    )


@pytest.mark.asyncio
async def test_error_response_send_survives_a_closed_socket(
    component_service,
) -> None:
    """Same race, surfaced by libzmq instead of the client's own guard."""

    async def dead_send(_struct) -> None:
        raise zmq.ZMQError(zmq.ENOTSOCK)

    component_service.control_client.send = dead_send

    async def handler(message):
        raise ValueError("boom")

    hook = SimpleNamespace(func=handler)
    await component_service._execute_control_command(
        Command(cid="c-1", cmd=CommandType.PROFILE_START), hook
    )


@pytest.mark.asyncio
async def test_shutdown_self_ack_survives_a_closed_control_client(
    component_service,
) -> None:
    """A service that cannot ack must still stop."""

    async def dead_send(_struct) -> None:
        raise NotInitializedError("control client is not initialized")

    component_service.control_client.send = dead_send
    component_service.stop = AsyncMock()

    with pytest.raises(asyncio.CancelledError):
        await component_service._on_shutdown_command(
            Command(cid="c-1", cmd=CommandType.SHUTDOWN)
        )

    component_service.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_self_ack_still_propagates_cancellation(
    component_service,
) -> None:
    """CancelledError must keep escaping the suppress()."""

    async def cancelled_send(_struct) -> None:
        raise asyncio.CancelledError()

    component_service.control_client.send = cancelled_send
    component_service.stop = AsyncMock()

    with pytest.raises(asyncio.CancelledError):
        await component_service._on_shutdown_command(
            Command(cid="c-1", cmd=CommandType.SHUTDOWN)
        )

    component_service.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_command_to_controller_correlates_on_cid(component_service) -> None:
    captured: list = []

    async def fake_request(struct, timeout):
        captured.append((struct, timeout))
        return CommandOk(cid=struct.cid, cmd=struct.cmd, sid="ctl", payload=b'{"a":1}')

    component_service.control_client.request = fake_request
    response = await component_service.send_command_to_controller(
        CommandType.GET_POD_STATES, timeout=1.0
    )
    assert isinstance(response, CommandOk)
    struct, timeout = captured[0]
    assert isinstance(struct, Command)
    assert struct.cmd == CommandType.GET_POD_STATES
    assert struct.cid and struct.payload == b""
    assert timeout == 1.0
    assert response.cid == struct.cid
    assert orjson.loads(response.payload) == {"a": 1}


@pytest.mark.asyncio
async def test_send_command_to_controller_mints_a_fresh_cid_per_call(
    component_service,
) -> None:
    """Two commands must not share a cid, or the second reply resolves the first."""
    captured: list = []

    async def fake_request(struct, timeout):
        captured.append(struct)
        return CommandAck(cid=struct.cid, cmd=struct.cmd, sid="ctl")

    component_service.control_client.request = fake_request
    await component_service.send_command_to_controller(CommandType.SPAWN_WORKERS)
    await component_service.send_command_to_controller(CommandType.SPAWN_WORKERS)
    assert captured[0].cid != captured[1].cid


@pytest.mark.asyncio
async def test_send_command_to_controller_forwards_the_payload(
    component_service,
) -> None:
    captured: list = []

    async def fake_request(struct, timeout):
        captured.append(struct)
        return CommandAck(cid=struct.cid, cmd=struct.cmd, sid="ctl")

    component_service.control_client.request = fake_request
    await component_service.send_command_to_controller(
        CommandType.SPAWN_WORKERS, payload=orjson.dumps({"num_workers": 7})
    )
    assert orjson.loads(captured[0].payload) == {"num_workers": 7}


@pytest.mark.asyncio
async def test_send_command_to_controller_propagates_timeout(component_service) -> None:
    """The caller, not this helper, decides what a missing controller means."""

    async def fake_request(struct, timeout):
        raise TimeoutError("controller did not answer")

    component_service.control_client.request = fake_request
    with pytest.raises(TimeoutError):
        await component_service.send_command_to_controller(
            CommandType.FINALIZE_ARTIFACTS, timeout=0.01
        )
