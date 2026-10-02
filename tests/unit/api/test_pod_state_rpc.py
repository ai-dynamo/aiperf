# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The API's controller worker-state query over the DEALER/ROUTER channel.

The remote branch rides ``send_command_to_controller`` and must fall back to
the caller's bus-fed cache (``None``) on any transport failure, a non-CommandOk
reply, or an undecodable payload. The local-controller fast path is used when
the API shares a process with the controller and must stay untouched.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest

from aiperf.api.pod_state_rpc import query_controller_pod_states
from aiperf.common.control_structs import CommandAck, CommandErr, CommandOk
from aiperf.common.enums import CommandType
from aiperf.controller.system_controller_models import PodStateSnapshot


def _conn_with_service(**service_attrs: object) -> SimpleNamespace:
    """An HTTPConnection stand-in exposing only ``app.state.service``."""
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                controller=None, service=SimpleNamespace(**service_attrs)
            )
        )
    )


@pytest.mark.asyncio
async def test_query_controller_pod_states_decodes_orjson_payload() -> None:
    send = AsyncMock(
        return_value=CommandOk(
            cid="c",
            cmd=CommandType.GET_POD_STATES,
            sid="ctl",
            payload=orjson.dumps({"pod_states": {}, "worker_startup_states": {}}),
        )
    )
    conn = _conn_with_service(send_command_to_controller=send)

    snapshot = await query_controller_pod_states(conn)

    assert isinstance(snapshot, PodStateSnapshot)
    assert send.await_args.args[0] == CommandType.GET_POD_STATES


@pytest.mark.asyncio
async def test_query_controller_pod_states_falls_back_to_cache_on_command_error() -> (
    None
):
    """A CommandErr must not be parsed as a snapshot."""
    send = AsyncMock(
        return_value=CommandErr(
            cid="c", cmd=CommandType.GET_POD_STATES, sid="ctl", error="x"
        )
    )
    conn = _conn_with_service(send_command_to_controller=send)
    assert await query_controller_pod_states(conn) is None
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_controller_pod_states_falls_back_on_bare_ack() -> None:
    """A payload-less ack carries no snapshot; the cache is the answer."""
    send = AsyncMock(
        return_value=CommandAck(cid="c", cmd=CommandType.GET_POD_STATES, sid="ctl")
    )
    conn = _conn_with_service(send_command_to_controller=send)
    assert await query_controller_pod_states(conn) is None
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_controller_pod_states_falls_back_on_undecodable_payload() -> None:
    send = AsyncMock(
        return_value=CommandOk(
            cid="c",
            cmd=CommandType.GET_POD_STATES,
            sid="ctl",
            payload=b"not json at all",
        )
    )
    conn = _conn_with_service(send_command_to_controller=send)
    assert await query_controller_pod_states(conn) is None
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_controller_pod_states_falls_back_on_transport_failure() -> None:
    send = AsyncMock(side_effect=TimeoutError("no controller"))
    conn = _conn_with_service(send_command_to_controller=send)
    assert await query_controller_pod_states(conn) is None
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_controller_pod_states_ignores_the_legacy_pubsub_facade() -> None:
    """The pub/sub command facade is no longer a path to controller state.

    A service exposing only ``send_command_and_wait_for_response`` must yield
    the cache, not a snapshot; otherwise the remote branch is still riding the
    event bus.
    """
    legacy = AsyncMock(side_effect=AssertionError("must not use the event bus"))
    conn = _conn_with_service(
        service_id="api-service", send_command_and_wait_for_response=legacy
    )
    assert await query_controller_pod_states(conn) is None
    legacy.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_controller_pod_states_prefers_the_local_controller() -> None:
    """The in-process fast path must never reach the control channel.

    A preservation test: this branch is unchanged by the control-channel move,
    so it passes both before and after. It exists to stop a future cleanup from
    deleting the fast path along with the pub/sub facade beside it.
    """
    expected = PodStateSnapshot(pod_states={})
    send = AsyncMock(side_effect=AssertionError("must not use the control channel"))
    conn = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                controller=SimpleNamespace(get_pod_state_snapshot=lambda: expected),
                service=SimpleNamespace(send_command_to_controller=send),
            )
        )
    )

    assert await query_controller_pod_states(conn) is expected
    send.assert_not_awaited()
