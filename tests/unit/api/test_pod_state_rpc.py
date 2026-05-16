# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared ``query_controller_pod_states`` helper.

Both the progress and debug routers rely on this helper to talk to the
SystemController. Pinning its behavior here so future drift between the
two routers fails the test instead of silently changing the contract.
"""

from __future__ import annotations

import orjson
import pytest

from aiperf.api.pod_state_rpc import query_controller_pod_states
from aiperf.common.control_structs import CommandErr, CommandOk
from aiperf.common.enums import CommandType

pytestmark = pytest.mark.asyncio


class _FakeApp:
    """Stand-in for ``request.app`` exposing only ``app.state``."""

    def __init__(self, service: object | None) -> None:
        class _State:
            pass

        self.state = _State()
        if service is not None:
            self.state.service = service


class _FakeConn:
    """Stand-in for ``starlette.requests.HTTPConnection``."""

    def __init__(self, service: object | None) -> None:
        self.app = _FakeApp(service)


class _Service:
    """Captures the RPC arguments and replays a canned response."""

    def __init__(self, response: object) -> None:
        self._response = response
        self.last_cmd: str | None = None
        self.last_timeout: float | None = None

    async def send_command_to_controller(self, cmd: str, timeout: float) -> object:
        self.last_cmd = cmd
        self.last_timeout = timeout
        return self._response


def _ok(payload_dict: dict) -> CommandOk:
    return CommandOk(
        cid="cid-1",
        sid="system_controller",
        payload=orjson.dumps(payload_dict),
    )


@pytest.mark.asyncio
async def test_returns_decoded_dict_on_command_ok() -> None:
    payload = {
        "pod_states": {"0": {"pod_index": "0"}},
        "worker_startup_states": {"w-0": "ready"},
    }
    svc = _Service(_ok(payload))
    conn = _FakeConn(svc)

    result = await query_controller_pod_states(conn, timeout=2.5)

    assert result == payload
    assert svc.last_cmd == CommandType.GET_POD_STATES
    assert svc.last_timeout == 2.5


@pytest.mark.asyncio
async def test_returns_none_on_command_err() -> None:
    svc = _Service(CommandErr(cid="cid-1", sid="system_controller", error="boom"))
    conn = _FakeConn(svc)

    assert await query_controller_pod_states(conn, timeout=2.0) is None


@pytest.mark.asyncio
async def test_returns_none_when_service_missing() -> None:
    conn = _FakeConn(None)

    assert await query_controller_pod_states(conn, timeout=2.0) is None


@pytest.mark.asyncio
async def test_returns_none_when_service_lacks_send_command() -> None:
    class _NoRPC:
        pass

    conn = _FakeConn(_NoRPC())

    assert await query_controller_pod_states(conn, timeout=2.0) is None


@pytest.mark.asyncio
async def test_returns_none_when_rpc_raises() -> None:
    class _Boom:
        async def send_command_to_controller(self, *_a, **_kw):
            raise RuntimeError("control channel not initialized")

    conn = _FakeConn(_Boom())

    assert await query_controller_pod_states(conn, timeout=2.0) is None


@pytest.mark.asyncio
async def test_returns_none_on_decode_error() -> None:
    bad = CommandOk(
        cid="cid-1",
        sid="system_controller",
        payload=b"\xff\xff\xff not json \xff",
    )
    svc = _Service(bad)
    conn = _FakeConn(svc)

    assert await query_controller_pod_states(conn, timeout=2.0) is None
