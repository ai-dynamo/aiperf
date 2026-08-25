# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Socket options that let a restarted pod rejoin an existing ROUTER.

Container networking drops idle TCP connections. Without ROUTER_HANDOVER the
stale routing entry survives the drop and libzmq refuses the reconnecting
peer's identity, so credits keep being routed into a dead entry and the phase
stalls with no error. Without a bounded reconnect backoff the peer takes far
longer than necessary to come back.
"""

import pytest
import zmq

from aiperf.common.environment import Environment
from aiperf.zmq.zmq_defaults import ZMQSocketDefaults


class _FakeSocket:
    def __init__(self) -> None:
        self.opts: dict[int, int] = {}
        self.call_order: list[str] = []
        """Ordered log of every call made on this socket, so tests can assert
        that connect-time-sensitive options land on the correct side of
        bind()/connect() -- not just that they were set at all."""

    def setsockopt(self, key: int, val: int) -> None:
        self.opts[key] = val
        self.call_order.append(f"setsockopt:{key}")

    def bind(self, _addr: str) -> None:
        self.call_order.append("bind")

    def connect(self, _addr: str) -> None:
        self.call_order.append("connect")

    def close(self, **_kw: object) -> None: ...


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> type:
    from aiperf.zmq.pull_client import ZMQPullClient  # any concrete client

    return ZMQPullClient


class TestReconnectOptions:
    def test_router_sets_handover(self) -> None:
        """A ROUTER must replace a stale identity, not reject the reconnect."""
        from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient

        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        sock = _FakeSocket()
        client.socket = sock
        client._apply_socket_options()

        assert sock.opts.get(zmq.ROUTER_HANDOVER) == 1

    def test_connecting_socket_bounds_reconnect_backoff(self) -> None:
        """A connecting peer recovers on a bounded backoff, not the default."""
        from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient

        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555", identity="w-1", bind=False
        )
        sock = _FakeSocket()
        client.socket = sock
        client._apply_socket_options()

        assert ZMQSocketDefaults.RECONNECT_IVL == Environment.ZMQ.RECONNECT_IVL
        assert ZMQSocketDefaults.RECONNECT_IVL_MAX == Environment.ZMQ.RECONNECT_IVL_MAX
        assert sock.opts.get(zmq.RECONNECT_IVL) == Environment.ZMQ.RECONNECT_IVL
        assert sock.opts.get(zmq.RECONNECT_IVL_MAX) == Environment.ZMQ.RECONNECT_IVL_MAX

    def test_binding_socket_does_not_set_reconnect(self) -> None:
        """Reconnect options are meaningless on a bound socket."""
        from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient

        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        sock = _FakeSocket()
        client.socket = sock
        client._apply_socket_options()

        assert zmq.RECONNECT_IVL not in sock.opts

    def test_non_router_does_not_set_handover(self) -> None:
        """ROUTER_HANDOVER is only valid on a ROUTER."""
        from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient

        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555", identity="w-1", bind=False
        )
        sock = _FakeSocket()
        client.socket = sock
        client._apply_socket_options()

        assert zmq.ROUTER_HANDOVER not in sock.opts


class TestSocketOptionOrdering:
    """Drive the real ``_initialize_socket`` path (not ``_apply_socket_options``
    directly) so the pre/post-connect ordering invariant is genuinely exercised.

    RECONNECT_IVL*/ROUTER_HANDOVER must land before bind()/connect() -- libzmq
    snapshots them at connect time, so a later setsockopt is a no-op. IMMEDIATE
    is deliberately the opposite: set after, so a send to a not-yet-connected
    peer queues instead of raising zmq.Again. Calling ``_apply_socket_options``
    in isolation cannot see either half of this invariant since it never runs
    bind()/connect() at all.
    """

    @pytest.mark.asyncio
    async def test_initialize_socket_connecting_client_sets_reconnect_before_connect_and_immediate_after(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient

        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555", identity="w-1", bind=False
        )
        sock = _FakeSocket()
        monkeypatch.setattr(client.context, "socket", lambda *_a, **_kw: sock)

        await client._initialize_socket()

        connect_idx = sock.call_order.index("connect")
        reconnect_ivl_idx = sock.call_order.index(f"setsockopt:{zmq.RECONNECT_IVL}")
        reconnect_ivl_max_idx = sock.call_order.index(
            f"setsockopt:{zmq.RECONNECT_IVL_MAX}"
        )
        immediate_idx = sock.call_order.index(f"setsockopt:{zmq.IMMEDIATE}")

        assert reconnect_ivl_idx < connect_idx, (
            "RECONNECT_IVL must be set before connect() or libzmq ignores it"
        )
        assert reconnect_ivl_max_idx < connect_idx, (
            "RECONNECT_IVL_MAX must be set before connect() or libzmq ignores it"
        )
        assert immediate_idx > connect_idx, (
            "IMMEDIATE must be set after connect(), not before"
        )

    @pytest.mark.asyncio
    async def test_initialize_socket_binding_router_sets_handover_before_bind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient

        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        sock = _FakeSocket()
        monkeypatch.setattr(client.context, "socket", lambda *_a, **_kw: sock)

        await client._initialize_socket()

        bind_idx = sock.call_order.index("bind")
        handover_idx = sock.call_order.index(f"setsockopt:{zmq.ROUTER_HANDOVER}")

        assert handover_idx < bind_idx, (
            "ROUTER_HANDOVER must be set before bind() or libzmq ignores it"
        )
