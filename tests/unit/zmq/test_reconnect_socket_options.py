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

from aiperf.zmq.zmq_defaults import ZMQSocketDefaults


class _FakeSocket:
    def __init__(self):
        self.opts: dict[int, int] = {}

    def setsockopt(self, key, val):
        self.opts[key] = val

    def bind(self, _addr): ...
    def connect(self, _addr): ...
    def close(self, **_kw): ...


@pytest.fixture
def router(monkeypatch):
    from aiperf.zmq.pull_client import ZMQPullClient  # any concrete client

    return ZMQPullClient


class TestReconnectOptions:
    def test_router_sets_handover(self):
        """A ROUTER must replace a stale identity, not reject the reconnect."""
        from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient

        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        sock = _FakeSocket()
        client.socket = sock
        client._apply_socket_options()

        assert sock.opts.get(zmq.ROUTER_HANDOVER) == 1

    def test_connecting_socket_bounds_reconnect_backoff(self):
        """A connecting peer recovers on a bounded backoff, not the default."""
        from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient

        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555", identity="w-1", bind=False
        )
        sock = _FakeSocket()
        client.socket = sock
        client._apply_socket_options()

        assert sock.opts.get(zmq.RECONNECT_IVL) == ZMQSocketDefaults.RECONNECT_IVL
        assert (
            sock.opts.get(zmq.RECONNECT_IVL_MAX) == ZMQSocketDefaults.RECONNECT_IVL_MAX
        )

    def test_binding_socket_does_not_set_reconnect(self):
        """Reconnect options are meaningless on a bound socket."""
        from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient

        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        sock = _FakeSocket()
        client.socket = sock
        client._apply_socket_options()

        assert zmq.RECONNECT_IVL not in sock.opts

    def test_non_router_does_not_set_handover(self):
        """ROUTER_HANDOVER is only valid on a ROUTER."""
        from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient

        client = ZMQStreamingDealerClient(
            address="tcp://127.0.0.1:5555", identity="w-1", bind=False
        )
        sock = _FakeSocket()
        client.socket = sock
        client._apply_socket_options()

        assert zmq.ROUTER_HANDOVER not in sock.opts
