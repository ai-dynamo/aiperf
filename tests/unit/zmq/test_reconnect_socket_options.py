# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Socket options that let a restarted pod rejoin an existing ROUTER.

Container networking drops idle TCP connections. Without ROUTER_HANDOVER the
stale routing entry survives the drop and libzmq refuses the reconnecting
peer's identity, so credits keep being routed into a dead entry and the phase
stalls with no error. Without a bounded reconnect backoff the peer takes far
longer than necessary to come back.
"""

import importlib

import pytest
import zmq

from aiperf.common.environment import Environment


class _FakeSocket:
    def __init__(self) -> None:
        self.opts: dict[int, int] = {}

    def setsockopt(self, key: int, val: int) -> None:
        self.opts[key] = val

    def bind(self, _addr: str) -> None: ...
    def connect(self, _addr: str) -> None: ...
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

    def test_connecting_socket_uses_runtime_reconnect_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A connecting peer receives non-default centralized reconnect values."""
        from aiperf.zmq import zmq_base_client, zmq_defaults
        from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient

        try:
            with monkeypatch.context() as overrides:
                overrides.setattr(Environment.ZMQ, "RECONNECT_IVL", 137)
                overrides.setattr(Environment.ZMQ, "RECONNECT_IVL_MAX", 2468)
                defaults = importlib.reload(zmq_defaults)
                overrides.setattr(
                    zmq_base_client, "ZMQSocketDefaults", defaults.ZMQSocketDefaults
                )

                client = ZMQStreamingDealerClient(
                    address="tcp://127.0.0.1:5555", identity="w-1", bind=False
                )
                sock = _FakeSocket()
                client.socket = sock
                client._apply_socket_options()

                assert sock.opts.get(zmq.RECONNECT_IVL) == 137
                assert sock.opts.get(zmq.RECONNECT_IVL_MAX) == 2468
        finally:
            importlib.reload(zmq_defaults)

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
