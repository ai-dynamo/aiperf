# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the credit-return channel readiness probe.

The probe is what makes ``WorkerDispatchable`` imply that BOTH halves of the
dual-channel credit protocol are live: dispatch on ROUTER/DEALER and returns on
the PUSH/PULL fan-in.
"""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import msgspec.msgpack
import pytest
import zmq

from aiperf.common.enums import CommAddress
from aiperf.common.exceptions import ConfigurationError
from aiperf.credit.messages import WorkerConnected, WorkerToRouterMessage
from aiperf.plugin.enums import CommClientType
from aiperf.workers.return_channel_probe import probe_return_channel
from aiperf.workers.worker import Worker
from tests.harness.fake_communication import FakeCommunication


class _FakePushClient:
    """Minimal stand-in exposing only what the probe touches."""

    def __init__(self, socket) -> None:
        self.socket = socket


def _client_raising(*side_effects, immediate: int = 1) -> _FakePushClient:
    socket = MagicMock(spec=zmq.Socket)
    socket.send.side_effect = list(side_effects)
    socket.getsockopt.return_value = immediate
    return _FakePushClient(socket)


@pytest.fixture
def sent_frames(monkeypatch) -> list[bytes]:
    """Capture frames the probe hands to ``zmq.Socket.send``."""
    frames: list[bytes] = []

    def _send(socket, data, flags=0, copy=True):
        frames.append(bytes(data))
        return socket.send(data, flags=flags, copy=copy)

    monkeypatch.setattr(zmq.Socket, "send", _send)
    return frames


@pytest.mark.asyncio
async def test_probe_returns_true_when_peer_accepts_frame(sent_frames):
    """A send libzmq accepts means a live PULL peer (IMMEDIATE=1)."""
    client = _client_raising(None)

    assert await probe_return_channel(
        client, worker_id="worker-1", budget=1.0, retry_delay=0.1
    )

    decoded = msgspec.msgpack.decode(sent_frames[0], type=WorkerToRouterMessage)
    assert isinstance(decoded, WorkerConnected)
    assert decoded.worker_id == "worker-1"


@pytest.mark.asyncio
async def test_probe_retries_until_peer_connects(sent_frames):
    """zmq.Again is the not-yet-connected signal, so keep retrying."""
    client = _client_raising(zmq.Again(), zmq.Again(), None)

    assert await probe_return_channel(
        client, worker_id="worker-1", budget=1.0, retry_delay=0.1
    )
    assert len(sent_frames) == 3


@pytest.mark.asyncio
async def test_probe_gives_up_after_budget(sent_frames):
    """Budget/retry_delay bounds the attempt count, then the probe fails."""
    client = _client_raising(*[zmq.Again()] * 20)

    assert not await probe_return_channel(
        client, worker_id="worker-1", budget=0.5, retry_delay=0.1
    )
    assert len(sent_frames) == 5


@pytest.mark.asyncio
async def test_probe_stops_on_socket_error(sent_frames):
    """A broken socket will not heal by retrying, so fail on the first error."""
    client = _client_raising(zmq.ZMQError(zmq.ENOTSOCK))

    assert not await probe_return_channel(
        client, worker_id="worker-1", budget=5.0, retry_delay=0.1
    )
    assert len(sent_frames) == 1


@pytest.mark.asyncio
async def test_probe_raises_when_immediate_not_set(sent_frames):
    """A caller that forgot socket_ops={zmq.IMMEDIATE: 1} must fail loudly.

    Without IMMEDIATE=1, libzmq buffers the NOBLOCK send in the not-yet-
    connected pipe and the send always succeeds -- so a probe that ignored
    this would report a dead return channel as live on the very first
    attempt, regardless of whether a real peer exists.
    """
    client = _client_raising(None, immediate=0)

    with pytest.raises(ConfigurationError):
        await probe_return_channel(
            client, worker_id="worker-1", budget=1.0, retry_delay=0.1
        )
    assert sent_frames == []


@pytest.mark.asyncio
async def test_probe_skips_transports_without_a_socket():
    """Non-ZMQ transports have no separate return socket to fail."""

    class _NoSocket:
        pass

    assert await probe_return_channel(
        _NoSocket(), worker_id="worker-1", budget=5.0, retry_delay=0.1
    )


@pytest.mark.asyncio
async def test_zero_budget_disables_probe(sent_frames):
    """0 is the documented escape hatch and must not send anything."""
    client = _client_raising(zmq.Again())

    assert await probe_return_channel(
        client, worker_id="worker-1", budget=0.0, retry_delay=0.1
    )
    assert sent_frames == []


def test_worker_enables_immediate_on_credit_return_client(
    benchmark_run, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[tuple[Any, Any], dict[Any, Any] | None] = {}
    create_client = FakeCommunication.create_client

    def capture_create_client(
        self: FakeCommunication,
        client_type: Any,
        address: Any,
        bind: bool = False,
        socket_ops: dict[Any, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        captured[(client_type, address)] = socket_ops
        return create_client(
            self,
            client_type,
            address,
            bind=bind,
            socket_ops=socket_ops,
            **kwargs,
        )

    monkeypatch.setattr(FakeCommunication, "create_client", capture_create_client)

    Worker(run=benchmark_run, service_id="worker-immediate-test")

    return_client = (CommClientType.STREAMING_PUSH, CommAddress.CREDIT_RETURN)
    assert captured[return_client] == {zmq.IMMEDIATE: 1}
    assert all(
        zmq.IMMEDIATE not in (socket_ops or {})
        for key, socket_ops in captured.items()
        if key != return_client
    )


class TestWorkerReadinessGate:
    """``WorkerDispatchable`` must not be sent until the return channel is live."""

    @staticmethod
    def _stub_worker(probe_result: bool, events: list[str]):
        """Minimal stand-in carrying only what the readiness path touches."""

        async def _send(struct) -> None:
            events.append(type(struct).__name__)

        async def _publish_startup_state(state) -> None:
            events.append(f"state:{state}")

        stub = SimpleNamespace(
            service_id="worker-1",
            _worker_ready_event=asyncio.Event(),
            _dataset_state_retry_task=None,
            credit_dealer_client=SimpleNamespace(send=_send),
            credit_return_push_client=object(),
            _publish_startup_state=_publish_startup_state,
            warning=lambda msg: events.append("warning"),
            _probe_result=probe_result,
        )
        # Bind the real methods so the production ordering is what is exercised.
        stub._await_return_channel_ready = lambda: Worker._await_return_channel_ready(
            stub
        )
        return stub

    async def _run_gate(self, worker, events, monkeypatch):
        async def _probe(push_client, *, worker_id, budget, retry_delay):
            events.append("probe")
            return worker._probe_result

        monkeypatch.setattr(
            "aiperf.workers.worker.probe_return_channel", _probe, raising=True
        )
        await Worker._mark_worker_ready_locked(worker)

    @pytest.mark.asyncio
    async def test_probe_runs_before_dispatchable_is_announced(self, monkeypatch):
        """The gate is the point: dispatchability implies both directions are live."""
        events: list[str] = []
        worker = self._stub_worker(True, events)

        await self._run_gate(worker, events, monkeypatch)

        assert events[:2] == ["probe", "WorkerDispatchable"]
        assert worker._worker_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_gate_degrades_open_with_a_warning(self, monkeypatch):
        """A never-arriving return channel must not wedge the worker forever.

        Returns are buffered and drained on reconnect, so announcing late-but-
        loudly beats never becoming dispatchable at all.
        """
        events: list[str] = []
        worker = self._stub_worker(False, events)

        await self._run_gate(worker, events, monkeypatch)

        assert events[:3] == ["probe", "warning", "WorkerDispatchable"]
        assert worker._worker_ready_event.is_set()
