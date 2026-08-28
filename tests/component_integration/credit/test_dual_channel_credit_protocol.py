# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dual-channel credit-protocol failure modes over real ZMQ sockets.

Credits are dispatched controller -> worker on ROUTER/DEALER
(``CommAddress.CREDIT_ROUTER``) and returned worker -> controller on a separate
PUSH/PULL fan-in (``CommAddress.CREDIT_RETURN``). The unit suite covers each
socket in isolation; what it cannot cover is the interaction between them, which
is where the interesting failures live:

* the return half being down while the dispatch half is healthy (a worker that
  can be routed credits it has no way to return),
* a return that lands after its worker left the routing pool (the two channels
  are not mutually ordered, so returns can trail lifecycle messages),
* a return racing its own dispatch across the two channels.

These run the real ``StickyCreditRouter`` against real client classes over live
``inproc`` sockets -- no mocks, real event loop, real FDs.
"""

import asyncio
import contextlib
import socket
import uuid

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import (
    CreditReturn,
    WorkerConnected,
    WorkerDispatchable,
    WorkerShutdown,
)
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.credit.structs import Credit
from aiperf.workers.return_channel_probe import probe_return_channel
from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient
from aiperf.zmq.streaming_pull_client import ZMQStreamingPullClient
from aiperf.zmq.streaming_push_client import ZMQStreamingPushClient
from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient

pytestmark = pytest.mark.component_integration

# Real (not virtual) time: these exercise libzmq connect/handshake, so a settle
# is needed before the first send on a freshly connected socket.
_SETTLE = 0.05
_RECV_TIMEOUT = 5.0


def _new_addr() -> str:
    """Loopback endpoint on a port nothing is listening on yet.

    TCP (not inproc) on purpose: ``IMMEDIATE=1`` only refuses sends when there
    is no completed connection, and libzmq's inproc transport completes a
    connect-before-bind pipe immediately, so an inproc endpoint cannot express
    "the peer is not there" -- which is the whole failure mode under test. TCP
    is also what the multi-pod Kubernetes deployment actually uses.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _make_run():
    """Minimal BenchmarkRun; the router only reads ``cfg.comm_config`` from it."""
    from aiperf.config import BenchmarkConfig, BenchmarkRun

    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["test-model"],
            "endpoint": {"type": "completions", "urls": ["http://localhost:8000/v1"]},
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 1,
                }
            ],
        }
    )
    return BenchmarkRun(
        benchmark_id=uuid.uuid4().hex,
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=None,
        variables={},
    )


def _credit(credit_id: int, correlation_id: str = "corr") -> Credit:
    return Credit(
        id=credit_id,
        phase=CreditPhase.PROFILING,
        conversation_id="conv",
        x_correlation_id=correlation_id,
        turn_index=0,
        num_turns=1,
        issued_at_ns=1,
    )


class _Fixture:
    """Router-side and worker-side halves of the dual-channel credit plane."""

    def __init__(self) -> None:
        self.dispatch_addr = _new_addr()
        self.return_addr = _new_addr()
        self.returns: asyncio.Queue[tuple[str, CreditReturn]] = asyncio.Queue()
        self.dispatched: asyncio.Queue = asyncio.Queue()
        self._clients: list = []
        self.router: StickyCreditRouter | None = None
        self.pull_client: ZMQStreamingPullClient | None = None
        self.dealer: ZMQStreamingDealerClient | None = None
        self.push: ZMQStreamingPushClient | None = None

    async def _make(self, cls, *, receiver=None, start=True, **kwargs):
        client = cls(**kwargs)
        await client.initialize()
        if receiver is not None:
            client.register_receiver(receiver)
        if start:
            await client.start()
        self._clients.append(client)
        return client

    async def start_router(self, *, bind_return: bool = True) -> StickyCreditRouter:
        """Build the real router, swapping in live sockets for both channels.

        The router's own constructor builds its clients from the comm factory;
        replacing them keeps every line of routing/accounting logic real while
        putting an actual libzmq transport underneath both channels.
        """
        router = StickyCreditRouter(run=_make_run(), service_id="test-router")
        router._router_client = await self._make(
            ZMQStreamingRouterClient,
            address=self.dispatch_addr,
            bind=True,
            receiver=router._handle_router_message,
        )
        if bind_return:
            await self.bind_return_channel(router)
        router.set_return_callback(self._on_return)
        self.router = router
        return router

    async def bind_return_channel(self, router: StickyCreditRouter) -> None:
        """Bring the PULL fan-in up (possibly long after the workers connected)."""
        self.pull_client = await self._make(
            ZMQStreamingPullClient,
            address=self.return_addr,
            bind=True,
            receiver=router._handle_return_pull_message,
        )
        router._return_pull_client = self.pull_client

    async def start_worker(self, worker_id: str, dispatch_receiver=None) -> None:
        """Connect a worker's dispatch DEALER and its return PUSH."""
        self.dealer = await self._make(
            ZMQStreamingDealerClient,
            address=self.dispatch_addr,
            bind=False,
            identity=worker_id,
            receiver=dispatch_receiver or self.dispatched.put,
        )
        self.push = await self._make(
            ZMQStreamingPushClient, address=self.return_addr, bind=False, start=False
        )
        await asyncio.sleep(_SETTLE)

    async def _on_return(self, worker_id: str, credit_return: CreditReturn) -> None:
        await self.returns.put((worker_id, credit_return))

    async def close(self) -> None:
        for client in reversed(self._clients):
            with contextlib.suppress(Exception):
                await client.stop()


@pytest.fixture
async def plane():
    fixture = _Fixture()
    yield fixture
    await fixture.close()


async def test_dispatch_healthy_return_channel_down_blocks_dispatchability(plane):
    """A live DEALER must not by itself make a worker dispatchable.

    The router binds ROUTER before PULL and each worker connects each socket on
    its own schedule, so "my dispatch channel handshaked" says nothing about the
    return channel. Announcing dispatchability off the DEALER alone puts a
    worker with a dead PUSH side into the routing pool, and every credit routed
    to it stalls until a run-level timeout.
    """
    router = await plane.start_router(bind_return=False)
    await plane.start_worker("worker-1")

    # Dispatch half is fully healthy: the router sees the worker's DEALER.
    await plane.dealer.send(WorkerConnected(worker_id="worker-1"))
    await asyncio.sleep(_SETTLE)
    assert "worker-1" in router._connected_workers

    # Return half has no peer, so the readiness gate refuses to let the worker
    # into the routing pool.
    assert not await probe_return_channel(
        plane.push, worker_id="worker-1", budget=0.3, retry_delay=0.05
    )
    assert router._workers == {}

    # Once the PULL fan-in is up the probe succeeds, and the probe frame itself
    # arrives on the return channel -- proving the direction end to end.
    await plane.bind_return_channel(router)
    await asyncio.sleep(_SETTLE)
    assert await probe_return_channel(
        plane.push, worker_id="worker-1", budget=2.0, retry_delay=0.05
    )

    await plane.dealer.send(WorkerDispatchable(worker_id="worker-1"))
    await asyncio.sleep(_SETTLE)
    assert "worker-1" in router._workers

    await router.send_credit(_credit(1))
    dispatched = await asyncio.wait_for(plane.dispatched.get(), timeout=_RECV_TIMEOUT)
    assert dispatched.id == 1
    await plane.push.send(CreditReturn(credit=dispatched, worker_id="worker-1"))
    worker_id, returned = await asyncio.wait_for(
        plane.returns.get(), timeout=_RECV_TIMEOUT
    )
    assert (worker_id, returned.credit.id) == ("worker-1", 1)
    assert router._workers["worker-1"].in_flight_credits == 0


async def test_returns_buffered_while_return_channel_down_drain_on_bind(plane):
    """Returns pushed with no PULL peer are held, not dropped.

    This is the other half of the readiness gate: the gate degrades open on
    budget expiry precisely because the PUSH client parks unsendable frames and
    drains them once a peer appears, so a late return channel costs latency
    rather than a hung phase.
    """
    router = await plane.start_router(bind_return=False)
    await plane.start_worker("worker-1")
    await plane.push.start()
    await plane.dealer.send(WorkerDispatchable(worker_id="worker-1"))
    await asyncio.sleep(_SETTLE)

    for credit_id in range(3):
        await plane.push.send(
            CreditReturn(credit=_credit(credit_id), worker_id="worker-1")
        )
    assert plane.returns.empty()

    await plane.bind_return_channel(router)

    drained = [
        (await asyncio.wait_for(plane.returns.get(), timeout=_RECV_TIMEOUT))[
            1
        ].credit.id
        for _ in range(3)
    ]
    assert drained == [0, 1, 2]


async def test_late_return_after_worker_left_the_routing_pool(plane):
    """A return that trails its worker's lifecycle message still reaches the consumer.

    Lifecycle messages ride the DEALER and returns ride the PUSH fan-in, so the
    two are not mutually ordered: a return can land after the router has already
    unregistered the worker. Dropping it would strand the credit's concurrency
    slot and hang the phase, so the return must still be delivered even though
    per-worker load accounting has nowhere to land.
    """
    router = await plane.start_router()
    await plane.start_worker("worker-1")
    await plane.push.start()
    await plane.dealer.send(WorkerDispatchable(worker_id="worker-1"))
    await asyncio.sleep(_SETTLE)

    await router.send_credit(_credit(7))
    dispatched = await asyncio.wait_for(plane.dispatched.get(), timeout=_RECV_TIMEOUT)

    await plane.dealer.send(WorkerShutdown(worker_id="worker-1"))
    await asyncio.sleep(_SETTLE)
    assert router._workers == {}
    assert "worker-1" in router._gracefully_shutdown_workers

    await plane.push.send(CreditReturn(credit=dispatched, worker_id="worker-1"))
    worker_id, returned = await asyncio.wait_for(
        plane.returns.get(), timeout=_RECV_TIMEOUT
    )
    assert (worker_id, returned.credit.id) == ("worker-1", 7)


async def test_return_racing_its_own_dispatch_keeps_accounting_exact(plane):
    """Returns pushed the instant a dispatch lands must not corrupt load accounting.

    The worker echoes every credit back on the PUSH channel as soon as the
    DEALER delivers it, so returns are continuously in flight against dispatches
    on the other socket. in_flight_credits must land back at 0 with no
    underflow and no lost or duplicated return.
    """
    router = await plane.start_router()
    total = 50

    async def echo(credit: Credit) -> None:
        """Return each credit the instant its dispatch lands, on the other socket."""
        await plane.push.send(CreditReturn(credit=credit, worker_id="worker-1"))

    await plane.start_worker("worker-1", dispatch_receiver=echo)
    await plane.push.start()
    await plane.dealer.send(WorkerDispatchable(worker_id="worker-1"))
    await asyncio.sleep(_SETTLE)

    for credit_id in range(total):
        await router.send_credit(_credit(credit_id, f"corr-{credit_id}"))

    returned_ids = [
        (await asyncio.wait_for(plane.returns.get(), timeout=_RECV_TIMEOUT))[
            1
        ].credit.id
        for _ in range(total)
    ]
    assert sorted(returned_ids) == list(range(total))

    load = router._workers["worker-1"]
    assert load.in_flight_credits == 0
    assert load.active_credit_ids == set()
    assert load.total_completed_credits == total
    assert load.total_cancelled_credits == 0
