# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial real-socket tests for the FD-driven ZMQ credit clients.

These run the actual client classes over a live ``libzmq`` ``inproc`` transport
(no mocks, real event loop, real FD) to cover edge cases the mocked unit suite
cannot reproduce:

* the FD edge-trigger re-arm under a tight burst (the kernel FD fires once on the
  0->nonzero transition; messages that land mid-drain must still be picked up),
* the bootstrap drain of a backlog queued *before* the reader is installed,
* true PUSH/PULL fan-in from many real worker sockets into one PULL,
* the shared-FD bidirectional DEALER/ROUTER drive (recv and send on one FD),
* graceful stop while messages are still in flight.
"""

import asyncio
from collections import Counter

import msgspec

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import (
    CreditReturn,
    WorkerReady,
)
from aiperf.credit.structs import Credit
from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient
from aiperf.zmq.streaming_pull_client import ZMQStreamingPullClient
from aiperf.zmq.streaming_push_client import ZMQStreamingPushClient
from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient

# Small real-time settle so inproc connect/subscription completes before the
# first send (real time here - no looptime in tests/zmq).
_SETTLE = 0.05


class _Bogus(msgspec.Struct, tag_field="t", tag="bogus"):
    """A tagged struct whose tag is NOT in WorkerToRouterMessage, so the PULL
    decoder raises a (non-ZMQ) decode error on it."""

    n: int = 0


def _credit(i: int) -> Credit:
    return Credit(
        id=i,
        phase=CreditPhase.PROFILING,
        conversation_id="conv",
        x_correlation_id="corr",
        turn_index=0,
        num_turns=1,
        issued_at_ns=1,
    )


async def test_push_pull_roundtrip(client_factory, new_addr):
    """A CreditReturn pushed over the real PUSH/PULL channel decodes intact,
    including the in-message worker_id (the channel carries no ZMQ identity)."""
    addr = new_addr()
    inbox: asyncio.Queue = asyncio.Queue()

    await client_factory(
        ZMQStreamingPullClient, address=addr, bind=True, start=True, receiver=inbox.put
    )
    push = await client_factory(ZMQStreamingPushClient, address=addr, bind=False)
    await asyncio.sleep(_SETTLE)

    await push.send(CreditReturn(credit=_credit(42), worker_id="worker-7"))

    msg = await asyncio.wait_for(inbox.get(), timeout=2.0)
    assert isinstance(msg, CreditReturn)
    assert msg.credit.id == 42
    assert msg.worker_id == "worker-7"


async def test_push_pull_fan_in_no_loss(client_factory, new_addr):
    """Many real worker PUSH sockets fan in to one PULL with no loss or dup."""
    addr = new_addr()
    n_workers, per_worker = 4, 64
    total = n_workers * per_worker
    received: list[tuple[str, int]] = []
    done = asyncio.Event()

    async def handler(msg: CreditReturn) -> None:
        received.append((msg.worker_id, msg.credit.id))
        if len(received) >= total:
            done.set()

    await client_factory(
        ZMQStreamingPullClient, address=addr, bind=True, start=True, receiver=handler
    )
    pushers = [
        await client_factory(ZMQStreamingPushClient, address=addr, bind=False)
        for _ in range(n_workers)
    ]
    await asyncio.sleep(_SETTLE)

    async def blast(worker_idx: int, push: ZMQStreamingPushClient) -> None:
        for i in range(per_worker):
            await push.send(
                CreditReturn(credit=_credit(i), worker_id=f"worker-{worker_idx}")
            )

    await asyncio.gather(*(blast(w, p) for w, p in enumerate(pushers)))
    await asyncio.wait_for(done.wait(), timeout=5.0)

    assert len(received) == total
    expected = Counter(
        (f"worker-{w}", i) for w in range(n_workers) for i in range(per_worker)
    )
    assert Counter(received) == expected


async def test_push_pull_burst_drains_all_in_order(client_factory, new_addr):
    """A tight burst from one PUSH is fully drained, in order.

    This is the FD re-arm stress: the raw FD only signals on the 0->nonzero
    ``ZMQ_EVENTS`` edge, so messages enqueued while ``_pump`` is mid-drain would
    be stranded if the reader did not re-check ``POLLIN`` and ``call_soon`` itself.
    A single pusher preserves order, so an exact ordered match proves no message
    was dropped or duplicated.
    """
    addr = new_addr()
    burst = 2000
    received: list[int] = []
    done = asyncio.Event()

    async def handler(msg: CreditReturn) -> None:
        received.append(msg.credit.id)
        if len(received) >= burst:
            done.set()

    await client_factory(
        ZMQStreamingPullClient, address=addr, bind=True, start=True, receiver=handler
    )
    push = await client_factory(ZMQStreamingPushClient, address=addr, bind=False)
    await asyncio.sleep(_SETTLE)

    for i in range(burst):
        await push.send(CreditReturn(credit=_credit(i)))

    await asyncio.wait_for(done.wait(), timeout=10.0)
    assert received == list(range(burst))


async def test_pull_bootstrap_drains_prestart_backlog(client_factory, new_addr):
    """Messages queued before the reader is installed are drained on start().

    The edge-triggered FD will not fire for messages already sitting in the recv
    buffer when ``add_reader`` registers, so ``start()`` must run a bootstrap
    ``_pump`` to clear the backlog. Here we send before ``pull.start()`` and
    assert nothing is delivered until start, then the whole backlog arrives.
    """
    addr = new_addr()
    received: list[int] = []
    got_all = asyncio.Event()
    backlog = 16

    async def handler(msg: CreditReturn) -> None:
        received.append(msg.credit.id)
        if len(received) >= backlog:
            got_all.set()

    # Initialize + register the handler, but do NOT start the FD reader yet.
    pull = await client_factory(
        ZMQStreamingPullClient, address=addr, bind=True, start=False, receiver=handler
    )
    push = await client_factory(ZMQStreamingPushClient, address=addr, bind=False)
    await asyncio.sleep(_SETTLE)

    for i in range(backlog):
        await push.send(CreditReturn(credit=_credit(i)))
    await asyncio.sleep(0.1)  # let the backlog settle in the PULL recv buffer

    assert received == []  # reader not started -> nothing drained yet

    await pull.start()  # bootstrap drain must clear the pre-queued backlog
    await asyncio.wait_for(got_all.wait(), timeout=2.0)
    assert sorted(received) == list(range(backlog))


async def test_dealer_router_bidirectional_roundtrip(client_factory, new_addr):
    """Real DEALER<->ROUTER round-trip exercises the shared-FD both-direction drive.

    The worker DEALER sends (recv on the router side) and the router replies by
    identity (recv on the dealer side), all driven off each socket's single raw FD.
    """
    addr = new_addr()
    router_inbox: asyncio.Queue = asyncio.Queue()
    dealer_inbox: asyncio.Queue = asyncio.Queue()

    async def router_handler(identity: str, msg) -> None:
        await router_inbox.put((identity, msg))

    router = await client_factory(
        ZMQStreamingRouterClient,
        address=addr,
        bind=True,
        start=True,
        receiver=router_handler,
    )
    dealer = await client_factory(
        ZMQStreamingDealerClient,
        address=addr,
        bind=False,
        start=True,
        receiver=dealer_inbox.put,
        identity="worker-1",
    )
    await asyncio.sleep(_SETTLE)

    # DEALER -> ROUTER
    await dealer.send(WorkerReady(worker_id="worker-1"))
    identity, msg = await asyncio.wait_for(router_inbox.get(), timeout=2.0)
    assert identity == "worker-1"
    assert isinstance(msg, WorkerReady)
    assert msg.worker_id == "worker-1"

    # ROUTER -> DEALER (routed back by the captured identity)
    await router.send_to(identity, _credit(7))
    reply = await asyncio.wait_for(dealer_inbox.get(), timeout=2.0)
    assert isinstance(reply, Credit)
    assert reply.id == 7


async def test_stop_is_clean_with_messages_in_flight(client_factory, new_addr):
    """Stopping the PULL client mid-stream tears down cleanly (no raised error,
    FD reader removed) even while a sender is still pushing."""
    addr = new_addr()
    received: list[int] = []

    async def handler(msg: CreditReturn) -> None:
        received.append(msg.credit.id)

    pull = await client_factory(
        ZMQStreamingPullClient, address=addr, bind=True, start=True, receiver=handler
    )
    push = await client_factory(ZMQStreamingPushClient, address=addr, bind=False)
    await asyncio.sleep(_SETTLE)

    for i in range(200):
        await push.send(CreditReturn(credit=_credit(i)))

    # Stop while the drain may still have work queued; must not raise.
    await pull.stop()
    assert pull._fd_reader is None


async def test_poison_message_does_not_strand_the_drain(client_factory, new_addr):
    """A frame that fails to decode is reported but does NOT stall the reader.

    Valid messages queued behind an undecodable one must still be delivered. We
    pre-queue [v0, v1, bogus, v2, v3] before start(), so the bootstrap drain hits
    the poison in a single pass: the FD is edge-triggered and POLLIN stays set,
    so if ``_pump`` returned without re-arming after the decode error, v2/v3 would
    be stranded forever (this test would time out). The re-arm keeps the drain going.
    """
    addr = new_addr()
    received: list[int] = []
    done = asyncio.Event()

    async def handler(msg: CreditReturn) -> None:
        received.append(msg.credit.id)
        if len(received) >= 4:
            done.set()

    pull = await client_factory(
        ZMQStreamingPullClient, address=addr, bind=True, start=False, receiver=handler
    )
    push = await client_factory(ZMQStreamingPushClient, address=addr, bind=False)
    await asyncio.sleep(_SETTLE)

    await push.send(CreditReturn(credit=_credit(0)))
    await push.send(CreditReturn(credit=_credit(1)))
    await push.send(_Bogus(n=99))  # undecodable as WorkerToRouterMessage
    await push.send(CreditReturn(credit=_credit(2)))
    await push.send(CreditReturn(credit=_credit(3)))
    await asyncio.sleep(0.1)  # everything queued in the PULL buffer

    await pull.start()  # one bootstrap drain pass hits the poison mid-stream
    await asyncio.wait_for(done.wait(), timeout=3.0)
    assert received == [0, 1, 2, 3]
