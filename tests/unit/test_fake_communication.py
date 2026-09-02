# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-channel behavior of the in-process FakeCommunication harness.

These pin the two properties later DEALER/ROUTER control-channel work depends
on: a dealer registry keyed by (address, identity) so a service's credit and
control channels do not collide, and a ROUTER reply path that mirrors
``ZMQStreamingRouterClient`` closely enough that a test passing here would also
pass over real ZMQ.
"""

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest

from aiperf.common.control_structs import (
    Command,
    CommandOk,
    Registration,
    RegistrationAck,
)
from tests.harness.fake_communication import FakeCommunication, FakeCommunicationBus


async def _append(sink: list[Any], message: Any) -> None:
    sink.append(message)


@pytest.fixture(autouse=True)
def _isolated_bus() -> Iterator[None]:
    """The shared bus is class-level state; a leak cross-wires unrelated tests."""
    FakeCommunication.set_shared_bus(FakeCommunicationBus())
    yield
    FakeCommunication.clear_shared_bus()


@pytest.mark.asyncio
async def test_two_dealers_with_same_identity_on_different_addresses_do_not_collide() -> (
    None
):
    """A worker holds both a credit DEALER and a control DEALER under the same
    service id; keying the registry by identity alone silently cross-wires them."""
    comm = FakeCommunication()
    credit_router = comm.create_streaming_router_client("fake://credit", bind=True)
    control_router = comm.create_streaming_router_client("fake://control", bind=True)
    credit_dealer = comm.create_streaming_dealer_client("fake://credit", identity="w-1")
    control_dealer = comm.create_streaming_dealer_client(
        "fake://control", identity="w-1"
    )

    credit_seen: list[Any] = []
    control_seen: list[Any] = []
    credit_dealer.register_receiver(lambda m: _append(credit_seen, m))
    control_dealer.register_receiver(lambda m: _append(control_seen, m))

    await comm.initialize()
    await comm.start()

    await control_router.send_to("w-1", RegistrationAck(rid="r-1"))
    await asyncio.sleep(0)

    assert control_seen == [RegistrationAck(rid="r-1")]
    assert credit_seen == []

    # Load-bearing: under bare-identity keying the second dealer registration
    # overwrites the first, so this send lands on the control dealer and the
    # assertions above still hold. Only exercising both routers detects it.
    await credit_router.send_to("w-1", RegistrationAck(rid="r-2"))
    await asyncio.sleep(0)

    assert credit_seen == [RegistrationAck(rid="r-2")]
    assert control_seen == [RegistrationAck(rid="r-1")]


@pytest.mark.asyncio
async def test_request_to_resolves_on_the_reply_the_dealer_sends_back() -> None:
    """``request_to`` resolves on a message the dealer sends on its own socket.

    Deliberately NOT on the dealer handler's return value:
    ``ZMQStreamingDealerClient._dispatch_dealer`` discards that, so services
    reply with an explicit ``send``.
    """
    comm = FakeCommunication()
    router = comm.create_streaming_router_client("fake://control", bind=True)
    dealer = comm.create_streaming_dealer_client("fake://control", identity="svc-1")

    async def dealer_handler(message: Any) -> None:
        assert isinstance(message, Command)
        await dealer.send(
            CommandOk(cid=message.cid, cmd=message.cmd, sid="svc-1", payload=b"42")
        )

    dealer.register_receiver(dealer_handler)
    await comm.initialize()
    await comm.start()

    response = await router.request_to(
        "svc-1", Command(cid="c-1", cmd="get_pod_states"), timeout=1.0
    )
    assert response == CommandOk(
        cid="c-1", cmd="get_pod_states", sid="svc-1", payload=b"42"
    )


@pytest.mark.asyncio
async def test_request_to_ignores_a_dealer_handler_return_value_and_times_out() -> None:
    """The fake-side twin of ``test_dealer_register_receiver_stays_fire_and_forget``.

    A DEALER receiver returning a struct is a no-op over real ZMQ. If the fake
    honored it, every later control-channel test would pass against production
    code that can never reply, so returning must be observably different from
    sending.
    """
    comm = FakeCommunication()
    router = comm.create_streaming_router_client("fake://control", bind=True)
    dealer = comm.create_streaming_dealer_client("fake://control", identity="svc-1")

    async def returning_handler(message: Any) -> Any:
        return CommandOk(cid=message.cid, cmd=message.cmd, sid="svc-1")

    dealer.register_receiver(returning_handler)
    await comm.initialize()
    await comm.start()

    with pytest.raises(TimeoutError):
        await router.request_to(
            "svc-1", Command(cid="c-2", cmd="get_pod_states"), timeout=0.05
        )


@pytest.mark.asyncio
async def test_dealer_request_resolves_on_router_reply_matched_by_rid() -> None:
    comm = FakeCommunication()
    router = comm.create_streaming_router_client("fake://control", bind=True)
    dealer = comm.create_streaming_dealer_client("fake://control", identity="svc-1")

    async def router_handler(identity: str, message: Any) -> Any:
        assert identity == "svc-1"
        return RegistrationAck(rid=message.rid)

    router.register_receiver(router_handler)
    await comm.initialize()
    await comm.start()

    ack = await dealer.request(
        Registration(sid="svc-1", rid="r-9", stype="worker", state="running"),
        timeout=1.0,
    )
    assert ack == RegistrationAck(rid="r-9")
