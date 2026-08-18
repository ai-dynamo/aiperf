# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Router half of the credit-channel RTT probe.

A worker measures baseline network latency by sending ``TimePing`` on its
credit DEALER and timing the ``TimePong`` echo. Without the echo the worker
stalls on every probe timeout and never establishes a baseline RTT, which is
what decomposes the measured clock offset into skew vs transit.
"""

from unittest.mock import AsyncMock

import pytest

from aiperf.credit.messages import TimePing, TimePong
from aiperf.credit.sticky_router import StickyCreditRouter


@pytest.mark.asyncio
async def test_time_ping_is_echoed_as_time_pong(benchmark_run) -> None:
    router = StickyCreditRouter(run=benchmark_run, service_id="test-router")
    router._router_client.send_to = AsyncMock()

    await router._handle_router_message(
        "worker-1", TimePing(sequence=3, sent_at_ns=1_234_567)
    )

    router._router_client.send_to.assert_awaited_once()
    worker_id, reply = router._router_client.send_to.await_args[0]
    assert worker_id == "worker-1"
    assert isinstance(reply, TimePong)
    # Both fields echo verbatim: the worker matches the reply to its probe and
    # computes RTT entirely against its own clock, so no router clock leaks in.
    assert reply.sequence == 3
    assert reply.sent_at_ns == 1_234_567


@pytest.mark.asyncio
async def test_time_ping_does_not_register_the_worker(benchmark_run) -> None:
    """A probing worker is not yet dispatchable, so it must not join routing."""
    router = StickyCreditRouter(run=benchmark_run, service_id="test-router")
    router._router_client.send_to = AsyncMock()

    await router._handle_router_message("worker-1", TimePing(sequence=0, sent_at_ns=1))

    assert "worker-1" not in router._workers
