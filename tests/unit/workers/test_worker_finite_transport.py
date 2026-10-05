# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite transport notifications preserve capture order without blocking HTTP."""

import asyncio
import time
from types import MethodType, SimpleNamespace

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import CreditReturn, TransportDispatched
from aiperf.credit.structs import Credit, CreditContext
from aiperf.workers.clock_offset_tracker import ClockOffsetTracker
from aiperf.workers.worker import Worker


@pytest.mark.asyncio
async def test_start_notification_is_queued_and_eof_precedes_terminal_return() -> None:
    allow_start_send = asyncio.Event()
    sent: list[TransportDispatched | CreditReturn] = []

    async def send(message: TransportDispatched | CreditReturn) -> None:
        if isinstance(message, TransportDispatched):
            await allow_start_send.wait()
        sent.append(message)

    worker = SimpleNamespace(
        service_id="worker-1",
        _tracks_clock_offset=False,
        clock_offset_tracker=ClockOffsetTracker(),
        credit_return_push_client=SimpleNamespace(send=send),
        _transport_start_tasks={},
    )
    worker._snapshot_finite_clock_state = MethodType(
        Worker._snapshot_finite_clock_state, worker
    )
    credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="root",
        x_correlation_id="root-correlation",
        turn_index=0,
        num_turns=1,
        issued_at_ns=1,
        finite_replay=True,
    )
    context = CreditContext(credit=credit, drop_perf_ns=1)
    start_callback = Worker._make_transport_start_callback(worker, context)
    eof_callback = Worker._make_transport_eof_callback(worker, context)
    assert start_callback is not None and eof_callback is not None
    start_ns = time.perf_counter_ns()

    start_callback(start_ns)
    assert context.transport_start_sent
    assert sent == []
    eof_callback(start_ns + 1_000_000)
    return_task = asyncio.create_task(
        Worker._send_ordered_credit_return(worker, context)
    )
    await asyncio.sleep(0)
    assert sent == []

    allow_start_send.set()
    await return_task
    assert [type(message) for message in sent] == [TransportDispatched, CreditReturn]
    start_message, terminal_message = sent
    assert isinstance(start_message, TransportDispatched)
    assert isinstance(terminal_message, CreditReturn)
    assert terminal_message.transport_eof_wall_ns is not None
    assert (
        start_message.transport_start_wall_ns < terminal_message.transport_eof_wall_ns
    )
    assert terminal_message.error is None
    assert context.returned
