# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness probe for the credit-RETURN half of the dual-channel credit protocol.

Credits are dispatched controller -> worker on ROUTER/DEALER
(``CommAddress.CREDIT_ROUTER``) and returned worker -> controller on a separate
PUSH/PULL fan-in (``CommAddress.CREDIT_RETURN``). The two sockets come up
independently -- the router binds ROUTER before PULL, and each worker connects
each socket on its own schedule -- so a DEALER that has handshaked proves
nothing about the PUSH side. A worker that announces ``WorkerDispatchable`` off
DEALER liveness alone can therefore be routed credits it has no way to return.

``IMMEDIATE=1`` (set for every socket in :class:`BaseZMQClient`) is what makes
the probe meaningful: libzmq refuses a send with ``zmq.Again`` while no peer
pipe exists, and accepts it the moment one does. So a single NOBLOCK send is a
direct test of "is there a live connection to the PULL fan-in", with no reply
needed from the router -- which matters because PUSH/PULL is unidirectional and
the router has no way to answer.

The probe frame is a real ``WorkerConnected``: the router already handles it on
the PULL path (idempotently, as set insertion), and sending it here is what
finally makes the message truthful -- it announces that the *return* path is
connected, which the DEALER copy never established.
"""

import asyncio
import math

import msgspec
import zmq

from aiperf.common.models.base_models import msgspec_enc_hook
from aiperf.common.protocols import StreamingPushClientProtocol
from aiperf.credit.messages import WorkerConnected

# Same wire as the streaming PUSH/DEALER clients: typed msgpack, no envelope.
_encoder = msgspec.msgpack.Encoder(enc_hook=msgspec_enc_hook)


async def probe_return_channel(
    push_client: StreamingPushClientProtocol,
    *,
    worker_id: str,
    budget: float,
    retry_delay: float,
) -> bool:
    """Wait until the credit-return PUSH channel has a live peer.

    Retries a NOBLOCK send of ``WorkerConnected`` until libzmq accepts it or the
    attempt budget is exhausted. Attempts are counted rather than clock-timed so
    the bound holds under virtual event-loop time as well as real time.

    Args:
        push_client: The worker's credit-return PUSH client.
        worker_id: Identity stamped into the probe frame (the return channel has
            no ZMQ envelope identity).
        budget: Total seconds to spend probing. 0 disables the probe.
        retry_delay: Seconds between attempts; with ``budget`` it sets the
            attempt count.

    Returns:
        True if the channel accepted the probe (or there is nothing to probe -- a
        non-ZMQ transport has no separate return socket to fail). False if the
        budget was exhausted or the socket errored.
    """
    socket = getattr(push_client, "socket", None)
    if budget <= 0 or socket is None:
        return True

    data = _encoder.encode(WorkerConnected(worker_id=worker_id))
    attempts = max(1, math.ceil(budget / retry_delay))
    for attempt in range(attempts):
        try:
            zmq.Socket.send(socket, data, flags=zmq.NOBLOCK, copy=False)
            return True
        except zmq.Again:
            pass
        except zmq.ContextTerminated:
            return False
        except zmq.ZMQError:
            # A closed or otherwise broken socket will not heal by retrying.
            return False
        if attempt < attempts - 1:
            await asyncio.sleep(retry_delay)
    return False
