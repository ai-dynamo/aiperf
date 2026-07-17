# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for real-socket ZMQ transport tests.

Unlike ``tests/unit/zmq`` (which mocks the socket and the FD via the autouse
fixtures in its own conftest), these tests drive the REAL ZMQ client classes
over a live ``libzmq`` transport. They share the process-wide
``zmq.asyncio.Context.instance()`` and talk over ``inproc://`` endpoints
(cross-platform, no ports or files), with real time (no looptime) so the
FD edge-trigger, re-arm, and bootstrap-drain behaviors are exercised against
the actual event loop and kernel objects.
"""

import asyncio
import contextlib
import itertools

import pytest

from aiperf.common.constants import IS_WINDOWS


@pytest.fixture(scope="session")
def event_loop_policy():
    """Force the selector loop on Windows so the FD-driver can register.

    The FD-driver calls ``loop.add_reader(getsockopt(zmq.FD), ...)``, which the
    default Windows ``ProactorEventLoop`` does not implement. Production switches
    to the selector policy in ``bootstrap``; these tests never run bootstrap, so
    mirror it here (no-op on Linux/macOS).
    """
    if IS_WINDOWS:
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


# Per-process counter -> unique inproc endpoint names. inproc namespaces are
# scoped to a Context, and each xdist worker is its own process with its own
# Context.instance(), so a per-process counter is collision-free.
_ADDR_COUNTER = itertools.count()


@pytest.fixture
def new_addr():
    """Return a factory yielding a fresh unique ``inproc://`` endpoint per call."""

    def _make() -> str:
        return f"inproc://aiperf-zmqtest-{next(_ADDR_COUNTER)}"

    return _make


@pytest.fixture
async def client_factory():
    """Create real ZMQ clients and tear them all down after the test.

    For ``inproc`` the bind side must exist before the connect side, so create
    ``bind=True`` clients (PULL/ROUTER) before ``bind=False`` ones (PUSH/DEALER).

    Args to the returned factory:
        cls: the client class to instantiate.
        address/bind: endpoint and bind flag.
        start: if True, ``await client.start()`` after init (installs the FD reader).
        receiver: optional handler passed to ``register_receiver`` before start.
        **kwargs: forwarded to the client constructor (e.g. ``identity=...``).
    """
    created = []

    async def _make(cls, *, address, bind, start=False, receiver=None, **kwargs):
        client = cls(address=address, bind=bind, **kwargs)
        await client.initialize()
        if receiver is not None:
            client.register_receiver(receiver)
        if start:
            await client.start()
        created.append(client)
        return client

    yield _make

    for client in reversed(created):
        with contextlib.suppress(Exception):  # best-effort teardown
            await client.stop()
