# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A broken ROUTER socket must be recreated, not logged and abandoned.

Two send failures look alike and are not: a DEALER that disconnected between
receive and reply (EHOSTUNREACH / ENOTCONN) leaves the socket valid for every
other peer, while a partial multipart send leaves the socket state machine in
EFSM. Treating EFSM as "log and continue" wedges the credit ROUTER for the
whole fleet, permanently, with no error after the first.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import zmq

from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient


@pytest.fixture
def router():
    client = ZMQStreamingRouterClient.__new__(ZMQStreamingRouterClient)
    client._stop_requested_event = MagicMock(is_set=lambda: False)
    client._recreate_socket = AsyncMock()
    client._fd_reader = MagicMock()
    client._start_fd_reader = MagicMock()
    client.warning = MagicMock()
    client.exception = MagicMock()
    client.debug = MagicMock()
    return client


class TestSendFailureRecovery:
    @pytest.mark.asyncio
    async def test_efsm_recreates_the_socket(self, router):
        await router._recover_from_send_failure(
            "worker-1", zmq.ZMQError(zmq.EFSM, "state machine")
        )
        router._recreate_socket.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_efsm_rebuilds_the_fd_reader(self, router):
        """The old FD is closed; leaving the reader on it leaks and goes deaf."""
        await router._recover_from_send_failure(
            "worker-1", zmq.ZMQError(zmq.EFSM, "state machine")
        )
        router._fd_reader.stop.assert_called_once()
        router._start_fd_reader.assert_called_once()

    @pytest.mark.parametrize("errno", [zmq.EHOSTUNREACH, zmq.ENOTCONN])
    @pytest.mark.asyncio
    async def test_peer_gone_does_not_recreate(self, router, errno):
        """One departed peer must not disturb the socket serving everyone else."""
        await router._recover_from_send_failure("worker-1", zmq.ZMQError(errno, "gone"))
        router._recreate_socket.assert_not_awaited()
        router.warning.assert_called()

    @pytest.mark.asyncio
    async def test_non_zmq_error_is_ignored(self, router):
        await router._recover_from_send_failure("worker-1", RuntimeError("boom"))
        router._recreate_socket.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_recovery_while_stopping(self, router):
        router._stop_requested_event = MagicMock(is_set=lambda: True)
        await router._recover_from_send_failure(
            "worker-1", zmq.ZMQError(zmq.EFSM, "state machine")
        )
        router._recreate_socket.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_recreate_is_contained(self, router):
        """A failed recreate must not escape into the dispatcher."""
        router._recreate_socket.side_effect = zmq.ZMQError(zmq.EINVAL, "nope")
        await router._recover_from_send_failure(
            "worker-1", zmq.ZMQError(zmq.EFSM, "state machine")
        )
        router.exception.assert_called()
