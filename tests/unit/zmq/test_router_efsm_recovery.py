# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ROUTER send failure is a peer problem, never a socket problem.

A DEALER that disconnected between receive and reply (EHOSTUNREACH / ENOTCONN)
leaves the socket valid for every other peer, and libzmq 4.3.5 gives ROUTER no
way to break its own state machine on send: EFSM comes only from REQ/REP and
the security mechanisms, and ``router_t::xsend`` fails only under
ROUTER_MANDATORY, which this codebase never sets.
"""

from unittest.mock import MagicMock

import pytest
import zmq

from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient


@pytest.fixture
def router():
    client = ZMQStreamingRouterClient.__new__(ZMQStreamingRouterClient)
    client._stop_requested_event = MagicMock(is_set=lambda: False)
    client._fd_reader = MagicMock()
    client._start_fd_reader = MagicMock()
    client.warning = MagicMock()
    client.exception = MagicMock()
    client.debug = MagicMock()
    return client


class TestSendFailureRecovery:
    @pytest.mark.parametrize("errno", [zmq.EHOSTUNREACH, zmq.ENOTCONN])
    @pytest.mark.asyncio
    async def test_peer_gone_is_logged_and_socket_untouched(self, router, errno):
        """One departed peer must not disturb the socket serving everyone else."""
        await router._recover_from_send_failure("worker-1", zmq.ZMQError(errno, "gone"))
        router.warning.assert_called()
        router._fd_reader.stop.assert_not_called()
        router._start_fd_reader.assert_not_called()

    @pytest.mark.asyncio
    async def test_efsm_is_not_treated_as_recoverable(self, router):
        """EFSM is unreachable for ROUTER; it must not trigger a rebuild."""
        await router._recover_from_send_failure(
            "worker-1", zmq.ZMQError(zmq.EFSM, "state machine")
        )
        router._start_fd_reader.assert_not_called()
        router.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_zmq_error_is_ignored(self, router):
        await router._recover_from_send_failure("worker-1", RuntimeError("boom"))
        router.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_recovery_while_stopping(self, router):
        router._stop_requested_event = MagicMock(is_set=lambda: True)
        await router._recover_from_send_failure(
            "worker-1", zmq.ZMQError(zmq.EHOSTUNREACH, "gone")
        )
        router.warning.assert_not_called()
