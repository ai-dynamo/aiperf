# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the typed credit-return fan-in clients.

``ZMQStreamingPushClient`` (worker side, send-only) and
``ZMQStreamingPullClient`` (timing-manager side, recv-only) carry typed msgpack
structs over the dedicated credit-return PUSH/PULL channel. These cover the
send encode path, the FD-drain receive path, handler registration, and stop.
"""

import asyncio
from unittest.mock import PropertyMock, patch

import msgspec.msgpack
import pytest
import zmq.asyncio

from aiperf.common.exceptions import CommunicationError, NotInitializedError
from aiperf.credit.messages import CreditReturn, WorkerToRouterMessage
from aiperf.zmq.streaming_pull_client import ZMQStreamingPullClient
from aiperf.zmq.streaming_push_client import ZMQStreamingPushClient

# ============================================================================
# ZMQStreamingPushClient
# ============================================================================


class TestStreamingPushClient:
    """Send-only typed PUSH client."""

    def test_creates_push_socket(self, mock_zmq_context):
        """Should create a PUSH socket."""
        client = ZMQStreamingPushClient(address="tcp://localhost:5555", bind=False)
        assert client.socket_type == zmq.SocketType.PUSH

    def test_ignores_max_pull_concurrency(self, mock_zmq_context):
        """max_pull_concurrency is accepted for factory uniformity and ignored."""
        client = ZMQStreamingPushClient(
            address="tcp://localhost:5555", bind=False, max_pull_concurrency=7
        )
        assert client.socket_type == zmq.SocketType.PUSH

    @pytest.mark.asyncio
    async def test_send_encodes_and_sync_sends(
        self, mock_zmq_socket, mock_zmq_context, sample_credit_return
    ):
        """send() msgpack-encodes the struct and issues one NOBLOCK sync send."""
        client = ZMQStreamingPushClient(address="tcp://localhost:5555", bind=False)
        await client.initialize()

        await client.send(sample_credit_return)

        mock_zmq_socket._sync_send.assert_called_once()
        call = mock_zmq_socket._sync_send.call_args
        decoded = msgspec.msgpack.decode(call.args[0], type=WorkerToRouterMessage)
        assert isinstance(decoded, CreditReturn)
        assert decoded.credit.id == sample_credit_return.credit.id
        assert call.kwargs["flags"] == zmq.NOBLOCK
        assert call.kwargs["copy"] is False

    @pytest.mark.asyncio
    async def test_send_traces_when_enabled(
        self, mock_zmq_socket, mock_zmq_context, sample_worker_ready
    ):
        """With trace logging on, send() emits a trace without altering the send."""
        client = ZMQStreamingPushClient(address="tcp://localhost:5555", bind=False)
        await client.initialize()

        with patch.object(
            type(client), "is_trace_enabled", new_callable=PropertyMock
        ) as trace_on:
            trace_on.return_value = True
            with patch.object(client, "trace") as trace:
                await client.send(sample_worker_ready)

        trace.assert_called_once()
        mock_zmq_socket._sync_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_if_not_initialized(
        self, mock_zmq_context, sample_worker_ready
    ):
        """send() before initialize() raises NotInitializedError."""
        client = ZMQStreamingPushClient(address="tcp://localhost:5555", bind=False)
        with pytest.raises(NotInitializedError):
            await client.send(sample_worker_ready)

    @pytest.mark.asyncio
    async def test_send_retries_on_zmq_again(
        self, mock_zmq_socket, mock_zmq_context, sample_credit_return
    ):
        """A transient zmq.Again (IMMEDIATE=1 no-peer race) is retried, not dropped."""
        client = ZMQStreamingPushClient(address="tcp://localhost:5555", bind=False)
        await client.initialize()

        # First sync send hits Again (no connected peer yet), second succeeds.
        mock_zmq_socket._sync_send.side_effect = [zmq.Again(), None]

        await client.send(sample_credit_return)

        assert mock_zmq_socket._sync_send.call_count == 2

    @pytest.mark.asyncio
    async def test_send_raises_after_max_retries(
        self, mock_zmq_socket, mock_zmq_context, sample_credit_return
    ):
        """A persistently-Again send raises CommunicationError after exhausting retries."""
        client = ZMQStreamingPushClient(address="tcp://localhost:5555", bind=False)
        await client.initialize()

        mock_zmq_socket._sync_send.side_effect = zmq.Again()

        with pytest.raises(CommunicationError, match="after 2 retries"):
            await client.send(sample_credit_return, max_retries=2)

        # 1 initial attempt + 2 retries = 3 sync sends.
        assert mock_zmq_socket._sync_send.call_count == 3

    @pytest.mark.asyncio
    async def test_send_swallows_context_terminated(
        self, mock_zmq_socket, mock_zmq_context, sample_credit_return
    ):
        """A send racing socket/context teardown returns quietly (no raise, no retry)."""
        client = ZMQStreamingPushClient(address="tcp://localhost:5555", bind=False)
        await client.initialize()

        mock_zmq_socket._sync_send.side_effect = zmq.ContextTerminated()

        await client.send(sample_credit_return)  # must not raise

        assert mock_zmq_socket._sync_send.call_count == 1


# ============================================================================
# ZMQStreamingPullClient
# ============================================================================


class TestStreamingPullClientRegistration:
    """Handler registration on the recv-only PULL client."""

    def test_creates_pull_socket(self, mock_zmq_context):
        """Should create a PULL socket."""
        client = ZMQStreamingPullClient(address="tcp://localhost:5555", bind=True)
        assert client.socket_type == zmq.SocketType.PULL

    def test_register_receiver(self, mock_zmq_context):
        """Registering a handler stores it."""
        client = ZMQStreamingPullClient(address="tcp://localhost:5555", bind=True)

        async def handler(message: WorkerToRouterMessage) -> None:
            pass

        client.register_receiver(handler)
        assert client._receiver_handler is handler

    def test_register_duplicate_receiver_raises(self, mock_zmq_context):
        """A second register_receiver is a programming error."""
        client = ZMQStreamingPullClient(address="tcp://localhost:5555", bind=True)

        async def handler(message: WorkerToRouterMessage) -> None:
            pass

        client.register_receiver(handler)
        with pytest.raises(ValueError, match="already registered"):
            client.register_receiver(handler)


class TestStreamingPullClientReceiver:
    """FD-drain receive path on the PULL client."""

    @pytest.mark.asyncio
    async def test_receives_and_calls_handler(
        self, mock_zmq_socket, mock_zmq_context, sample_credit_return, fd_enqueue
    ):
        """A queued typed struct is decoded and delivered to the handler."""
        handler_called = asyncio.Event()
        received: WorkerToRouterMessage | None = None

        async def handler(message: WorkerToRouterMessage) -> None:
            nonlocal received
            received = message
            handler_called.set()

        fd_enqueue(
            mock_zmq_socket, messages=[msgspec.msgpack.encode(sample_credit_return)]
        )

        client = ZMQStreamingPullClient(address="tcp://localhost:5555", bind=True)
        client.register_receiver(handler)
        await client.initialize()
        await client.start()

        try:
            await asyncio.wait_for(handler_called.wait(), timeout=1.0)
            assert isinstance(received, CreditReturn)
            assert received.credit.id == sample_credit_return.credit.id
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_warns_when_no_handler_registered(
        self, mock_zmq_socket, mock_zmq_context, sample_worker_ready, fd_enqueue
    ):
        """A message with no registered handler logs a warning and does not crash."""
        fd_enqueue(
            mock_zmq_socket, messages=[msgspec.msgpack.encode(sample_worker_ready)]
        )

        client = ZMQStreamingPullClient(address="tcp://localhost:5555", bind=True)
        await client.initialize()

        warned = asyncio.Event()
        with patch.object(client, "warning", side_effect=lambda *a, **k: warned.set()):
            await client.start()
            await asyncio.wait_for(warned.wait(), timeout=1.0)
            await client.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_reader_and_handler(
        self, mock_zmq_socket, mock_zmq_context, sample_credit_return, fd_enqueue
    ):
        """stop() tears down the FD reader and clears the handler."""
        delivered = asyncio.Event()

        async def handler(message: WorkerToRouterMessage) -> None:
            delivered.set()

        fd_enqueue(
            mock_zmq_socket, messages=[msgspec.msgpack.encode(sample_credit_return)]
        )

        client = ZMQStreamingPullClient(address="tcp://localhost:5555", bind=True)
        client.register_receiver(handler)
        await client.initialize()
        await client.start()
        # Waiting for a delivery guarantees the @background_task has installed
        # the FD reader, avoiding a race on start() scheduling.
        await asyncio.wait_for(delivered.wait(), timeout=1.0)
        assert client._fd_reader is not None

        await client.stop()

        assert client._fd_reader is None
        assert client._receiver_handler is None
