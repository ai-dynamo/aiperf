# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for streaming ZMQ clients - ZMQStreamingRouterClient and ZMQStreamingDealerClient.

Tests focus on behavior and functionality, not implementation details.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import msgspec.msgpack
import pytest
import zmq.asyncio

from aiperf.common.exceptions import NotInitializedError
from aiperf.credit.messages import (
    RouterToWorkerMessage,
    WorkerReady,
    WorkerToRouterMessage,
)
from aiperf.credit.structs import Credit
from aiperf.zmq.streaming_dealer_client import ZMQStreamingDealerClient
from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient

# ============================================================================
# ZMQStreamingRouterClient Tests
# ============================================================================


class TestStreamingRouterClientInitialization:
    """Test ZMQStreamingRouterClient initialization."""

    def test_creates_router_socket(self, mock_zmq_context):
        """Should create a ROUTER socket."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)

        assert client.socket_type == zmq.SocketType.ROUTER

    @pytest.mark.parametrize(
        "address,bind",
        [
            ("tcp://*:5555", True),
            ("tcp://localhost:5555", False),
            ("ipc:///tmp/router.ipc", True),
            ("ipc:///tmp/router.ipc", False),
        ],
        ids=["tcp_bind", "tcp_connect", "ipc_bind", "ipc_connect"],
    )  # fmt: skip
    def test_supports_various_transports(self, address, bind, mock_zmq_context):
        """Should support both TCP and IPC transports."""
        client = ZMQStreamingRouterClient(address=address, bind=bind)

        assert client.address == address
        assert client.bind == bind


class TestStreamingRouterClientSendTo:
    """Test ZMQStreamingRouterClient.send_to method."""

    @pytest.mark.asyncio
    async def test_sends_credit_to_specific_identity(
        self, mock_zmq_socket, mock_zmq_context, sample_credit
    ):
        """Should send credit to specific DEALER by identity."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        await client.initialize()

        await client.send_to("worker-42", sample_credit)

        # Verify multipart message with correct envelope
        mock_zmq_socket.send_multipart.assert_called_once()
        call_args = mock_zmq_socket.send_multipart.call_args[0][0]

        assert call_args[0] == b"worker-42"  # Identity
        decoded = msgspec.msgpack.decode(call_args[1], type=Credit)
        assert decoded.id == sample_credit.id

    @pytest.mark.asyncio
    async def test_raises_if_not_initialized(self, mock_zmq_context, sample_credit):
        """Should raise NotInitializedError if socket not initialized."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)

        with pytest.raises(NotInitializedError):
            await client.send_to("worker-1", sample_credit)


class TestStreamingRouterClientReceiver:
    """Test ZMQStreamingRouterClient message receiving."""

    @pytest.mark.asyncio
    async def test_receives_worker_ready_and_calls_handler(
        self, mock_zmq_context, sample_worker_ready, wait_for_background_task
    ):
        """Should receive messages from DEALERs and call registered handler."""
        handler_called = asyncio.Event()
        received_identity = None
        received_message = None

        async def handler(identity: str, message: WorkerToRouterMessage):
            nonlocal received_identity, received_message
            received_identity = identity
            received_message = message
            handler_called.set()

        # Setup mock socket to return one message
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.send_multipart = AsyncMock()

        # Return message in ROUTER format: [identity, message_bytes]
        call_count = 0

        async def recv_multipart_handler():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.01)  # Small delay
                return [b"worker-1", msgspec.msgpack.encode(sample_worker_ready)]
            else:
                # Block forever on subsequent calls
                await asyncio.Future()

        mock_socket.recv_multipart = recv_multipart_handler
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        client.register_receiver(handler)
        await client.initialize()
        await client.start()

        try:
            # Wait for message to be received
            await asyncio.wait_for(handler_called.wait(), timeout=1.0)

            assert received_identity == "worker-1"
            assert isinstance(received_message, WorkerReady)
            assert received_message.worker_id == sample_worker_ready.worker_id
        finally:
            await client.stop()


# ============================================================================
# ZMQStreamingDealerClient Tests
# ============================================================================


class TestStreamingDealerClientInitialization:
    """Test ZMQStreamingDealerClient initialization."""

    def test_creates_dealer_socket(self, mock_zmq_context):
        """Should create a DEALER socket."""
        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )

        assert client.socket_type == zmq.SocketType.DEALER
        assert client.identity == "worker-1"

    @pytest.mark.asyncio
    async def test_sets_identity_in_socket_options(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Should set IDENTITY socket option for routing."""
        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-42"
        )
        await client.initialize()

        # Verify IDENTITY was set in socket options
        setsockopt_calls = mock_zmq_socket.setsockopt.call_args_list
        identity_calls = [
            call for call in setsockopt_calls if call[0][0] == zmq.IDENTITY
        ]

        assert len(identity_calls) == 1
        assert identity_calls[0][0][1] == b"worker-42"

    @pytest.mark.parametrize(
        "address,identity",
        [
            ("tcp://localhost:5555", "worker-1"),
            ("tcp://localhost:6666", "worker-2"),
            ("ipc:///tmp/router.ipc", "worker-3"),
        ],
        ids=["tcp_worker1", "tcp_worker2", "ipc_worker3"],
    )  # fmt: skip
    async def test_supports_various_transports(
        self, address, identity, mock_zmq_context
    ):
        """Should support both TCP and IPC transports."""
        client = ZMQStreamingDealerClient(address=address, identity=identity)

        assert client.address == address
        assert client.identity == identity


class TestStreamingDealerClientSend:
    """Test ZMQStreamingDealerClient.send method."""

    @pytest.mark.asyncio
    async def test_sends_struct_with_msgpack(
        self, mock_zmq_socket, mock_zmq_context, sample_worker_ready
    ):
        """Should send struct using msgpack and single-frame send."""
        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )
        await client.initialize()

        await client.send(sample_worker_ready)

        # Verify message was sent (DEALER uses send, not send_multipart)
        mock_zmq_socket.send.assert_called_once()
        call_args = mock_zmq_socket.send.call_args[0][0]

        decoded = msgspec.msgpack.decode(call_args, type=WorkerReady)
        assert decoded.worker_id == sample_worker_ready.worker_id

    @pytest.mark.asyncio
    async def test_raises_if_not_initialized(
        self, mock_zmq_context, sample_worker_ready
    ):
        """Should raise NotInitializedError if socket not initialized."""
        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )

        with pytest.raises(NotInitializedError):
            await client.send(sample_worker_ready)


class TestStreamingDealerClientReceiver:
    """Test ZMQStreamingDealerClient message receiving."""

    @pytest.mark.asyncio
    async def test_receives_credits_and_calls_handler(
        self, mock_zmq_context, sample_credit, wait_for_background_task
    ):
        """Should receive credits from ROUTER and call registered handler."""
        handler_called = asyncio.Event()
        received_message = None

        async def handler(message: RouterToWorkerMessage):
            nonlocal received_message
            received_message = message
            handler_called.set()

        # Setup mock socket to return one message
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.connect = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.send = AsyncMock()

        # DEALER receives using recv() not recv_multipart()
        call_count = 0

        async def recv_handler():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.01)
                return msgspec.msgpack.encode(sample_credit)
            else:
                await asyncio.Future()

        mock_socket.recv = recv_handler
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )
        client.register_receiver(handler)
        await client.initialize()
        await client.start()

        try:
            await asyncio.wait_for(handler_called.wait(), timeout=1.0)
            assert isinstance(received_message, Credit)
            assert received_message.id == sample_credit.id
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_receives_credit_with_msgpack_framing(
        self, mock_zmq_context, sample_credit, wait_for_background_task
    ):
        """Should handle msgpack encoded credits (DEALER uses recv, framing handled by ZMQ)."""
        handler_called = asyncio.Event()
        received_message = None

        async def handler(message: RouterToWorkerMessage):
            nonlocal received_message
            received_message = message
            handler_called.set()

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.connect = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.send = AsyncMock()

        # DEALER uses recv() - ZMQ handles framing automatically
        call_count = 0

        async def recv_handler():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.01)
                return msgspec.msgpack.encode(sample_credit)
            else:
                await asyncio.Future()

        mock_socket.recv = recv_handler
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )
        client.register_receiver(handler)
        await client.initialize()
        await client.start()

        try:
            await asyncio.wait_for(handler_called.wait(), timeout=1.0)
            assert isinstance(received_message, Credit)
            assert received_message.conversation_id == sample_credit.conversation_id
        finally:
            await client.stop()


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestStreamingClientsErrorHandling:
    """Test error handling in streaming clients."""

    @pytest.mark.asyncio
    async def test_router_handles_malformed_envelope(
        self, mock_zmq_context, wait_for_background_task
    ):
        """Should handle malformed ROUTER envelopes gracefully."""
        handler = AsyncMock()

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.send_multipart = AsyncMock()

        # Return malformed envelope (missing message part)
        call_count = 0

        async def recv_multipart_handler():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.01)
                return [b"worker-1"]  # Missing message bytes
            else:
                await asyncio.Future()

        mock_socket.recv_multipart = recv_multipart_handler
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        client.register_receiver(handler)
        await client.initialize()
        await client.start()

        try:
            await wait_for_background_task()
            # Should not crash, should log error and continue
            handler.assert_not_called()
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_dealer_handles_malformed_message(
        self, mock_zmq_context, wait_for_background_task
    ):
        """Should handle malformed messages gracefully."""
        handler = AsyncMock()

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.connect = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.send = AsyncMock()

        # Return invalid msgpack data
        call_count = 0

        async def recv_handler():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.01)
                return b"not valid msgpack"
            else:
                await asyncio.Future()

        mock_socket.recv = recv_handler
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )
        client.register_receiver(handler)
        await client.initialize()
        await client.start()

        try:
            await wait_for_background_task()
            # Should not crash, should log error and continue
            handler.assert_not_called()
        finally:
            await client.stop()


# ============================================================================
# Lifecycle Tests
# ============================================================================


class TestStreamingClientsLifecycle:
    """Test lifecycle behavior of streaming clients."""

    @pytest.mark.asyncio
    async def test_router_cleanup_on_stop(self, mock_zmq_context):
        """Should clean up resources on stop."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        client.register_receiver(AsyncMock())
        await client.initialize()
        await client.start()

        await client.stop()

        # Verify handlers cleared
        assert client._receiver_handler is None

    @pytest.mark.asyncio
    async def test_dealer_cleanup_on_stop(self, mock_zmq_context):
        """Should clean up resources on stop."""
        client = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )
        client.register_receiver(AsyncMock())
        await client.initialize()
        await client.start()

        await client.stop()

        # Verify handler cleared
        assert client._receiver_handler is None

    @pytest.mark.asyncio
    async def test_cannot_register_multiple_handlers(self, mock_zmq_context):
        """Should raise error if trying to register multiple handlers."""
        router = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        dealer = ZMQStreamingDealerClient(
            address="tcp://localhost:5555", identity="worker-1"
        )

        # First registration should work
        router.register_receiver(AsyncMock())
        dealer.register_receiver(AsyncMock())

        # Second registration should fail
        with pytest.raises(ValueError, match="already registered"):
            router.register_receiver(AsyncMock())

        with pytest.raises(ValueError, match="already registered"):
            dealer.register_receiver(AsyncMock())
