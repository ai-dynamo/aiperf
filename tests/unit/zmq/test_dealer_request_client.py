# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for dealer_request_client.py - ZMQDealerRequestClient class.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import zmq
import zmq.asyncio

from aiperf.common.enums import LifecycleState, MessageType
from aiperf.common.exceptions import CommunicationError
from aiperf.common.messages import HeartbeatMessage, Message
from aiperf.zmq.dealer_request_client import ZMQDealerRequestClient


class TestZMQDealerRequestClientInitialization:
    """Test ZMQDealerRequestClient initialization."""

    def test_init_creates_dealer_socket(self, mock_zmq_context):
        """Test that initialization creates a DEALER socket."""
        client = ZMQDealerRequestClient(address="tcp://127.0.0.1:5555", bind=False)

        assert client.socket_type == zmq.SocketType.DEALER
        assert client.request_callbacks == {}

    def test_init_with_various_addresses(self, address_and_bind, mock_zmq_context):
        """Test initialization with various address types."""
        address, bind = address_and_bind
        client = ZMQDealerRequestClient(address=address, bind=bind)

        assert client.address == address
        assert client.bind == bind


class TestZMQDealerRequestClientRequestAsync:
    """Test ZMQDealerRequestClient.request_async method."""

    @pytest.mark.asyncio
    async def test_request_async_sends_message(
        self, mock_zmq_socket, mock_zmq_context, sample_message
    ):
        """Test that request_async sends the message."""
        # No need to patch - autouse fixture handles this
        client = ZMQDealerRequestClient(address="tcp://127.0.0.1:5555", bind=False)
        await client.initialize()

        callback = AsyncMock()
        await client.request_async(sample_message, callback)

        mock_zmq_socket.send.assert_called_once()
        sent_data = mock_zmq_socket.send.call_args[0][0]
        # sent_data is bytes, so decode to string for comparison
        assert sample_message.request_id in sent_data.decode()

    @pytest.mark.asyncio
    async def test_request_async_generates_request_id_if_missing(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that request_async generates request_id if not provided."""
        # No need to patch - autouse fixture handles this
        client = ZMQDealerRequestClient(address="tcp://127.0.0.1:5555", bind=False)
        await client.initialize()

        message = Message(message_type=MessageType.HEARTBEAT)
        assert message.request_id is None

        callback = AsyncMock()
        await client.request_async(message, callback)

        # request_id should now be set
        assert message.request_id is not None

    @pytest.mark.asyncio
    async def test_request_async_registers_callback(
        self, mock_zmq_socket, mock_zmq_context, sample_message
    ):
        """Test that request_async registers the callback."""
        # No need to patch - autouse fixture handles this
        client = ZMQDealerRequestClient(address="tcp://127.0.0.1:5555", bind=False)
        await client.initialize()

        callback = AsyncMock()
        await client.request_async(sample_message, callback)

        assert sample_message.request_id in client.request_callbacks
        assert client.request_callbacks[sample_message.request_id] == callback

    @pytest.mark.asyncio
    async def test_request_async_raises_on_non_message_type(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that request_async raises TypeError for non-Message objects."""
        # No need to patch - autouse fixture handles this
        client = ZMQDealerRequestClient(address="tcp://127.0.0.1:5555", bind=False)
        await client.initialize()

        callback = AsyncMock()
        with pytest.raises(TypeError, match="must be an instance of Message"):
            await client.request_async("not a message", callback)

    @pytest.mark.asyncio
    async def test_request_async_raises_communication_error_on_send_failure(
        self, dealer_test_helper
    ):
        """Test that request_async raises CommunicationError on send failure."""
        async with dealer_test_helper.create_client(
            send_side_effect=Exception("Send failed")
        ) as client:
            message = Message(message_type=MessageType.HEARTBEAT, request_id="test-123")
            callback = AsyncMock()

            with pytest.raises(CommunicationError, match="Exception sending request"):
                await client.request_async(message, callback)


class TestZMQDealerRequestClientRequest:
    """Test ZMQDealerRequestClient.request method (with response waiting)."""

    @pytest.mark.asyncio
    async def test_request_sends_and_waits_for_response(
        self, mock_zmq_context, sample_message, wait_for_background_task
    ):
        """Test that request sends message and waits for response."""
        response_message = HeartbeatMessage(
            service_id="test-service",
            state=LifecycleState.RUNNING,
            service_type="test",
            request_id=sample_message.request_id,
        )

        # Create a mock socket that only returns response after send is called
        response_ready = asyncio.Event()

        async def mock_send(*args, **kwargs):
            response_ready.set()

        async def mock_recv(*args, **kwargs):
            if response_ready.is_set():
                response_ready.clear()
                return response_message.to_json_bytes()
            raise zmq.Again()

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.send = AsyncMock(side_effect=mock_send)
        mock_socket.recv = AsyncMock(side_effect=mock_recv)
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQDealerRequestClient(address="tcp://127.0.0.1:5555", bind=False)
        await client.initialize()
        await client.start()
        await wait_for_background_task()

        try:
            response = await client.request(sample_message, timeout=1.0)
            assert response is not None
            assert response.request_id == sample_message.request_id
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_request_times_out_if_no_response(
        self, dealer_test_helper, sample_message
    ):
        """Test that request times out if no response is received."""
        async with dealer_test_helper.create_client(auto_start=True) as client:
            with pytest.raises(asyncio.TimeoutError):
                await client.request(sample_message, timeout=0.1)


class TestZMQDealerRequestClientReceiverExceptions:
    """Test dealer receiver task exception handling paths."""

    @pytest.mark.asyncio
    async def test_receiver_handles_generic_exception(
        self, dealer_test_helper, wait_for_background_task
    ):
        """Test that receiver task handles generic exceptions without crashing."""
        async with dealer_test_helper.create_client(
            auto_start=True,
            recv_side_effect=[RuntimeError("Unexpected error")],
        ) as _client:
            await wait_for_background_task()
            # Should not crash - exception is logged and task continues
