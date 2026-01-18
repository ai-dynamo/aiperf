# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for push_client.py - ZMQPushClient class.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import zmq
import zmq.asyncio

from aiperf.common.enums import MessageType
from aiperf.common.environment import Environment
from aiperf.common.exceptions import CommunicationError, NotInitializedError
from aiperf.common.messages import Message
from aiperf.zmq.push_client import ZMQPushClient


class TestZMQPushClientInitialization:
    """Test ZMQPushClient initialization."""

    def test_init_creates_push_socket(self, mock_zmq_context):
        """Test that initialization creates a PUSH socket."""
        client = ZMQPushClient(address="tcp://127.0.0.1:5555", bind=True)

        assert client.socket_type == zmq.SocketType.PUSH

    def test_init_with_various_addresses(self, address_and_bind, mock_zmq_context):
        """Test initialization with various address types."""
        address, bind = address_and_bind
        client = ZMQPushClient(address=address, bind=bind)

        assert client.address == address
        assert client.bind == bind


class TestZMQPushClientPush:
    """Test ZMQPushClient.push method."""

    @pytest.mark.asyncio
    async def test_push_sends_message(
        self, mock_zmq_socket, mock_zmq_context, sample_message
    ):
        """Test that push sends the message."""
        client = ZMQPushClient(address="tcp://127.0.0.1:5555", bind=True)
        await client.initialize()

        await client.push(sample_message)

        mock_zmq_socket.send.assert_called_once()
        sent_data = mock_zmq_socket.send.call_args[0][0]
        # sent_data is bytes, so decode to string for comparison
        assert sample_message.message_type.value.encode() in sent_data

    @pytest.mark.asyncio
    async def test_push_serializes_message_correctly(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that push serializes message as JSON."""
        client = ZMQPushClient(address="tcp://127.0.0.1:5555", bind=True)
        await client.initialize()

        message = Message(message_type=MessageType.HEARTBEAT, request_id="test-123")

        await client.push(message)

        sent_data = mock_zmq_socket.send.call_args[0][0]
        # Verify it's valid JSON containing our data (sent_data is bytes)
        sent_str = sent_data.decode()
        assert (
            '"message_type":"heartbeat"' in sent_str
            or '"message_type": "heartbeat"' in sent_str
        )
        assert "test-123" in sent_str

    @pytest.mark.asyncio
    async def test_push_retries_on_zmq_again(self, mock_zmq_context):
        """Test that push retries on zmq.Again (timeout)."""
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        # First call fails, second succeeds
        mock_socket.send = AsyncMock(side_effect=[zmq.Again(), None])
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        with (
            patch("zmq.asyncio.Context.instance", return_value=mock_zmq_context),
            patch.object(Environment.ZMQ, "PUSH_RETRY_DELAY", 0.01),
        ):
            client = ZMQPushClient(address="tcp://127.0.0.1:5555", bind=True)
            await client.initialize()

            message = Message(message_type=MessageType.HEARTBEAT)

            # Should succeed after retry
            await client.push(message)

            # Verify it was called twice (once failed, once succeeded)
            assert mock_socket.send.call_count == 2

    @pytest.mark.asyncio
    async def test_push_raises_communication_error_after_max_retries(
        self, mock_zmq_context
    ):
        """Test that push raises CommunicationError after max retries."""
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.send = AsyncMock(side_effect=zmq.Again())
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        with (
            patch("zmq.asyncio.Context.instance", return_value=mock_zmq_context),
            patch.object(Environment.ZMQ, "PUSH_MAX_RETRIES", 2),
            patch.object(Environment.ZMQ, "PUSH_RETRY_DELAY", 0.01),
        ):  # fmt: skip
            client = ZMQPushClient(address="tcp://127.0.0.1:5555", bind=True)
            await client.initialize()

            message = Message(message_type=MessageType.HEARTBEAT)

            with pytest.raises(CommunicationError, match="Failed to push data after"):
                await client.push(message)

    @pytest.mark.asyncio
    async def test_push_handles_graceful_errors(self, push_test_helper, graceful_error):
        """Test that push handles graceful errors (CancelledError, ContextTerminated)."""
        async with push_test_helper.create_client(
            send_side_effect=graceful_error
        ) as client:
            message = Message(message_type=MessageType.HEARTBEAT)

            # Should not raise, just return
            await client.push(message)

    @pytest.mark.asyncio
    async def test_push_raises_communication_error_on_other_exceptions(
        self, push_test_helper, non_graceful_error
    ):
        """Test that push raises CommunicationError for other exceptions."""
        async with push_test_helper.create_client(
            send_side_effect=non_graceful_error
        ) as client:
            message = Message(message_type=MessageType.HEARTBEAT)

            with pytest.raises(CommunicationError, match="Failed to push data"):
                await client.push(message)


class TestZMQPushClientEdgeCases:
    """Test edge cases for ZMQPushClient."""

    @pytest.mark.asyncio
    async def test_push_before_initialization_raises_error(
        self, mock_zmq_context, sample_message
    ):
        """Test that push before initialization raises an error."""
        client = ZMQPushClient(address="tcp://127.0.0.1:5555", bind=True)

        # Don't initialize
        client.socket = None
        with pytest.raises(NotInitializedError):
            await client.push(sample_message)

    @pytest.mark.asyncio
    async def test_multiple_sequential_pushes(self, mock_zmq_socket, mock_zmq_context):
        """Test multiple sequential push operations."""
        client = ZMQPushClient(address="tcp://127.0.0.1:5555", bind=True)
        await client.initialize()

        messages = [
            Message(message_type=MessageType.HEARTBEAT, request_id=f"req-{i}")
            for i in range(5)
        ]

        for msg in messages:
            await client.push(msg)

        assert mock_zmq_socket.send.call_count == 5
