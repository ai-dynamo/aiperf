# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for sub_client.py - ZMQSubClient class.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import zmq
import zmq.asyncio

from aiperf.common.enums import (
    CommandResponseStatus,
    CommandType,
    LifecycleState,
    MessageType,
)
from aiperf.common.exceptions import CommunicationError
from aiperf.common.messages import (
    CommandMessage,
    CommandResponse,
    HeartbeatMessage,
    Message,
)
from aiperf.zmq.sub_client import ZMQSubClient
from aiperf.zmq.zmq_defaults import TOPIC_DELIMITER, TOPIC_END, TOPIC_END_ENCODED


class TestZMQSubClientInitialization:
    """Test ZMQSubClient initialization."""

    def test_init_creates_sub_socket(self, mock_zmq_context):
        """Test that initialization creates a SUB socket."""
        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)

        assert client.socket_type == zmq.SocketType.SUB
        assert client._subscribers == {}


class TestZMQSubClientSubscription:
    """Test subscription methods."""

    @pytest.mark.asyncio
    async def test_subscribe_registers_callback(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that subscribe registers a callback."""
        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)
        await client.initialize()

        async def callback(msg: Message) -> None:
            pass

        await client.subscribe(MessageType.HEARTBEAT, callback)

        assert MessageType.HEARTBEAT in client._subscribers
        assert callback in client._subscribers[MessageType.HEARTBEAT]

        # Verify socket subscription was set
        mock_zmq_socket.setsockopt.assert_any_call(
            zmq.SUBSCRIBE, f"{MessageType.HEARTBEAT}".encode() + TOPIC_END_ENCODED
        )

    @pytest.mark.asyncio
    async def test_subscribe_multiple_callbacks_same_topic(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test subscribing multiple callbacks to the same topic."""
        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)
        await client.initialize()

        async def callback1(msg: Message) -> None:
            pass

        async def callback2(msg: Message) -> None:
            pass

        await client.subscribe(MessageType.HEARTBEAT, callback1)
        await client.subscribe(MessageType.HEARTBEAT, callback2)

        # Both callbacks should be registered
        assert len(client._subscribers[MessageType.HEARTBEAT]) == 2

        # Socket subscription should only be set once
        subscribe_calls = [
            call
            for call in mock_zmq_socket.setsockopt.call_args_list
            if call[0][0] == zmq.SUBSCRIBE
        ]
        # Should be called once for this topic
        topic_calls = [
            call
            for call in subscribe_calls
            if f"{MessageType.HEARTBEAT}".encode() in call[0][1]
        ]
        assert len(topic_calls) == 1

    @pytest.mark.asyncio
    async def test_subscribe_all_registers_multiple_callbacks(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that subscribe_all registers multiple callbacks."""
        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)
        await client.initialize()

        async def callback1(msg: Message) -> None:
            pass

        async def callback2(msg: Message) -> None:
            pass

        await client.subscribe_all(
            {
                MessageType.HEARTBEAT: callback1,
                MessageType.ERROR: callback2,
            }
        )

        assert MessageType.HEARTBEAT in client._subscribers
        assert MessageType.ERROR in client._subscribers

    @pytest.mark.asyncio
    async def test_subscribe_all_with_list_of_callbacks(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that subscribe_all handles lists of callbacks."""
        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)
        await client.initialize()

        async def callback1(msg: Message) -> None:
            pass

        async def callback2(msg: Message) -> None:
            pass

        await client.subscribe_all({MessageType.HEARTBEAT: [callback1, callback2]})

        assert len(client._subscribers[MessageType.HEARTBEAT]) == 2

    @pytest.mark.asyncio
    async def test_subscribe_raises_communication_error_on_failure(
        self, mock_zmq_context
    ):
        """Test that subscribe raises CommunicationError on failure."""

        def setsockopt_side_effect(option, value):
            """Only raise error for SUBSCRIBE option."""
            if option == zmq.SUBSCRIBE:
                raise zmq.ZMQError("Subscription failed")

        mock_socket = Mock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock(side_effect=setsockopt_side_effect)
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)
        await client.initialize()

        async def callback(msg: Message) -> None:
            pass

        with pytest.raises(CommunicationError, match="Failed to subscribe"):
            await client.subscribe(MessageType.HEARTBEAT, callback)


class TestZMQSubClientMessageHandling:
    """Test message handling logic."""

    @pytest.mark.asyncio
    async def test_handle_message_parses_topic_and_message(
        self, mock_zmq_context, sample_message, create_callback_tracker
    ):
        """Test that _handle_message correctly parses topic and message."""
        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)

        callback, event, received_messages = create_callback_tracker()
        client._subscribers[sample_message.message_type] = [callback]

        topic_bytes = f"{sample_message.message_type}{TOPIC_END}".encode()
        message_bytes = sample_message.model_dump_json().encode()

        await client._handle_message(topic_bytes, message_bytes)
        await asyncio.wait_for(event.wait(), timeout=1.0)

        assert len(received_messages) == 1
        assert received_messages[0].message_type == sample_message.message_type

    @pytest.mark.asyncio
    async def test_handle_message_with_targeted_topic(self, mock_zmq_context):
        """Test handling messages with targeted topics (service_id/type)."""
        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)

        callback_called = asyncio.Event()

        async def callback(msg: Message) -> None:
            callback_called.set()

        # Register callback for the FULL targeted topic (not just base message type)
        targeted_topic = f"{MessageType.COMMAND}{TOPIC_DELIMITER}service-123"
        client._subscribers[targeted_topic] = [callback]

        # Message with targeted topic
        message = CommandMessage(
            service_id="test-service",
            command=CommandType.SHUTDOWN,
        )
        topic_bytes = f"{targeted_topic}{TOPIC_END}".encode()
        message_bytes = message.model_dump_json().encode()

        await client._handle_message(topic_bytes, message_bytes)
        await asyncio.wait_for(callback_called.wait(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_handle_message_calls_all_callbacks(
        self, mock_zmq_context, create_callback_tracker
    ):
        """Test that _handle_message calls all registered callbacks."""
        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)

        callback1, event1, msgs1 = create_callback_tracker()
        callback2, event2, msgs2 = create_callback_tracker()

        message = HeartbeatMessage(
            service_id="test-service",
            state=LifecycleState.RUNNING,
            service_type="test",
        )
        client._subscribers[MessageType.HEARTBEAT] = [callback1, callback2]

        topic_bytes = f"{MessageType.HEARTBEAT}{TOPIC_END}".encode()
        message_bytes = message.model_dump_json().encode()

        await client._handle_message(topic_bytes, message_bytes)

        await asyncio.wait_for(event1.wait(), timeout=1.0)
        await asyncio.wait_for(event2.wait(), timeout=1.0)

        assert len(msgs1) == 1 and len(msgs2) == 1

    @pytest.mark.asyncio
    async def test_handle_message_deserializes_command_message(
        self, mock_zmq_context, create_callback_tracker
    ):
        """Test that COMMAND messages are deserialized as CommandMessage."""
        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)

        callback, event, received_messages = create_callback_tracker()
        client._subscribers[MessageType.COMMAND] = [callback]

        message = CommandMessage(
            service_id="test-service",
            command=CommandType.SHUTDOWN,
        )
        topic_bytes = f"{MessageType.COMMAND}{TOPIC_END}".encode()
        message_bytes = message.model_dump_json().encode()

        await client._handle_message(topic_bytes, message_bytes)
        await event.wait()

        assert isinstance(received_messages[0], CommandMessage)

    @pytest.mark.asyncio
    async def test_handle_message_deserializes_command_response(
        self, mock_zmq_context, create_callback_tracker
    ):
        """Test that COMMAND_RESPONSE messages are deserialized as CommandResponse."""
        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)

        callback, event, received_messages = create_callback_tracker()
        client._subscribers[MessageType.COMMAND_RESPONSE] = [callback]

        message = CommandResponse(
            service_id="test-service",
            command=CommandType.SHUTDOWN,
            command_id="cmd-123",
            status=CommandResponseStatus.SUCCESS,
        )
        topic_bytes = f"{MessageType.COMMAND_RESPONSE}{TOPIC_END}".encode()
        message_bytes = message.model_dump_json().encode()

        await client._handle_message(topic_bytes, message_bytes)
        await event.wait()

        assert isinstance(received_messages[0], CommandResponse)


class TestZMQSubClientBackgroundTask:
    """Test background task for receiving messages."""

    @pytest.mark.asyncio
    async def test_background_task_receives_and_handles_message(
        self, mock_zmq_context, sample_message, wait_for_background_task
    ):
        """Test that background task receives and handles messages."""
        topic_bytes = f"{sample_message.message_type}{TOPIC_END}".encode()
        message_bytes = sample_message.model_dump_json().encode()

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.recv_multipart = AsyncMock(
            side_effect=[[topic_bytes, message_bytes], zmq.Again()]
        )
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)

        callback_called = asyncio.Event()

        async def callback(msg: Message) -> None:
            callback_called.set()

        await client.initialize()
        await client.subscribe(sample_message.message_type, callback)
        await client.start()
        await wait_for_background_task()

        # Wait for callback to be called
        await asyncio.wait_for(callback_called.wait(), timeout=1.0)

        await client.stop()

    @pytest.mark.asyncio
    async def test_background_task_handles_zmq_again(
        self, mock_zmq_context, wait_for_background_task
    ):
        """Test that background task handles zmq.Again gracefully."""
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.recv_multipart = AsyncMock(side_effect=zmq.Again())
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)

        await client.initialize()
        await client.start()
        await wait_for_background_task()

        # Should not raise
        await client.stop()

    @pytest.mark.asyncio
    async def test_background_task_handles_exceptions(
        self, mock_zmq_context, wait_for_background_task
    ):
        """Test that background task handles exceptions gracefully."""
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.recv_multipart = AsyncMock(
            side_effect=[RuntimeError("Test error"), zmq.Again()]
        )
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)

        await client.initialize()
        await client.start()
        await wait_for_background_task()

        # Should not crash
        await client.stop()

    @pytest.mark.asyncio
    async def test_background_task_stops_on_cancellation(
        self, mock_zmq_context, wait_for_background_task
    ):
        """Test that background task stops properly on cancellation."""
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.recv_multipart = AsyncMock(side_effect=asyncio.CancelledError())
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)

        await client.initialize()
        await client.start()
        await wait_for_background_task()

        # Should complete without hanging
        await client.stop()

    @pytest.mark.asyncio
    async def test_sub_receiver_stops_on_context_terminated(self, mock_zmq_context):
        """Test that _sub_receiver breaks on ContextTerminated."""
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.recv_multipart = AsyncMock(side_effect=zmq.ContextTerminated())
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        client = ZMQSubClient(address="tcp://127.0.0.1:5555", bind=False)
        await client.initialize()

        # Call _sub_receiver directly to exercise the except clause
        await client._sub_receiver()

        # Should have tried to recv and hit ContextTerminated
        mock_socket.recv_multipart.assert_called()
