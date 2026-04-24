# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for pub_client.py - ZMQPubClient class.
"""

import pytest
import zmq
import zmq.asyncio

from aiperf.common.enums import MessageType
from aiperf.common.exceptions import CommunicationError
from aiperf.common.messages import ConnectionProbeMessage
from aiperf.zmq.pub_client import ZMQPubClient
from aiperf.zmq.zmq_defaults import TOPIC_END


class TestZMQPubClientInitialization:
    """Test ZMQPubClient initialization."""

    def test_init_creates_pub_socket(self, mock_zmq_context):
        """Test that initialization creates a PUB socket."""
        client = ZMQPubClient(address="tcp://127.0.0.1:5555", bind=True)

        assert client.socket_type == zmq.SocketType.PUB


class TestZMQPubClientPublish:
    """Test ZMQPubClient.publish method."""

    @pytest.mark.asyncio
    async def test_publish_sends_message_with_topic(
        self, mock_zmq_socket, mock_zmq_context, sample_message
    ):
        """Test that publish sends message with topic."""
        client = ZMQPubClient(address="tcp://127.0.0.1:5555", bind=True)
        await client.initialize()

        await client.publish(sample_message)

        mock_zmq_socket.send_multipart.assert_called_once()
        sent_parts = mock_zmq_socket.send_multipart.call_args[0][0]

        # First part is topic
        topic = sent_parts[0].decode()
        assert topic == f"{sample_message.message_type}{TOPIC_END}"

        # Second part is message
        message_json = sent_parts[1].decode()
        assert sample_message.request_id in message_json

    @pytest.mark.asyncio
    async def test_publish_with_basic_message_uses_simple_topic(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that publishing a basic message uses simple topic."""
        client = ZMQPubClient(address="tcp://127.0.0.1:5555", bind=True)
        await client.initialize()

        message = ConnectionProbeMessage(service_id="test")

        await client.publish(message)

        sent_parts = mock_zmq_socket.send_multipart.call_args[0][0]
        topic = sent_parts[0].decode()

        assert topic == f"{MessageType.CONNECTION_PROBE}{TOPIC_END}"

    @pytest.mark.asyncio
    async def test_publish_handles_graceful_errors(
        self, pub_test_helper, graceful_error
    ):
        """Test that publish handles graceful errors (CancelledError, ContextTerminated)."""
        async with pub_test_helper.create_client(
            send_multipart_side_effect=graceful_error
        ) as client:
            message = ConnectionProbeMessage(service_id="test")

            # Should not raise, just return
            await client.publish(message)

    @pytest.mark.asyncio
    async def test_publish_raises_communication_error_on_exception(
        self, pub_test_helper, non_graceful_error
    ):
        """Test that publish raises CommunicationError on other exceptions."""
        async with pub_test_helper.create_client(
            send_multipart_side_effect=non_graceful_error
        ) as client:
            message = ConnectionProbeMessage(service_id="test")

            with pytest.raises(CommunicationError, match="Failed to publish message"):
                await client.publish(message)


class TestZMQPubClientTopicDetermination:
    """Test topic determination logic."""

    @pytest.mark.parametrize(
        "message_type",
        [
            MessageType.HEARTBEAT,
            MessageType.ERROR,
            MessageType.CONNECTION_PROBE,
            MessageType.STATUS,
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_topic_uses_message_type_with_end_sentinel(
        self, message_type, mock_zmq_socket, mock_zmq_context
    ):
        """Test that topic is always message_type + TOPIC_END."""
        client = ZMQPubClient(address="tcp://127.0.0.1:5555", bind=True)
        await client.initialize()

        # Need a concrete message for each tag; use a helper
        from aiperf.common.enums import LifecycleState
        from aiperf.common.messages import (
            ConnectionProbeMessage,
            ErrorMessage,
            HeartbeatMessage,
            StatusMessage,
        )
        from aiperf.common.models import ErrorDetails

        if message_type == MessageType.HEARTBEAT:
            message = HeartbeatMessage(
                service_id="s", service_type="worker", state=LifecycleState.RUNNING
            )
        elif message_type == MessageType.STATUS:
            message = StatusMessage(
                service_id="s", service_type="worker", state=LifecycleState.RUNNING
            )
        elif message_type == MessageType.CONNECTION_PROBE:
            message = ConnectionProbeMessage(service_id="s")
        else:
            message = ErrorMessage(error=ErrorDetails(type="x", message="m"))
        await client.publish(message)

        sent_parts = mock_zmq_socket.send_multipart.call_args[0][0]
        topic = sent_parts[0].decode()
        assert topic == f"{message_type}{TOPIC_END}"


class TestZMQPubClientEdgeCases:
    """Test edge cases for ZMQPubClient."""

    @pytest.mark.asyncio
    async def test_multiple_sequential_publishes(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test multiple sequential publish operations."""
        client = ZMQPubClient(address="tcp://127.0.0.1:5555", bind=True)
        await client.initialize()

        messages = [
            ConnectionProbeMessage(service_id="test", request_id=f"req-{i}")
            for i in range(5)
        ]

        for msg in messages:
            await client.publish(msg)

        assert mock_zmq_socket.send_multipart.call_count == 5
