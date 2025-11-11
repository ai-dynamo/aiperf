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
from aiperf.common.messages import Message, TargetedServiceMessage
from aiperf.zmq.pub_client import ZMQPubClient
from aiperf.zmq.zmq_defaults import TOPIC_DELIMITER, TOPIC_END


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

        message = Message(message_type=MessageType.HEARTBEAT)

        await client.publish(message)

        sent_parts = mock_zmq_socket.send_multipart.call_args[0][0]
        topic = sent_parts[0].decode()

        assert topic == f"{MessageType.HEARTBEAT}{TOPIC_END}"

    @pytest.mark.asyncio
    async def test_publish_with_targeted_service_id(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that publishing with target_service_id includes it in topic."""
        client = ZMQPubClient(address="tcp://127.0.0.1:5555", bind=True)
        await client.initialize()

        message = TargetedServiceMessage(
            service_id="test-service",
            message_type=MessageType.COMMAND,
            target_service_id="service-123",
        )

        await client.publish(message)

        sent_parts = mock_zmq_socket.send_multipart.call_args[0][0]
        topic = sent_parts[0].decode()

        expected_topic = f"{MessageType.COMMAND}{TOPIC_DELIMITER}service-123{TOPIC_END}"
        assert topic == expected_topic

    @pytest.mark.asyncio
    async def test_publish_with_targeted_service_type(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that publishing with target_service_type includes it in topic."""
        client = ZMQPubClient(address="tcp://127.0.0.1:5555", bind=True)
        await client.initialize()

        message = TargetedServiceMessage(
            service_id="test-service",
            message_type=MessageType.COMMAND,
            target_service_type="WORKER",
        )

        await client.publish(message)

        sent_parts = mock_zmq_socket.send_multipart.call_args[0][0]
        topic = sent_parts[0].decode()

        expected_topic = f"{MessageType.COMMAND}{TOPIC_DELIMITER}WORKER{TOPIC_END}"
        assert topic == expected_topic

    @pytest.mark.asyncio
    async def test_publish_service_id_takes_precedence_over_type(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that target_service_id is used in topic when provided."""
        client = ZMQPubClient(address="tcp://127.0.0.1:5555", bind=True)
        await client.initialize()

        # Can only provide one of target_service_id or target_service_type
        message = TargetedServiceMessage(
            service_id="test-service",
            message_type=MessageType.COMMAND,
            target_service_id="service-123",
        )

        await client.publish(message)

        sent_parts = mock_zmq_socket.send_multipart.call_args[0][0]
        topic = sent_parts[0].decode()

        # Should use service_id in topic
        assert "service-123" in topic

    @pytest.mark.asyncio
    async def test_publish_handles_graceful_errors(
        self, pub_test_helper, graceful_error
    ):
        """Test that publish handles graceful errors (CancelledError, ContextTerminated)."""
        async with pub_test_helper.create_client(
            send_multipart_side_effect=graceful_error
        ) as client:
            message = Message(message_type=MessageType.HEARTBEAT)

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
            message = Message(message_type=MessageType.HEARTBEAT)

            with pytest.raises(CommunicationError, match="Failed to publish message"):
                await client.publish(message)


class TestZMQPubClientTopicDetermination:
    """Test topic determination logic."""

    def test_determine_topic_for_basic_message(self, mock_zmq_context):
        """Test topic determination for basic message."""
        client = ZMQPubClient(address="tcp://127.0.0.1:5555", bind=True)

        message = Message(message_type=MessageType.HEARTBEAT)
        topic = client._determine_topic(message)

        assert topic == f"{MessageType.HEARTBEAT}{TOPIC_END}"

    @pytest.mark.parametrize(
        "message_type",
        [
            MessageType.HEARTBEAT,
            MessageType.ERROR,
            MessageType.COMMAND,
            MessageType.COMMAND_RESPONSE,
        ],
    )  # fmt: skip
    def test_determine_topic_for_various_message_types(
        self, message_type, mock_zmq_context
    ):
        """Test topic determination for various message types."""
        client = ZMQPubClient(address="tcp://127.0.0.1:5555", bind=True)

        message = Message(message_type=message_type)
        topic = client._determine_topic(message)

        assert topic == f"{message_type}{TOPIC_END}"

    def test_determine_topic_with_service_id(self, mock_zmq_context):
        """Test topic determination with service ID."""
        client = ZMQPubClient(address="tcp://127.0.0.1:5555", bind=True)

        message = TargetedServiceMessage(
            service_id="test-service",
            message_type=MessageType.COMMAND,
            target_service_id="svc-456",
        )
        topic = client._determine_topic(message)

        assert topic == f"{MessageType.COMMAND}{TOPIC_DELIMITER}svc-456{TOPIC_END}"

    def test_determine_topic_with_service_type(self, mock_zmq_context):
        """Test topic determination with service type."""
        client = ZMQPubClient(address="tcp://127.0.0.1:5555", bind=True)

        message = TargetedServiceMessage(
            service_id="test-service",
            message_type=MessageType.COMMAND,
            target_service_type="PROCESSOR",
        )
        topic = client._determine_topic(message)

        assert topic == f"{MessageType.COMMAND}{TOPIC_DELIMITER}PROCESSOR{TOPIC_END}"


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
            Message(message_type=MessageType.HEARTBEAT, request_id=f"req-{i}")
            for i in range(5)
        ]

        for msg in messages:
            await client.publish(msg)

        assert mock_zmq_socket.send_multipart.call_count == 5
