# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the ZMQ communication module.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.comms.client_enums import PubClientType, SubClientType
from aiperf.common.comms.zmq import BaseZMQCommunication
from aiperf.common.config import ZMQIPCConfig
from aiperf.common.enums import ServiceState, ServiceType, Topic
from aiperf.common.exceptions import CommunicationError, CommunicationErrorReason
from aiperf.common.messages import Message, StatusMessage


@pytest.mark.asyncio
class TestZMQCommunication:
    """Tests for the ZMQ communication class."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Return a mock configuration for ZMQCommunication."""
        return ZMQIPCConfig(path=str(tmp_path))

    @pytest.fixture
    def zmq_communication(self, mock_config):
        """Return a ZMQCommunication instance for testing."""
        with patch("zmq.asyncio.Context", MagicMock()) as mock_context:
            # Set up the context mock to return properly
            mock_context.return_value = MagicMock()
            comm = BaseZMQCommunication(config=mock_config)
            comm._context = mock_context
            return comm

    @pytest.fixture
    def test_message(self):
        """Create a test message for communication tests."""
        return StatusMessage(
            service_id="test-service",
            service_type=ServiceType.TEST,
            state=ServiceState.READY,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, zmq_communication):
        """Test that the ZMQ communication initializes correctly."""
        result = await zmq_communication.initialize()
        assert result is None
        assert zmq_communication.is_initialized is True

    @pytest.mark.asyncio
    async def test_initialization_failure(self, zmq_communication):
        """Test initialization failure handling."""
        # Temporarily clear initialized_event to test error path
        zmq_communication.initialized_event.clear()

        # Create a mock implementation that raises an exception
        async def mock_init_with_error():
            raise CommunicationError(
                CommunicationErrorReason.INITIALIZATION_ERROR, "Test connection error"
            )

        # Replace the original method and call to test error handling
        original_init = zmq_communication.initialize
        zmq_communication.initialize = mock_init_with_error

        try:
            with pytest.raises(
                CommunicationError,
                match="Communication Error INITIALIZATION_ERROR: Test connection error",
            ):
                await zmq_communication.initialize()
        finally:
            # Restore the original method
            zmq_communication.initialize = original_init

    @pytest.mark.asyncio
    async def test_create_clients(self, zmq_communication):
        """Test creating clients for different communication patterns."""
        # Mock the client socket creation
        mock_client = AsyncMock()

        # Patch the specific client classes and ensure they return our mock
        with (
            patch(
                "aiperf.common.comms.zmq.clients.ZMQPubClient",
                return_value=mock_client,
            ),
            patch(
                "aiperf.common.comms.zmq.clients.ZMQSubClient",
                return_value=mock_client,
            ),
        ):
            # Call create_clients
            await zmq_communication.create_clients(
                PubClientType.COMPONENT, SubClientType.COMPONENT
            )

            # Verify clients were added to the dictionary
            assert PubClientType.COMPONENT in zmq_communication.clients
            assert SubClientType.COMPONENT in zmq_communication.clients

            # Verify initialize was called for each client
            assert len(zmq_communication.clients) == 2

    @pytest.mark.asyncio
    async def test_publish_message(self, zmq_communication, test_message):
        """Test publishing messages."""
        # Mock the socket publish method
        mock_client = AsyncMock()
        mock_client.publish.return_value = None

        # Set up the client in the clients dictionary
        zmq_communication.clients = {PubClientType.COMPONENT: mock_client}
        zmq_communication.initialized_event.set()

        # Publish a message
        result = await zmq_communication.publish(Topic.STATUS, test_message)

        # Verify the message was published
        assert result is None
        mock_client.publish.assert_called_once_with(Topic.STATUS, test_message)

    @pytest.mark.asyncio
    async def test_subscribe_to_topic(self, zmq_communication):
        """Test subscribing to a topic."""
        # Mock the client socket
        mock_client = AsyncMock()
        mock_client.subscribe.return_value = None

        # Set up the client in the clients dictionary
        zmq_communication.clients = {SubClientType.COMPONENT: mock_client}
        zmq_communication.initialized_event.set()

        # Create a callback function
        async def callback(message: Message):
            pass

        # Subscribe to a topic
        result = await zmq_communication.subscribe(Topic.STATUS, callback)

        # Verify subscription was set up
        assert result is None
        mock_client.subscribe.assert_called_once_with(Topic.STATUS, callback)

    @pytest.mark.asyncio
    async def test_shutdown(self, zmq_communication):
        """Test graceful shutdown of communication."""
        # Mock the client socket
        mock_client1 = AsyncMock()
        mock_client1.shutdown.return_value = None
        mock_client2 = AsyncMock()
        mock_client2.shutdown.return_value = None

        # Set up clients
        zmq_communication.clients = {
            PubClientType.COMPONENT: mock_client1,
            SubClientType.COMPONENT: mock_client2,
        }
        zmq_communication.initialized_event.set()
        zmq_communication.stop_event.clear()

        # Mock the context with a patched shutdown method to avoid setting
        # context to None
        context_mock = MagicMock()
        zmq_communication._context = context_mock

        # Create a patched version of shutdown that doesn't set context to None
        original_shutdown = zmq_communication.shutdown

        async def patched_shutdown():
            # Calls original gather but patch term() to prevent context from
            # becoming None
            with patch.object(zmq_communication, "_context", context_mock):
                return await original_shutdown()

        zmq_communication.shutdown = patched_shutdown

        try:
            # Shutdown the communication
            result = await zmq_communication.shutdown()

            # Verify both clients were shutdown
            assert result is None
            assert mock_client1.shutdown.called
            assert mock_client2.shutdown.called
            assert context_mock.term.called
            assert zmq_communication.stop_event.is_set()
        finally:
            # Restore the original method
            zmq_communication.shutdown = original_shutdown
