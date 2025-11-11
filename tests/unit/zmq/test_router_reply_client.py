# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for router_reply_client.py - ZMQRouterReplyClient class.
"""

import asyncio

import pytest
import zmq
import zmq.asyncio

from aiperf.common.enums import MessageType
from aiperf.common.messages import ErrorMessage, Message
from aiperf.zmq.router_reply_client import ZMQRouterReplyClient


class TestZMQRouterReplyClientInitialization:
    """Test ZMQRouterReplyClient initialization."""

    def test_init_creates_router_socket(self, mock_zmq_context):
        """Test that initialization creates a ROUTER socket."""
        client = ZMQRouterReplyClient(address="tcp://127.0.0.1:5555", bind=True)

        assert client.socket_type == zmq.SocketType.ROUTER
        assert client._request_handlers == {}
        assert client._response_futures == {}


class TestZMQRouterReplyClientHandlerRegistration:
    """Test request handler registration."""

    def test_register_request_handler(self, mock_zmq_context):
        """Test registering a request handler."""
        # No need to patch - autouse fixture handles this
        client = ZMQRouterReplyClient(address="tcp://127.0.0.1:5555", bind=True)

        async def handler(msg: Message) -> Message:
            return msg

        client.register_request_handler(
            service_id="test-service",
            message_type=MessageType.HEARTBEAT,
            handler=handler,
        )

        assert MessageType.HEARTBEAT in client._request_handlers
        assert client._request_handlers[MessageType.HEARTBEAT][0] == "test-service"
        assert client._request_handlers[MessageType.HEARTBEAT][1] == handler

    def test_register_duplicate_handler_raises_error(self, mock_zmq_context):
        """Test that registering duplicate handler raises ValueError."""
        client = ZMQRouterReplyClient(address="tcp://127.0.0.1:5555", bind=True)

        async def handler(msg: Message) -> Message:
            return msg

        client.register_request_handler(
            service_id="test-service",
            message_type=MessageType.HEARTBEAT,
            handler=handler,
        )

        with pytest.raises(ValueError, match="Handler already registered"):
            client.register_request_handler(
                service_id="test-service2",
                message_type=MessageType.HEARTBEAT,
                handler=handler,
            )

    @pytest.mark.asyncio
    async def test_clear_request_handlers_on_stop(self, mock_zmq_context):
        """Test that request handlers are cleared on stop."""
        # No need to patch - autouse fixture handles this
        client = ZMQRouterReplyClient(address="tcp://127.0.0.1:5555", bind=True)

        async def handler(msg: Message) -> Message:
            return msg

        client.register_request_handler(
            service_id="test-service",
            message_type=MessageType.HEARTBEAT,
            handler=handler,
        )
        await client.initialize()

        assert len(client._request_handlers) > 0

        await client.stop()

        assert len(client._request_handlers) == 0


class TestZMQRouterReplyClientRequestHandling:
    """Test request handling logic."""

    @pytest.mark.asyncio
    async def test_handle_request_calls_handler(
        self, router_test_helper, sample_message
    ):
        """Test that _handle_request calls the registered handler."""
        response = Message(
            message_type=MessageType.HEARTBEAT, request_id=sample_message.request_id
        )

        async def handler(msg: Message) -> Message:
            return response

        async with router_test_helper.create_client() as client:
            client.register_request_handler(
                service_id="test-service",
                message_type=sample_message.message_type,
                handler=handler,
            )

            # Create response future
            client._response_futures[sample_message.request_id] = asyncio.Future()

            # Handle request
            await client._handle_request(sample_message.request_id, sample_message)

            # Check that future was set
            result = await client._response_futures[sample_message.request_id]
            assert result == response

    @pytest.mark.asyncio
    async def test_handle_request_returns_error_on_exception(
        self, router_test_helper, sample_message
    ):
        """Test that _handle_request returns ErrorMessage on handler exception."""

        async def failing_handler(msg: Message) -> Message:
            raise ValueError("Handler failed")

        async with router_test_helper.create_client() as client:
            client.register_request_handler(
                service_id="test-service",
                message_type=sample_message.message_type,
                handler=failing_handler,
            )

            # Create response future
            client._response_futures[sample_message.request_id] = asyncio.Future()

            # Handle request
            await client._handle_request(sample_message.request_id, sample_message)

            # Check that future was set with error message
            result = await client._response_futures[sample_message.request_id]
            assert isinstance(result, ErrorMessage)
            assert result.request_id == sample_message.request_id

    @pytest.mark.asyncio
    async def test_wait_for_response_sends_message(
        self, router_test_helper, sample_message
    ):
        """Test that _wait_for_response sends the response back."""
        response = Message(
            message_type=MessageType.HEARTBEAT, request_id=sample_message.request_id
        )

        mock_socket = router_test_helper.setup_mock_socket()

        async with router_test_helper.create_client() as client:
            # Create and set response future
            client._response_futures[sample_message.request_id] = asyncio.Future()
            client._response_futures[sample_message.request_id].set_result(response)

            routing_envelope = (b"client_id",)

            # Wait for and send response
            await client._wait_for_response(sample_message.request_id, routing_envelope)

            # Verify send was called
            mock_socket.send_multipart.assert_called_once()
            sent_data = mock_socket.send_multipart.call_args[0][0]
            assert sent_data[0] == b"client_id"

    @pytest.mark.asyncio
    async def test_wait_for_response_handles_none_response(
        self, router_test_helper, sample_message
    ):
        """Test that _wait_for_response handles None response gracefully."""
        mock_socket = router_test_helper.setup_mock_socket()

        async with router_test_helper.create_client() as client:
            # Create and set response future with None
            client._response_futures[sample_message.request_id] = asyncio.Future()
            client._response_futures[sample_message.request_id].set_result(None)

            routing_envelope = (b"client_id",)

            # Wait for and send response
            await client._wait_for_response(sample_message.request_id, routing_envelope)

            # Verify error message was sent
            mock_socket.send_multipart.assert_called_once()


class TestZMQRouterReplyClientBackgroundTask:
    """Test router reply client background task."""

    @pytest.mark.asyncio
    async def test_background_task_receives_and_processes_request(
        self, router_test_helper, sample_message
    ):
        """Test that background task receives and processes requests."""
        request_data = [b"client_id", sample_message.model_dump_json().encode()]

        response = Message(
            message_type=MessageType.HEARTBEAT, request_id=sample_message.request_id
        )

        async def handler(msg: Message) -> Message:
            return response

        mock_socket = router_test_helper.setup_mock_socket(
            recv_multipart_side_effect=[request_data, zmq.Again()]
        )

        async with router_test_helper.create_client(auto_start=True) as client:
            client.register_request_handler(
                service_id="test-service",
                message_type=sample_message.message_type,
                handler=handler,
            )

            await asyncio.sleep(0.2)

            mock_socket.send_multipart.assert_called()

    @pytest.mark.asyncio
    async def test_background_task_handles_zmq_again(self, router_test_helper):
        """Test that background task handles zmq.Again gracefully."""
        async with router_test_helper.create_client(
            auto_start=True, recv_multipart_side_effect=zmq.Again()
        ):
            await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_background_task_ignores_request_without_id(self, router_test_helper):
        """Test that background task ignores requests without request_id."""
        message_no_id = Message(message_type=MessageType.HEARTBEAT)
        message_no_id.request_id = None
        request_data = [b"client_id", message_no_id.model_dump_json().encode()]

        mock_socket = router_test_helper.setup_mock_socket(
            recv_multipart_side_effect=[request_data, zmq.Again()]
        )

        async with router_test_helper.create_client(auto_start=True):
            await asyncio.sleep(0.1)

            mock_socket.send_multipart.assert_not_called()
