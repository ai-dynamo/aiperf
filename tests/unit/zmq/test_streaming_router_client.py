# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for streaming_router_client.py - ZMQStreamingRouterClient class.
"""

import asyncio

import msgspec.msgpack
import pytest
import zmq

from aiperf.common.enums import LifecycleState
from aiperf.common.exceptions import NotInitializedError
from aiperf.credit.messages import (
    WorkerReady,
    WorkerToRouterMessage,
)
from aiperf.credit.structs import (
    Credit,
)
from aiperf.zmq.streaming_router_client import ZMQStreamingRouterClient


class TestZMQStreamingRouterClientInitialization:
    """Test ZMQStreamingRouterClient initialization."""

    def test_init_creates_router_socket(self, mock_zmq_context):
        """Test that initialization creates a ROUTER socket."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)

        assert client.socket_type == zmq.SocketType.ROUTER
        assert client._receiver_handler is None

    @pytest.mark.parametrize(
        "address,bind",
        [
            ("tcp://*:5555", True),
            ("tcp://127.0.0.1:5556", False),
            ("ipc:///tmp/test.ipc", True),
            ("ipc:///tmp/test.ipc", False),
        ],
        ids=["tcp_bind", "tcp_connect", "ipc_bind", "ipc_connect"],
    )  # fmt: skip
    def test_init_with_various_addresses(self, address, bind, mock_zmq_context):
        """Test initialization with various address types."""
        client = ZMQStreamingRouterClient(address=address, bind=bind)

        assert client.address == address
        assert client.bind == bind

    def test_init_with_custom_socket_options(self, mock_zmq_context):
        """Test initialization with custom socket options."""
        custom_ops = {zmq.ROUTER_MANDATORY: 1}
        client = ZMQStreamingRouterClient(
            address="tcp://*:5555",
            bind=True,
            socket_ops=custom_ops,
        )

        assert client.socket_ops == custom_ops


class TestZMQStreamingRouterClientRegisterReceiver:
    """Test ZMQStreamingRouterClient.register_receiver method."""

    @pytest.mark.asyncio
    async def test_register_receiver_succeeds(self, mock_zmq_context):
        """Test that register_receiver successfully registers a handler."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)

        async def handler(identity: str, message: WorkerToRouterMessage) -> None:
            pass

        client.register_receiver(handler)
        assert client._receiver_handler == handler

    @pytest.mark.asyncio
    async def test_register_receiver_raises_when_already_registered(
        self, mock_zmq_context
    ):
        """Test that register_receiver raises ValueError if already registered."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)

        async def handler1(identity: str, message: WorkerToRouterMessage) -> None:
            pass

        async def handler2(identity: str, message: WorkerToRouterMessage) -> None:
            pass

        client.register_receiver(handler1)

        with pytest.raises(ValueError, match="already registered"):
            client.register_receiver(handler2)


class TestZMQStreamingRouterClientSendTo:
    """Test ZMQStreamingRouterClient.send_to method."""

    @pytest.mark.asyncio
    async def test_send_to_sends_credit_with_routing(
        self, streaming_router_test_helper, sample_credit
    ):
        """Test that send_to sends credit with routing envelope."""
        async with streaming_router_test_helper.create_client() as client:
            identity = "worker-1"
            mock_socket = client.socket

            await client.send_to(identity, sample_credit)

            # FD path frames the message as two sync sends: identity + payload.
            assert mock_socket._sync_send.call_count == 2
            calls = mock_socket._sync_send.call_args_list
            assert calls[0][0][0] == identity.encode()
            decoded = msgspec.msgpack.decode(calls[1][0][0], type=Credit)
            assert decoded.id == sample_credit.id

    @pytest.mark.asyncio
    async def test_send_to_multiple_identities(
        self, streaming_router_test_helper, sample_credit, multiple_identities
    ):
        """Test sending to different worker identities."""
        async with streaming_router_test_helper.create_client() as client:
            mock_socket = client.socket

            for identity in multiple_identities:
                await client.send_to(identity, sample_credit)

            # Two sync sends (identity + payload) per message.
            assert mock_socket._sync_send.call_count == len(multiple_identities) * 2

    @pytest.mark.asyncio
    async def test_send_to_raises_when_not_initialized(
        self, mock_zmq_context, sample_credit
    ):
        """Test that send_to raises NotInitializedError when not initialized."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)
        client.socket = None

        with pytest.raises(NotInitializedError, match="Socket not initialized"):
            await client.send_to("worker-1", sample_credit)

    @pytest.mark.asyncio
    async def test_send_to_handles_send_failure(
        self, streaming_router_test_helper, sample_credit
    ):
        """Test that send_to handles send failures."""
        async with streaming_router_test_helper.create_client(
            send_multipart_side_effect=Exception("Send failed")
        ) as client:
            with pytest.raises(Exception, match="Send failed"):
                await client.send_to("worker-1", sample_credit)

    @pytest.mark.asyncio
    async def test_send_to_with_special_identity(
        self, streaming_router_test_helper, sample_credit, special_identity
    ):
        """Test identity encoding with special characters."""
        async with streaming_router_test_helper.create_client() as client:
            mock_socket = client.socket

            await client.send_to(special_identity, sample_credit)

            assert mock_socket._sync_send.call_args_list[0][0][0] == (
                special_identity.encode()
            )


class TestZMQStreamingRouterClientReceiver:
    """Test ZMQStreamingRouterClient receiver background task."""

    @pytest.mark.asyncio
    async def test_receiver_task_starts_on_start(self, streaming_router_test_helper):
        """Test that receiver task starts when client starts."""
        async with streaming_router_test_helper.create_client(
            auto_start=True
        ) as client:
            assert client.state == LifecycleState.RUNNING

    @pytest.mark.asyncio
    async def test_receiver_calls_handler_on_worker_ready(
        self, streaming_router_test_helper, sample_worker_ready, create_callback_tracker
    ):
        """Test that receiver calls handler when WorkerReady arrives."""
        identity = "worker-1"
        callback, event, received = create_callback_tracker()

        async def test_handler(
            recv_identity: str, message: WorkerToRouterMessage
        ) -> None:
            await callback((recv_identity, message))

        # FD path drains the [identity, payload] frames off the raw socket.
        streaming_router_test_helper.setup_mock_socket(
            recv_multipart_side_effect=[
                [identity.encode(), msgspec.msgpack.encode(sample_worker_ready)]
            ]
        )

        async with streaming_router_test_helper.create_client() as client:
            # Register handler BEFORE starting to avoid race condition
            client.register_receiver(test_handler)
            await client.start()

            await asyncio.wait_for(event.wait(), timeout=1.0)
            assert len(received) == 1
            recv_identity, recv_message = received[0]
            assert recv_identity == identity
            assert isinstance(recv_message, WorkerReady)
            assert recv_message.worker_id == sample_worker_ready.worker_id

    @pytest.mark.asyncio
    async def test_receiver_warns_when_no_handler_registered(
        self,
        streaming_router_test_helper,
        sample_worker_ready,
        wait_for_background_task,
    ):
        """Test that receiver logs warning when no handler is registered."""
        streaming_router_test_helper.setup_mock_socket(
            recv_multipart_side_effect=[
                [b"worker-1", msgspec.msgpack.encode(sample_worker_ready)]
            ]
        )

        async with streaming_router_test_helper.create_client(auto_start=True):
            # Don't register handler
            await wait_for_background_task(iterations=5)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception,iterations",
        [
            (zmq.Again(), 3),
            (RuntimeError("Test error"), 3),
        ],
        ids=["zmq_again", "generic_error"],
    )  # fmt: skip
    async def test_receiver_handles_exceptions(
        self,
        streaming_router_test_helper,
        exception,
        iterations,
    ):
        """Test that the FD-driver surfaces a recv error without crashing.

        A raising recv is reported via the driver's on_error callback (generic
        error) or yields no data (zmq.Again); the client stays RUNNING.
        """
        streaming_router_test_helper.setup_mock_socket(
            recv_multipart_side_effect=[exception]
        )

        async with streaming_router_test_helper.create_client(
            auto_start=True
        ) as client:
            await asyncio.sleep(0)
            assert client.state == LifecycleState.RUNNING

    @pytest.mark.asyncio
    async def test_receiver_stops_on_cancelled_error(
        self, streaming_router_test_helper, wait_for_background_task
    ):
        """Test that receiver stops gracefully on CancelledError."""
        streaming_router_test_helper.setup_mock_socket(
            recv_multipart_side_effect=asyncio.CancelledError()
        )

        async with streaming_router_test_helper.create_client(
            auto_start=True
        ) as client:
            # Wait for the background task to run and exit due to CancelledError
            await wait_for_background_task()
            # The receiver task should exit gracefully without raising an unhandled exception
            # Client remains in RUNNING state until explicitly stopped
            assert client.state == LifecycleState.RUNNING

    @pytest.mark.asyncio
    async def test_receiver_with_empty_routing_envelope(
        self, streaming_router_test_helper, sample_worker_ready, create_callback_tracker
    ):
        """Test receiver handling of message with empty routing envelope."""
        callback, event, received = create_callback_tracker()

        async def test_handler(identity: str, message: WorkerToRouterMessage) -> None:
            await callback((identity, message))

        streaming_router_test_helper.setup_mock_socket(
            recv_multipart_side_effect=[
                [b"", msgspec.msgpack.encode(sample_worker_ready)]
            ]
        )

        async with streaming_router_test_helper.create_client() as client:
            # Register handler BEFORE starting to avoid race condition
            client.register_receiver(test_handler)
            await client.start()

            await asyncio.wait_for(event.wait(), timeout=1.0)
            assert len(received) == 1
            recv_identity, _ = received[0]
            assert recv_identity == ""  # Empty identity


class TestZMQStreamingRouterClientLifecycle:
    """Test ZMQStreamingRouterClient lifecycle management."""

    @pytest.mark.asyncio
    async def test_clear_receiver_on_stop(self, streaming_router_test_helper):
        """Test that receiver handler is cleared on stop."""
        async with streaming_router_test_helper.create_client() as client:

            async def handler(identity: str, message: WorkerToRouterMessage) -> None:
                pass

            client.register_receiver(handler)
            assert client._receiver_handler == handler

        # Client stopped after context exit
        assert client._receiver_handler is None

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, streaming_router_test_helper):
        """Test full client lifecycle: initialize -> start -> stop."""
        client = ZMQStreamingRouterClient(address="tcp://*:5555", bind=True)

        async def handler(identity: str, message: WorkerToRouterMessage) -> None:
            pass

        client.register_receiver(handler)

        # Initialize
        await client.initialize()
        assert client.state == LifecycleState.INITIALIZED

        # Start
        await client.start()
        assert client.state == LifecycleState.RUNNING

        # Stop
        await client.stop()
        assert client.state == LifecycleState.STOPPED
        assert client._receiver_handler is None

    @pytest.mark.asyncio
    async def test_send_to_after_stop_raises(
        self, streaming_router_test_helper, sample_credit
    ):
        """Test that send_to raises after client is stopped."""
        async with streaming_router_test_helper.create_client() as client:
            pass  # Client stopped after context exit

        with pytest.raises(asyncio.CancelledError, match="Socket was stopped"):
            await client.send_to("worker-1", sample_credit)


class TestZMQStreamingRouterClientEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_sends(
        self, streaming_router_test_helper, sample_credit, multiple_identities
    ):
        """Test multiple concurrent sends to different workers."""
        async with streaming_router_test_helper.create_client() as client:
            mock_socket = client.socket

            await asyncio.gather(
                *[
                    client.send_to(identity, sample_credit)
                    for identity in multiple_identities
                ]
            )

            assert mock_socket._sync_send.call_count == len(multiple_identities) * 2
