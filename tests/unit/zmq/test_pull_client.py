# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for pull_client.py - ZMQPullClient class.
"""

import asyncio
from unittest.mock import patch

import pytest
import zmq

from aiperf.common.enums import LifecycleState, MessageType
from aiperf.common.environment import Environment
from aiperf.common.messages import HeartbeatMessage, Message
from aiperf.zmq.pull_client import ZMQPullClient


class TestZMQPullClientInitialization:
    """Test ZMQPullClient initialization."""

    def test_init_creates_pull_socket(self, mock_zmq_context):
        """Test that initialization creates a PULL socket."""
        client = ZMQPullClient(address="tcp://127.0.0.1:5555", bind=False)

        assert client.socket_type == zmq.SocketType.PULL
        assert client._pull_callbacks == {}

    def test_init_with_default_concurrency(self, mock_zmq_context):
        """Test initialization with default max_pull_concurrency."""
        with (
            patch("zmq.asyncio.Context.instance", return_value=mock_zmq_context),
            patch.object(Environment.ZMQ, "PULL_MAX_CONCURRENCY", 10),
        ):
            client = ZMQPullClient(address="tcp://127.0.0.1:5555", bind=False)

            assert client._max_inflight == 10

    def test_init_with_custom_concurrency(self, mock_zmq_context):
        """Test initialization with custom max_pull_concurrency."""
        client = ZMQPullClient(
            address="tcp://127.0.0.1:5555", bind=False, max_pull_concurrency=5
        )

        assert client._max_inflight == 5


class TestZMQPullClientCallbackRegistration:
    """Test callback registration."""

    def test_register_pull_callback(self, mock_zmq_context):
        """Test registering a pull callback."""
        client = ZMQPullClient(address="tcp://127.0.0.1:5555", bind=False)

        async def callback(msg: Message) -> None:
            pass

        client.register_pull_callback(MessageType.HEARTBEAT, callback)

        assert MessageType.HEARTBEAT in client._pull_callbacks
        assert client._pull_callbacks[MessageType.HEARTBEAT] == callback

    def test_register_duplicate_callback_raises_error(self, mock_zmq_context):
        """Test that registering duplicate callback raises ValueError."""
        client = ZMQPullClient(address="tcp://127.0.0.1:5555", bind=False)

        async def callback(msg: Message) -> None:
            pass

        client.register_pull_callback(MessageType.HEARTBEAT, callback)

        with pytest.raises(ValueError, match="Callback already registered"):
            client.register_pull_callback(MessageType.HEARTBEAT, callback)

    def test_register_multiple_different_callbacks(self, mock_zmq_context):
        """Test registering callbacks for different message types."""
        client = ZMQPullClient(address="tcp://127.0.0.1:5555", bind=False)

        async def callback1(msg: Message) -> None:
            pass

        async def callback2(msg: Message) -> None:
            pass

        client.register_pull_callback(MessageType.HEARTBEAT, callback1)
        client.register_pull_callback(MessageType.ERROR, callback2)

        assert len(client._pull_callbacks) == 2


class TestZMQPullClientBackgroundTask:
    """Test pull client background task for receiving messages."""

    @pytest.mark.asyncio
    async def test_background_task_receives_and_processes_message(
        self,
        pull_test_helper,
        sample_message,
        create_callback_tracker,
        wait_for_background_task,
    ):
        """Test that background task receives and processes messages."""
        callback, event, received_messages = create_callback_tracker()

        async with pull_test_helper.create_client(
            auto_start=False,
            recv_side_effect=[sample_message.to_json_bytes()],
        ) as client:
            # Register callback BEFORE starting
            client.register_pull_callback(sample_message.message_type, callback)

            # Now start the client
            await client.start()
            await (
                wait_for_background_task()
            )  # Yield to event loop to let background task run

            # Wait for callback to be called
            await asyncio.wait_for(event.wait(), timeout=1.0)

            assert len(received_messages) == 1

    @pytest.mark.asyncio
    async def test_background_task_handles_zmq_again(
        self, pull_test_helper, wait_for_background_task
    ):
        """Test that background task handles zmq.Again gracefully."""
        async with pull_test_helper.create_client(
            auto_start=True,
            recv_side_effect=zmq.Again(),
        ):
            # Should not raise
            await wait_for_background_task()

    @pytest.mark.asyncio
    async def test_background_task_idle_on_no_data(
        self, pull_test_helper, wait_for_background_task
    ):
        """FD drain stays idle (no in-flight) when recv yields no data."""
        async with pull_test_helper.create_client(
            auto_start=True,
            max_pull_concurrency=5,
            recv_side_effect=zmq.Again(),
        ) as client:
            await wait_for_background_task()

            # Nothing was delivered, so no credits are in flight.
            assert client._inflight == 0

    @pytest.mark.asyncio
    async def test_background_task_handles_exception_gracefully(
        self, pull_test_helper, wait_for_background_task
    ):
        """Test that background task handles exceptions gracefully."""
        async with pull_test_helper.create_client(
            auto_start=True,
            recv_side_effect=[RuntimeError("Test error")],
        ):
            # Should not crash, just continue
            await wait_for_background_task()

    @pytest.mark.asyncio
    async def test_background_task_warns_on_message_without_callback(
        self, pull_test_helper, wait_for_background_task
    ):
        """Test that background task logs warning for messages without callbacks."""
        message = Message(message_type=MessageType.HEARTBEAT)

        async with pull_test_helper.create_client(
            auto_start=True,
            recv_side_effect=[message.to_json_bytes()],
        ):
            # Don't register any callbacks
            # Should not crash, just continue
            await wait_for_background_task()


class TestZMQPullClientConcurrency:
    """Test FD-drain flow control via the _inflight counter."""

    @pytest.mark.asyncio
    async def test_inflight_processes_multiple_messages(
        self, mock_zmq_socket, mock_zmq_context, fd_enqueue
    ):
        """The FD drain delivers all queued messages (bounded by _max_inflight)."""
        messages = [
            HeartbeatMessage(
                service_id="test-service",
                state=LifecycleState.RUNNING,
                service_type="test",
                request_id=f"req-{i}",
            )
            for i in range(10)
        ]
        fd_enqueue(mock_zmq_socket, messages=[m.to_json_bytes() for m in messages])

        client = ZMQPullClient(
            address="tcp://127.0.0.1:5555", bind=False, max_pull_concurrency=3
        )

        processing = []
        done_event = asyncio.Event()

        async def slow_callback(msg: Message) -> None:
            processing.append(msg.request_id)
            if len(processing) >= 3:
                done_event.set()
            await asyncio.sleep(0.1)

        client.register_pull_callback(MessageType.HEARTBEAT, slow_callback)

        await client.initialize()
        await client.start()

        await asyncio.wait_for(done_event.wait(), timeout=1.0)

        await client.stop()

        # max_pull_concurrency=3 bounds the drain, and done_event fires once the
        # third callback starts, so at least the inflight cap's worth is delivered.
        assert len(processing) >= 3

    @pytest.mark.asyncio
    async def test_inflight_released_after_processing(
        self, mock_zmq_socket, mock_zmq_context, sample_message, fd_enqueue
    ):
        """_inflight returns to 0 once a delivered message finishes processing."""
        fd_enqueue(mock_zmq_socket, messages=[sample_message.to_json_bytes()])

        client = ZMQPullClient(
            address="tcp://127.0.0.1:5555", bind=False, max_pull_concurrency=5
        )

        done_event = asyncio.Event()

        async def callback(msg: Message) -> None:
            await asyncio.sleep(0.01)
            done_event.set()

        client.register_pull_callback(sample_message.message_type, callback)

        await client.initialize()
        await client.start()

        await asyncio.wait_for(done_event.wait(), timeout=1.0)
        # Let the _process_message_fd finally-block run (_inflight -= 1).
        await asyncio.sleep(0)

        assert client._inflight == 0
        await client.stop()
