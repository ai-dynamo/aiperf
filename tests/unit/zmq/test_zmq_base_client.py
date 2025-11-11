# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for zmq_base_client.py - BaseZMQClient class.
"""

import asyncio
from unittest.mock import Mock

import pytest
import zmq
import zmq.asyncio

from aiperf.common.enums import LifecycleState
from aiperf.common.exceptions import NotInitializedError
from aiperf.zmq.zmq_base_client import BaseZMQClient


class TestBaseZMQClientInitialization:
    """Test BaseZMQClient initialization."""

    @pytest.mark.parametrize(
        "socket_type,address,bind",
        [
            (zmq.SocketType.PUB, "tcp://127.0.0.1:5555", True),
            (zmq.SocketType.SUB, "tcp://127.0.0.1:5556", False),
            (zmq.SocketType.PUSH, "ipc:///tmp/test.ipc", True),
            (zmq.SocketType.PULL, "ipc:///tmp/test.ipc", False),
            (zmq.SocketType.DEALER, "tcp://127.0.0.1:5557", True),
            (zmq.SocketType.ROUTER, "tcp://127.0.0.1:5558", False),
        ],
    )  # fmt: skip
    def test_init_creates_socket_with_correct_params(
        self, socket_type, address, bind, mock_zmq_context
    ):
        """Test that __init__ creates socket with correct parameters."""
        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(socket_type=socket_type, address=address, bind=bind)

        assert client.socket_type == socket_type
        assert client.address == address
        assert client.bind == bind
        assert client.socket is not None
        mock_zmq_context.socket.assert_called_once_with(socket_type)

    def test_init_with_custom_client_id(self, mock_zmq_context):
        """Test initialization with custom client ID."""
        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(
            socket_type=zmq.SocketType.PUB,
            address="tcp://127.0.0.1:5555",
            bind=True,
            client_id="custom-client-id",
        )

        assert client.client_id == "custom-client-id"

    def test_init_generates_client_id_if_not_provided(self, mock_zmq_context):
        """Test that client_id is auto-generated if not provided."""
        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(
            socket_type=zmq.SocketType.PUB,
            address="tcp://127.0.0.1:5555",
            bind=True,
        )

        assert client.client_id is not None
        assert "pub_client_" in client.client_id

    def test_socket_type_name_property(self, mock_zmq_context):
        """Test socket_type_name property returns correct name."""
        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(
            socket_type=zmq.SocketType.PUB,
            address="tcp://127.0.0.1:5555",
            bind=True,
        )

        assert client.socket_type_name == "PUB"


class TestBaseZMQClientLifecycle:
    """Test BaseZMQClient lifecycle methods."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bind", [True, False])
    async def test_initialize_binds_or_connects_socket(
        self, bind, mock_zmq_socket, mock_zmq_context, assert_socket_configured
    ):
        """Test that initialize() binds or connects socket based on bind parameter."""
        address = "tcp://127.0.0.1:5555"

        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(
            socket_type=zmq.SocketType.PUB, address=address, bind=bind
        )

        await client.initialize()

        if bind:
            mock_zmq_socket.bind.assert_called_once_with(address)
            assert not mock_zmq_socket.connect.called
        else:
            mock_zmq_socket.connect.assert_called_once_with(address)
            assert not mock_zmq_socket.bind.called

        # Verify socket options were set
        assert_socket_configured(mock_zmq_socket)
        assert client.state == LifecycleState.INITIALIZED

    @pytest.mark.asyncio
    async def test_initialize_sets_socket_options(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that initialize() sets all required socket options."""
        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(
            socket_type=zmq.SocketType.PUB,
            address="tcp://127.0.0.1:5555",
            bind=True,
        )

        await client.initialize()

        # Check all socket options were set
        calls = mock_zmq_socket.setsockopt.call_args_list
        option_types = [call[0][0] for call in calls]

        expected_options = [
            zmq.RCVTIMEO,
            zmq.SNDTIMEO,
            zmq.SNDHWM,
            zmq.RCVHWM,
            zmq.TCP_KEEPALIVE,
            zmq.TCP_KEEPALIVE_IDLE,
            zmq.TCP_KEEPALIVE_INTVL,
            zmq.TCP_KEEPALIVE_CNT,
            zmq.IMMEDIATE,
            zmq.LINGER,
        ]

        for option in expected_options:
            assert option in option_types

    @pytest.mark.asyncio
    async def test_initialize_with_custom_socket_ops(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that custom socket options are applied."""
        custom_ops = {zmq.SUBSCRIBE: b"test_topic"}

        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(
            socket_type=zmq.SocketType.SUB,
            address="tcp://127.0.0.1:5555",
            bind=False,
            socket_ops=custom_ops,
        )

        await client.initialize()

        # Verify custom socket option was set
        mock_zmq_socket.setsockopt.assert_any_call(zmq.SUBSCRIBE, b"test_topic")

    @pytest.mark.asyncio
    async def test_initialize_raises_on_bind_error(self, mock_zmq_context):
        """Test that initialize() raises CancelledError on bind failure (wrapped by lifecycle)."""
        mock_socket = Mock(spec=zmq.asyncio.Socket)
        mock_socket.bind = Mock(side_effect=zmq.ZMQError("Bind failed"))
        mock_socket.setsockopt = Mock()
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(
            socket_type=zmq.SocketType.PUB,
            address="tcp://127.0.0.1:5555",
            bind=True,
        )

        with pytest.raises(asyncio.CancelledError, match="Failed for BaseZMQClient"):
            await client.initialize()

    @pytest.mark.asyncio
    async def test_stop_closes_socket(self, mock_zmq_socket, mock_zmq_context):
        """Test that stop() closes the socket."""
        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(
            socket_type=zmq.SocketType.PUB,
            address="tcp://127.0.0.1:5555",
            bind=True,
        )

        await client.initialize()
        await client.stop()

        mock_zmq_socket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_handles_context_terminated(self, mock_zmq_context):
        """Test that stop() handles ContextTerminated gracefully."""
        mock_socket = Mock(spec=zmq.asyncio.Socket)
        mock_socket.close = Mock(side_effect=zmq.ContextTerminated())
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(
            socket_type=zmq.SocketType.PUB,
            address="tcp://127.0.0.1:5555",
            bind=True,
        )

        await client.initialize()
        # Should not raise an exception
        await client.stop()


class TestBaseZMQClientOperations:
    """Test BaseZMQClient operations."""

    @pytest.mark.asyncio
    async def test_check_initialized_raises_when_not_initialized(
        self, mock_zmq_context
    ):
        """Test that _check_initialized raises when socket is not initialized."""
        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(
            socket_type=zmq.SocketType.PUB,
            address="tcp://127.0.0.1:5555",
            bind=True,
        )
        # Don't initialize
        client.socket = None

        with pytest.raises(NotInitializedError, match="Socket not initialized"):
            await client._check_initialized()

    @pytest.mark.asyncio
    async def test_check_initialized_raises_when_stopped(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that _check_initialized raises CancelledError when stopped."""
        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(
            socket_type=zmq.SocketType.PUB,
            address="tcp://127.0.0.1:5555",
            bind=True,
        )

        await client.initialize()
        await client.stop()

        with pytest.raises(asyncio.CancelledError, match="Socket was stopped"):
            await client._check_initialized()

    @pytest.mark.asyncio
    async def test_check_initialized_succeeds_when_initialized(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that _check_initialized passes when socket is initialized."""
        # No need to patch - autouse fixture handles this
        client = BaseZMQClient(
            socket_type=zmq.SocketType.PUB,
            address="tcp://127.0.0.1:5555",
            bind=True,
        )

        await client.initialize()
        # Should not raise
        await client._check_initialized()
