# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for zmq_proxy_base.py and zmq_proxy_sockets.py - ZMQ proxy classes.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import zmq

from aiperf.common.config import ZMQTCPConfig, ZMQTCPProxyConfig
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ZMQProxyType
from aiperf.zmq.zmq_proxy_base import ProxyEndType, ProxySocketClient
from aiperf.zmq.zmq_proxy_sockets import (
    ZMQDealerRouterProxy,
    ZMQPushPullProxy,
    ZMQXPubXSubProxy,
)


class TestProxySocketClient:
    """Test ProxySocketClient class."""

    @pytest.mark.asyncio
    async def test_init_creates_client_with_correct_params(self, mock_zmq_context):
        """Test that ProxySocketClient initializes correctly."""
        client = ProxySocketClient(
            socket_type=zmq.SocketType.ROUTER,
            address="tcp://127.0.0.1:5555",
            end_type=ProxyEndType.Frontend,
            proxy_uuid="test-uuid",
        )

        assert client.socket_type == zmq.SocketType.ROUTER
        assert client.address == "tcp://127.0.0.1:5555"
        assert "frontend" in client.client_id.lower()
        assert "router" in client.client_id.lower()
        assert "test-uuid" in client.client_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "socket_type,end_type",
        [
            (zmq.SocketType.ROUTER, ProxyEndType.Frontend),
            (zmq.SocketType.DEALER, ProxyEndType.Backend),
            (zmq.SocketType.PUB, ProxyEndType.Capture),
            (zmq.SocketType.REP, ProxyEndType.Control),
        ],
    )  # fmt: skip
    async def test_init_with_various_socket_types(
        self, socket_type, end_type, mock_zmq_context
    ):
        """Test initialization with various socket and end types."""
        client = ProxySocketClient(
            socket_type=socket_type,
            address="tcp://127.0.0.1:5555",
            end_type=end_type,
        )

        assert client.socket_type == socket_type
        assert client.bind is True  # Proxy sockets always bind

    @pytest.mark.asyncio
    async def test_init_generates_uuid_if_not_provided(self, mock_zmq_context):
        """Test that ProxySocketClient generates UUID if not provided."""
        client = ProxySocketClient(
            socket_type=zmq.SocketType.ROUTER,
            address="tcp://127.0.0.1:5555",
            end_type=ProxyEndType.Frontend,
        )

        # Should have a UUID in the client_id
        assert len(client.client_id.split("_")) >= 3


class TestZMQXPubXSubProxy:
    """Test ZMQXPubXSubProxy class."""

    @pytest.mark.asyncio
    async def test_from_config_creates_proxy(self, mock_zmq_context):
        """Test that from_config creates a proxy instance."""
        config = ZMQTCPConfig()

        proxy = ZMQXPubXSubProxy.from_config(config.event_bus_proxy_config)

        assert proxy is not None
        assert proxy.config.frontend_address is not None
        assert proxy.config.backend_address is not None

    def test_from_config_returns_none_if_config_is_none(self):
        """Test that from_config returns None if config is None."""
        proxy = ZMQXPubXSubProxy.from_config(None)

        assert proxy is None

    @pytest.mark.asyncio
    async def test_init_creates_frontend_and_backend_sockets(self, mock_zmq_context):
        """Test that initialization creates frontend and backend sockets."""
        config = ZMQTCPConfig()

        proxy = ZMQXPubXSubProxy.from_config(config.event_bus_proxy_config)

        assert proxy.frontend_socket is not None
        assert proxy.backend_socket is not None
        assert proxy.frontend_socket.socket_type == zmq.SocketType.XSUB
        assert proxy.backend_socket.socket_type == zmq.SocketType.XPUB

    @pytest.mark.asyncio
    async def test_initialize_binds_sockets(self, mock_zmq_socket, mock_zmq_context):
        """Test that initialize binds frontend and backend sockets."""
        config = ZMQTCPConfig()

        proxy = ZMQXPubXSubProxy.from_config(config.event_bus_proxy_config)

        await proxy.initialize()

        # Both sockets should be bound
        assert mock_zmq_socket.bind.call_count >= 2

    @pytest.mark.asyncio
    async def test_xpub_socket_sets_verbose_option(self, mock_zmq_context):
        """Test that XPUB socket sets XPUB_VERBOSE option."""
        config = ZMQTCPConfig()

        mock_socket = Mock()
        mock_socket.bind = Mock()
        mock_socket.setsockopt = Mock()
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        proxy = ZMQXPubXSubProxy.from_config(config.event_bus_proxy_config)

        await proxy.initialize()

        # Verify XPUB_VERBOSE was set (for subscription forwarding)
        calls = mock_socket.setsockopt.call_args_list
        option_names = [call[0][0] for call in calls if len(call[0]) > 0]

        # XPUB_VERBOSE should be set on the backend (XPUB) socket
        assert any(opt == zmq.XPUB_VERBOSE for opt in option_names)


class TestZMQDealerRouterProxy:
    """Test ZMQDealerRouterProxy class."""

    @pytest.mark.asyncio
    async def test_from_config_creates_proxy(self, mock_zmq_context):
        """Test that from_config creates a proxy instance."""
        proxy_config = ZMQTCPProxyConfig(frontend_port=5661, backend_port=5662)

        proxy = ZMQDealerRouterProxy.from_config(proxy_config)

        assert proxy is not None
        assert proxy.frontend_socket.socket_type == zmq.SocketType.ROUTER
        assert proxy.backend_socket.socket_type == zmq.SocketType.DEALER

    @pytest.mark.asyncio
    async def test_initialize_binds_sockets(self, mock_zmq_socket, mock_zmq_context):
        """Test that initialize binds frontend and backend sockets."""
        proxy_config = ZMQTCPProxyConfig(frontend_port=5661, backend_port=5662)

        proxy = ZMQDealerRouterProxy.from_config(proxy_config)

        await proxy.initialize()

        # Both sockets should be bound
        assert mock_zmq_socket.bind.call_count >= 2


class TestZMQPushPullProxy:
    """Test ZMQPushPullProxy class."""

    @pytest.mark.asyncio
    async def test_from_config_creates_proxy(self, mock_zmq_context):
        """Test that from_config creates a proxy instance."""
        config = ZMQTCPConfig()

        proxy = ZMQPushPullProxy.from_config(config.raw_inference_proxy_config)

        assert proxy is not None
        assert proxy.frontend_socket.socket_type == zmq.SocketType.PULL
        assert proxy.backend_socket.socket_type == zmq.SocketType.PUSH

    @pytest.mark.asyncio
    async def test_initialize_binds_sockets(self, mock_zmq_socket, mock_zmq_context):
        """Test that initialize binds frontend and backend sockets."""
        config = ZMQTCPConfig()

        proxy = ZMQPushPullProxy.from_config(config.raw_inference_proxy_config)

        await proxy.initialize()

        # Both sockets should be bound
        assert mock_zmq_socket.bind.call_count >= 2


class TestProxyLifecycle:
    """Test proxy lifecycle methods."""

    @pytest.mark.asyncio
    async def test_proxy_initialize_start_stop(self, mock_zmq_socket, mock_zmq_context):
        """Test proxy lifecycle: initialize, start, stop."""
        config = ZMQTCPConfig()

        with (
            patch("zmq.asyncio.Context.instance", return_value=mock_zmq_context),
            patch("zmq.proxy_steerable"),
        ):
            proxy = ZMQXPubXSubProxy.from_config(config.event_bus_proxy_config)

            # Initialize
            await proxy.initialize()

            # Start
            await proxy.start()
            # Yield to event loop to let background task run
            await asyncio.sleep(0)

            # Stop
            await proxy.stop()

            # Sockets should be closed
            assert mock_zmq_socket.close.call_count >= 2

    @pytest.mark.asyncio
    async def test_proxy_with_capture_socket(self, mock_zmq_context):
        """Test proxy with capture socket enabled."""
        config = ZMQTCPConfig()
        base_config = config.event_bus_proxy_config
        # Create a new config with capture address
        proxy_config = ZMQTCPProxyConfig(
            host=base_config.host,
            frontend_port=base_config.frontend_port,
            backend_port=base_config.backend_port,
            capture_port=9999,
        )

        proxy = ZMQXPubXSubProxy(zmq_proxy_config=proxy_config)

        assert proxy.capture_client is not None
        assert proxy.capture_client.socket_type == zmq.SocketType.PUB

    @pytest.mark.asyncio
    async def test_proxy_with_control_socket(self, mock_zmq_context):
        """Test proxy with control socket enabled."""
        config = ZMQTCPConfig()
        base_config = config.event_bus_proxy_config
        # Create a new config with control address
        proxy_config = ZMQTCPProxyConfig(
            host=base_config.host,
            frontend_port=base_config.frontend_port,
            backend_port=base_config.backend_port,
            control_port=9998,
        )

        proxy = ZMQXPubXSubProxy(zmq_proxy_config=proxy_config)

        assert proxy.control_client is not None
        assert proxy.control_client.socket_type == zmq.SocketType.REP

    @pytest.mark.asyncio
    async def test_proxy_stop_closes_all_sockets(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that stop closes all sockets (frontend, backend, capture, control)."""
        config = ZMQTCPConfig()
        base_config = config.event_bus_proxy_config
        # Create a new config with both capture and control addresses
        proxy_config = ZMQTCPProxyConfig(
            host=base_config.host,
            frontend_port=base_config.frontend_port,
            backend_port=base_config.backend_port,
            capture_port=9999,
            control_port=9998,
        )

        proxy = ZMQXPubXSubProxy(zmq_proxy_config=proxy_config)

        await proxy.initialize()
        await proxy.stop()

        # Should close frontend, backend, capture, and control sockets
        assert mock_zmq_socket.close.call_count >= 4


class TestProxyEdgeCases:
    """Test edge cases for proxies."""

    @pytest.mark.asyncio
    async def test_proxy_handles_initialization_error(self, mock_zmq_context):
        """Test that proxy handles initialization errors gracefully."""
        config = ZMQTCPConfig()

        mock_socket = Mock()
        mock_socket.bind = Mock(side_effect=zmq.ZMQError("Bind failed"))
        mock_socket.setsockopt = Mock()
        mock_zmq_context.socket = Mock(return_value=mock_socket)

        proxy = ZMQXPubXSubProxy.from_config(config.event_bus_proxy_config)

        # The lifecycle pattern converts all errors to CancelledError
        with pytest.raises(asyncio.CancelledError):
            await proxy.initialize()

    @pytest.mark.asyncio
    async def test_proxy_has_unique_id(self, mock_zmq_context):
        """Test that each proxy has a unique ID."""
        config = ZMQTCPConfig()

        proxy1 = ZMQXPubXSubProxy.from_config(config.event_bus_proxy_config)
        proxy2 = ZMQXPubXSubProxy.from_config(config.event_bus_proxy_config)

        # Each proxy should have a unique ID
        assert proxy1.proxy_id != proxy2.proxy_id
        assert proxy1.proxy_uuid != proxy2.proxy_uuid

    @pytest.mark.asyncio
    async def test_proxy_custom_uuid(self, mock_zmq_context):
        """Test creating proxy with custom UUID."""
        config = ZMQTCPConfig()

        proxy = ZMQXPubXSubProxy(
            zmq_proxy_config=config.event_bus_proxy_config,
            socket_ops=None,
        )
        # Constructor sets a UUID
        assert proxy.proxy_uuid is not None

    @pytest.mark.asyncio
    async def test_all_proxy_types_are_registered(self):
        """Test that all proxy types are registered in the plugin system."""
        # Verify all expected proxy types are registered
        registered_names = [e.name for e in plugins.iter_entries(PluginType.ZMQ_PROXY)]
        assert ZMQProxyType.XPUB_XSUB in registered_names
        assert ZMQProxyType.DEALER_ROUTER in registered_names
        assert ZMQProxyType.PUSH_PULL in registered_names


class TestProxyMonitor:
    """Test proxy _monitor_messages background task."""

    @pytest.mark.asyncio
    async def test_monitor_returns_early_without_capture(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that _monitor_messages returns early when no capture socket."""
        config = ZMQTCPConfig()

        with patch("zmq.proxy_steerable"):
            proxy = ZMQXPubXSubProxy.from_config(config.event_bus_proxy_config)
            await proxy.initialize()
            await proxy.start()
            await asyncio.sleep(0)  # Let background tasks run
            await proxy.stop()

        # No capture socket, monitor should return early (line 228)
        assert proxy.capture_client is None

    @pytest.mark.asyncio
    async def test_monitor_connects_and_handles_cancellation(
        self, mock_zmq_socket, mock_zmq_context
    ):
        """Test that _monitor_messages connects to capture and handles cancellation."""
        config = ZMQTCPConfig()
        base_config = config.event_bus_proxy_config
        proxy_config = ZMQTCPProxyConfig(
            host=base_config.host,
            frontend_port=base_config.frontend_port,
            backend_port=base_config.backend_port,
            capture_port=9999,
        )

        proxy = ZMQXPubXSubProxy(zmq_proxy_config=proxy_config)
        assert proxy.capture_client is not None

        # Set up a capture monitor socket that raises CancelledError on recv
        capture_socket = AsyncMock()
        capture_socket.recv_multipart = AsyncMock(side_effect=asyncio.CancelledError())
        capture_socket.close = Mock()
        capture_socket.connect = Mock()
        capture_socket.setsockopt = Mock()
        mock_zmq_context.socket = Mock(return_value=capture_socket)

        # Call the monitor method directly (bypassing background_task decorator)
        await proxy._monitor_messages()

        # Monitor should have connected and closed in the finally block
        capture_socket.connect.assert_called_once()
        capture_socket.close.assert_called_once()
