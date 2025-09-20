# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test suite for create_tcp_connector function.

This module tests the create_tcp_connector function, which is used to create a
TCP connector for use with aiohttp.ClientSession.
"""

import socket
from unittest.mock import Mock, patch

import pytest

from aiperf.clients.http import SocketDefaults
from aiperf.clients.http.aiohttp_client import create_tcp_connector
from aiperf.common import constants

################################################################################
# Test create_tcp_connector
################################################################################


class TestCreateTcpConnector:
    """Test suite for create_tcp_connector function."""

    def test_create_default_connector(self) -> None:
        """Test creating connector with default parameters."""
        with patch("aiohttp.TCPConnector") as mock_connector_class:
            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector

            result = create_tcp_connector()

            assert result == mock_connector
            mock_connector_class.assert_called_once()
            call_kwargs = mock_connector_class.call_args[1]

            # Verify default parameters
            assert call_kwargs["limit"] == constants.AIPERF_HTTP_CONNECTION_LIMIT
            assert call_kwargs["limit_per_host"] == 0
            assert call_kwargs["ttl_dns_cache"] == 300
            assert call_kwargs["use_dns_cache"] is True
            assert call_kwargs["enable_cleanup_closed"] is False
            assert call_kwargs["force_close"] is False
            assert call_kwargs["keepalive_timeout"] == 300
            assert call_kwargs["happy_eyeballs_delay"] is None
            assert call_kwargs["family"] == socket.AF_INET
            assert callable(call_kwargs["socket_factory"])

    def test_create_connector_with_custom_kwargs(self) -> None:
        """Test creating connector with custom parameters."""
        custom_kwargs = {
            "limit": 1000,
            "limit_per_host": 500,
            "ttl_dns_cache": 600,
            "keepalive_timeout": 120,
        }

        with patch("aiohttp.TCPConnector") as mock_connector_class:
            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector

            result = create_tcp_connector(**custom_kwargs)

            assert result == mock_connector
            call_kwargs = mock_connector_class.call_args[1]

            # Verify custom parameters override defaults
            assert call_kwargs["limit"] == 1000
            assert call_kwargs["limit_per_host"] == 500
            assert call_kwargs["ttl_dns_cache"] == 600
            assert call_kwargs["keepalive_timeout"] == 120

            # Verify other defaults are preserved
            assert call_kwargs["use_dns_cache"] is True
            assert call_kwargs["family"] == socket.AF_INET

    def test_socket_factory_configuration(self, socket_factory_setup) -> None:
        """Test that socket factory configures sockets correctly."""
        _, socket_factory = socket_factory_setup()

        with patch("socket.socket") as mock_socket_class:
            mock_socket = Mock()
            mock_socket_class.return_value = mock_socket

            addr_info = (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", 80),
            )
            result_socket = socket_factory(addr_info)

            assert result_socket == mock_socket

            mock_socket_class.assert_called_once_with(
                family=socket.AF_INET,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )

            expected_calls = [
                (socket.SOL_TCP, socket.TCP_NODELAY, 1),
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                (socket.SOL_SOCKET, socket.SO_RCVBUF, SocketDefaults.SO_RCVBUF),
                (socket.SOL_SOCKET, socket.SO_SNDBUF, SocketDefaults.SO_SNDBUF),
            ]

            for option_level, option_name, option_value in expected_calls:
                mock_socket.setsockopt.assert_any_call(
                    option_level, option_name, option_value
                )

    # Only run these tests on Linux
    if hasattr(socket, "TCP_KEEPIDLE"):

        @pytest.mark.parametrize(
            "has_attribute,attribute_name,tcp_option,expected_value",
            [
                (
                    True,
                    "TCP_KEEPIDLE",
                    socket.TCP_KEEPIDLE,
                    SocketDefaults.TCP_KEEPIDLE,
                ),
                (
                    True,
                    "TCP_KEEPINTVL",
                    socket.TCP_KEEPINTVL,
                    SocketDefaults.TCP_KEEPINTVL,
                ),
                (True, "TCP_KEEPCNT", socket.TCP_KEEPCNT, SocketDefaults.TCP_KEEPCNT),
                (
                    True,
                    "TCP_QUICKACK",
                    socket.TCP_QUICKACK,
                    SocketDefaults.TCP_QUICKACK,
                ),
                (
                    True,
                    "TCP_USER_TIMEOUT",
                    socket.TCP_USER_TIMEOUT,
                    SocketDefaults.TCP_USER_TIMEOUT,
                ),
                (False, "TCP_KEEPIDLE", socket.TCP_KEEPIDLE, None),
            ],
        )
        def test_socket_factory_linux_specific_options(
            self,
            has_attribute: bool,
            attribute_name: str,
            tcp_option: int,
            expected_value: int | None,
        ) -> None:
            """Test socket factory handles Linux-specific TCP options."""
            with patch("aiohttp.TCPConnector") as mock_connector_class:
                create_tcp_connector()

                socket_factory = mock_connector_class.call_args[1]["socket_factory"]

                with patch("socket.socket") as mock_socket_class:
                    mock_socket = Mock()
                    mock_socket_class.return_value = mock_socket

                    addr_info = (
                        socket.AF_INET,
                        socket.SOCK_STREAM,
                        socket.IPPROTO_TCP,
                        "",
                        ("127.0.0.1", 80),
                    )
                    socket_factory(addr_info)

                    if has_attribute and expected_value is not None:
                        # Mock the socket attribute to exist
                        with patch.object(
                            socket, attribute_name, expected_value, create=True
                        ):
                            mock_socket.setsockopt.assert_any_call(
                                socket.SOL_TCP, tcp_option, expected_value
                            )

    @pytest.mark.parametrize(
        "family,sock_type,proto",
        [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP),
            (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP),
        ],
    )
    def test_socket_factory_different_address_families(
        self, family: int, sock_type: int, proto: int
    ) -> None:
        """Test socket factory with different address families."""
        with patch("aiohttp.TCPConnector") as mock_connector_class:
            create_tcp_connector()

            socket_factory = mock_connector_class.call_args[1]["socket_factory"]

            with patch("socket.socket") as mock_socket_class:
                mock_socket = Mock()
                mock_socket_class.return_value = mock_socket

                addr_info = (family, sock_type, proto, "", ("127.0.0.1", 80))
                result = socket_factory(addr_info)

                assert result == mock_socket
                mock_socket_class.assert_called_once_with(
                    family=family, type=sock_type, proto=proto
                )

    def test_invalid_socket_options(self, socket_factory_setup) -> None:
        """Test socket factory with invalid options."""
        _, socket_factory = socket_factory_setup()

        with patch("socket.socket") as mock_socket_class:
            mock_socket = Mock()
            mock_socket.setsockopt.side_effect = OSError("Invalid socket option")
            mock_socket_class.return_value = mock_socket

            addr_info = (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", 80),
            )

            with pytest.raises(OSError, match="Invalid socket option"):
                socket_factory(addr_info)
