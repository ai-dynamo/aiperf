# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test suite for create_tcp_connector function.

This module tests the create_tcp_connector function, which is used to create a
TCP connector for use with aiohttp.ClientSession.
"""

import socket
import ssl
from unittest.mock import Mock, patch

import aiohttp
import pytest
import trustme
from aiohttp import web

from aiperf.common.enums import IPVersion
from aiperf.common.environment import Environment, _HTTPSettings
from aiperf.transports.aiohttp_client import create_tcp_connector
from aiperf.transports.http_defaults import AioHttpDefaults, _get_socket_family

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
            assert call_kwargs["limit"] == Environment.HTTP.CONNECTION_LIMIT
            assert call_kwargs["limit_per_host"] == 0
            assert call_kwargs["ttl_dns_cache"] == Environment.HTTP.TTL_DNS_CACHE
            assert call_kwargs["use_dns_cache"] is Environment.HTTP.USE_DNS_CACHE
            assert (
                call_kwargs["enable_cleanup_closed"]
                is Environment.HTTP.ENABLE_CLEANUP_CLOSED
            )
            assert call_kwargs["force_close"] is Environment.HTTP.FORCE_CLOSE
            assert (
                call_kwargs["keepalive_timeout"] == Environment.HTTP.KEEPALIVE_TIMEOUT
            )
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
            assert call_kwargs["use_dns_cache"] is Environment.HTTP.USE_DNS_CACHE
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
                (socket.SOL_SOCKET, socket.SO_RCVBUF, Environment.HTTP.SO_RCVBUF),
                (socket.SOL_SOCKET, socket.SO_SNDBUF, Environment.HTTP.SO_SNDBUF),
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
                    Environment.HTTP.TCP_KEEPIDLE,
                ),
                (
                    True,
                    "TCP_KEEPINTVL",
                    socket.TCP_KEEPINTVL,
                    Environment.HTTP.TCP_KEEPINTVL,
                ),
                (True, "TCP_KEEPCNT", socket.TCP_KEEPCNT, Environment.HTTP.TCP_KEEPCNT),
                (
                    True,
                    "TCP_QUICKACK",
                    socket.TCP_QUICKACK,
                    1,
                ),
                (
                    True,
                    "TCP_USER_TIMEOUT",
                    socket.TCP_USER_TIMEOUT,
                    Environment.HTTP.TCP_USER_TIMEOUT,
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

    @pytest.mark.parametrize("ssl_verify", [True, False])
    def test_ssl_verify_passed_to_connector(
        self, ssl_verify: bool, monkeypatch
    ) -> None:
        """Test that SSL_VERIFY setting is correctly passed to the connector."""
        monkeypatch.setattr(AioHttpDefaults, "SSL_VERIFY", ssl_verify)

        with patch("aiohttp.TCPConnector") as mock_connector_class:
            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector

            create_tcp_connector()

            call_kwargs = mock_connector_class.call_args[1]
            assert call_kwargs["ssl"] is ssl_verify

    def test_ssl_can_be_overridden_by_kwargs(self, monkeypatch) -> None:
        """Test that SSL setting can be overridden via kwargs."""
        # Even if SSL_VERIFY is True, passing ssl=False should override
        monkeypatch.setattr(AioHttpDefaults, "SSL_VERIFY", True)

        with patch("aiohttp.TCPConnector") as mock_connector_class:
            mock_connector = Mock()
            mock_connector_class.return_value = mock_connector

            create_tcp_connector(ssl=False)

            call_kwargs = mock_connector_class.call_args[1]
            assert call_kwargs["ssl"] is False


class TestGetSocketFamily:
    """Test suite for _get_socket_family function."""

    @pytest.mark.parametrize(
        "ip_version,expected_family",
        [
            (IPVersion.V4, socket.AF_INET),
            (IPVersion.V6, socket.AF_INET6),
            (IPVersion.AUTO, socket.AF_UNSPEC),
        ],
    )
    def test_get_socket_family_ip_version_maps_to_family(
        self, ip_version: str, expected_family: int, monkeypatch
    ) -> None:
        """Test _get_socket_family returns correct socket family for IP_VERSION."""
        monkeypatch.setattr(Environment.HTTP, "IP_VERSION", ip_version)
        assert _get_socket_family() == expected_family


class TestAioHttpDefaults:
    """Test suite for AioHttpDefaults class."""

    def test_ssl_verify_reads_from_environment(self) -> None:
        """Test that SSL_VERIFY is read from Environment.HTTP.SSL_VERIFY."""
        assert AioHttpDefaults.SSL_VERIFY is Environment.HTTP.SSL_VERIFY

    def test_ssl_verify_default_is_true(self, monkeypatch) -> None:
        """Test that SSL_VERIFY defaults to True when env var is not set."""
        monkeypatch.delenv("AIPERF_HTTP_SSL_VERIFY", raising=False)
        settings = _HTTPSettings()
        assert settings.SSL_VERIFY is True

    def test_ssl_verify_can_be_disabled_via_env(self, monkeypatch) -> None:
        """Test that SSL_VERIFY can be disabled via environment variable."""
        monkeypatch.setenv("AIPERF_HTTP_SSL_VERIFY", "false")
        settings = _HTTPSettings()
        assert settings.SSL_VERIFY is False

    def test_get_default_kwargs_includes_ssl(self, monkeypatch) -> None:
        """Test that get_default_kwargs includes ssl parameter."""
        monkeypatch.setattr(AioHttpDefaults, "SSL_VERIFY", True)
        kwargs = AioHttpDefaults.get_default_kwargs()
        assert "ssl" in kwargs
        assert kwargs["ssl"] is True

    @pytest.mark.parametrize("ssl_value", [True, False])
    def test_get_default_kwargs_ssl_reflects_setting(
        self, ssl_value: bool, monkeypatch
    ) -> None:
        """Test that ssl in kwargs reflects the SSL_VERIFY setting."""
        monkeypatch.setattr(AioHttpDefaults, "SSL_VERIFY", ssl_value)
        kwargs = AioHttpDefaults.get_default_kwargs()
        assert kwargs["ssl"] is ssl_value


class TestSSLVerificationWithServer:
    """Integration tests for SSL verification using a real HTTPS server."""

    @pytest.fixture(autouse=True)
    def reset_ssl_verify_to_default(self, monkeypatch):
        """Ensure SSL_VERIFY is set to default (True) regardless of local env settings."""
        monkeypatch.setattr(Environment.HTTP, "SSL_VERIFY", True)
        monkeypatch.setattr(AioHttpDefaults, "SSL_VERIFY", True)

    @pytest.fixture
    def ca(self) -> trustme.CA:
        """Create a certificate authority for testing."""
        return trustme.CA()

    @pytest.fixture
    def server_ssl_ctx(self, ca: trustme.CA) -> "trustme.SSLContext":
        """Create SSL context for the server."""
        server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ca.issue_cert("127.0.0.1", "localhost").configure_cert(server_ctx)
        return server_ctx

    @pytest.fixture
    async def https_server(
        self, server_ssl_ctx: "trustme.SSLContext"
    ) -> tuple[str, web.AppRunner]:
        """Start an HTTPS server with certificate from test CA."""

        async def handler(request: web.Request) -> web.Response:
            return web.Response(
                text='{"status": "ok"}', content_type="application/json"
            )

        app = web.Application()
        app.router.add_get("/health", handler)

        runner = web.AppRunner(app)
        await runner.setup()

        # Find an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        site = web.TCPSite(runner, "127.0.0.1", port, ssl_context=server_ssl_ctx)
        await site.start()

        url = f"https://127.0.0.1:{port}"

        yield url, runner

        await runner.cleanup()

    @pytest.mark.asyncio
    async def test_connection_succeeds_with_ssl_verify_disabled(
        self, https_server: tuple[str, web.AppRunner]
    ) -> None:
        """Test that connection succeeds when SSL verification is disabled."""
        url, _ = https_server

        connector = create_tcp_connector(ssl=False)
        try:
            async with (
                aiohttp.ClientSession(connector=connector) as session,
                session.get(f"{url}/health") as response,
            ):
                assert response.status == 200
                data = await response.json()
                assert data["status"] == "ok"
        finally:
            await connector.close()

    @pytest.mark.asyncio
    async def test_connection_fails_with_ssl_verify_enabled(
        self, https_server: tuple[str, web.AppRunner]
    ) -> None:
        """Test that connection fails when SSL verification is enabled with untrusted CA."""
        url, _ = https_server

        connector = create_tcp_connector(ssl=True)
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                with pytest.raises(aiohttp.ClientConnectorCertificateError):
                    async with session.get(f"{url}/health"):
                        pass
        finally:
            await connector.close()

    @pytest.mark.asyncio
    async def test_default_ssl_verification_rejects_untrusted_cert(
        self, https_server: tuple[str, web.AppRunner]
    ) -> None:
        """Test that default SSL verification rejects certificates from untrusted CA."""
        url, _ = https_server

        connector = create_tcp_connector()
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                with pytest.raises(aiohttp.ClientConnectorCertificateError):
                    async with session.get(f"{url}/health"):
                        pass
        finally:
            await connector.close()

    @pytest.mark.asyncio
    async def test_connection_succeeds_with_trusted_ca(
        self, ca: trustme.CA, https_server: tuple[str, web.AppRunner]
    ) -> None:
        """Test that connection succeeds when the CA is trusted."""
        url, _ = https_server

        # Create client SSL context that trusts our test CA
        client_ctx = ssl.create_default_context()
        ca.configure_trust(client_ctx)

        connector = create_tcp_connector(ssl=client_ctx)
        try:
            async with (
                aiohttp.ClientSession(connector=connector) as session,
                session.get(f"{url}/health") as response,
            ):
                assert response.status == 200
                data = await response.json()
                assert data["status"] == "ok"
        finally:
            await connector.close()
