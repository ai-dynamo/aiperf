# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HealthServerMixin."""

import asyncio
import socket
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import LifecycleState
from aiperf.common.mixins.health_server_mixin import (
    HealthServerMixin,
    _active_health_servers,
)


class MockServiceWithHealthServer(HealthServerMixin):
    """Mock service for testing HealthServerMixin."""

    def __init__(self, state: LifecycleState = LifecycleState.RUNNING) -> None:
        """Initialize mock service."""
        # Set _state directly (property has no setter)
        self._state = state
        self.id = "test-health-server"
        # Mock logging methods (normally from AIPerfLoggerMixin)
        self.debug = MagicMock()
        self.info = MagicMock()
        self.warning = MagicMock()
        self.error = MagicMock()
        self._health_server = None
        # Required by AIPerfLifecycleMixin but not used in tests
        self.initialized_event = MagicMock()
        self.started_event = MagicMock()
        self.stopped_event = MagicMock()

    @property
    def state(self) -> LifecycleState:
        """Return the current state."""
        return self._state


async def make_http_request(port: int, path: str) -> tuple[int, str]:
    """Make a simple HTTP GET request.

    Args:
        port: Port to connect to.
        path: Request path (e.g., "/healthz").

    Returns:
        Tuple of (status_code, body).
    """
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        request = f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"
        writer.write(request.encode())
        await writer.drain()

        # Read response
        response = await asyncio.wait_for(reader.read(1024), timeout=5.0)
        response_str = response.decode()

        # Parse status code from first line
        first_line = response_str.split("\r\n")[0]
        status_code = int(first_line.split()[1])

        # Get body (after double CRLF)
        body = (
            response_str.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in response_str else ""
        )

        return status_code, body
    finally:
        writer.close()
        await writer.wait_closed()


def _bound_port(service: HealthServerMixin) -> int:
    """Return the port the service's health server actually bound to.

    Tests configure ``HEALTH_PORT=0`` so the kernel hands out a free ephemeral
    port. Hardcoded ports collide with unrelated listeners on the developer's
    machine (a stray ``kubectl port-forward`` is enough) and with parallel
    xdist workers, turning an unrelated environment detail into a test failure.
    """
    assert service._health_server is not None, "health server is not running"
    return service._health_server.sockets[0].getsockname()[1]


def _free_port() -> int:
    """Reserve and release an ephemeral port, returning its number.

    Used only by the test that must pass an explicit port to exercise the
    configured-port path. The kernel does not immediately re-hand out a just
    released port, so the reuse window is wide enough in practice.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(autouse=True)
def _reset_health_server_registry():
    """Clear the process-level bind registry around every test.

    ``_active_health_servers`` is module-global: a leaked entry makes the next
    test's ``_health_server_start`` silently skip binding, which surfaces as a
    confusing ``_health_server is None`` failure far from the real cause.
    """
    _active_health_servers.clear()
    yield
    _active_health_servers.clear()


@pytest.fixture
def mock_env_settings():
    """Fixture to mock Environment.SERVICE settings for health server."""

    def _mock(
        enabled: bool = True,
        host: str = "127.0.0.1",
        port: int = 0,
        request_timeout: float = 5.0,
    ):
        return patch.multiple(
            "aiperf.common.mixins.health_server_mixin.Environment.SERVICE",
            HEALTH_ENABLED=enabled,
            HEALTH_HOST=host,
            HEALTH_PORT=port,
            HEALTH_REQUEST_TIMEOUT=request_timeout,
        )

    return _mock


class TestHealthServerMixin:
    """Test HealthServerMixin functionality."""

    @pytest.mark.asyncio
    async def test_start_and_stop_server(self, mock_env_settings) -> None:
        """Test starting and stopping the health server."""
        service = MockServiceWithHealthServer()

        with mock_env_settings(enabled=True):
            await service._health_server_start()

            assert service._health_server is not None
            service.info.assert_called_once()

            await service._health_server_stop()
            assert service._health_server is None

    @pytest.mark.asyncio
    async def test_server_not_started_when_disabled(self, mock_env_settings) -> None:
        """Test health server does not start when disabled."""
        service = MockServiceWithHealthServer()

        with mock_env_settings(enabled=False):
            await service._health_server_start()

            assert service._health_server is None
            service.info.assert_not_called()

    @pytest.mark.asyncio
    async def test_healthz_returns_ok_when_healthy(self, mock_env_settings) -> None:
        """Test /healthz returns 200 when service is healthy."""
        service = MockServiceWithHealthServer(LifecycleState.RUNNING)

        with mock_env_settings(enabled=True):
            await service._health_server_start()

            try:
                status, body = await make_http_request(_bound_port(service), "/healthz")
                assert status == 200
                assert body == "ok"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_healthz_returns_503_when_failed(self, mock_env_settings) -> None:
        """Test /healthz returns 503 when service has failed."""
        service = MockServiceWithHealthServer(LifecycleState.FAILED)

        with mock_env_settings(enabled=True):
            await service._health_server_start()

            try:
                status, body = await make_http_request(_bound_port(service), "/healthz")
                assert status == 503
                assert body == "unhealthy"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_readyz_returns_ok_when_running(self, mock_env_settings) -> None:
        """Test /readyz returns 200 when service is running."""
        service = MockServiceWithHealthServer(LifecycleState.RUNNING)

        with mock_env_settings(enabled=True):
            await service._health_server_start()

            try:
                status, body = await make_http_request(_bound_port(service), "/readyz")
                assert status == 200
                assert body == "ok"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_readyz_returns_503_when_not_ready(self, mock_env_settings) -> None:
        """Test /readyz returns 503 when service is not ready."""
        service = MockServiceWithHealthServer(LifecycleState.INITIALIZING)

        with mock_env_settings(enabled=True):
            await service._health_server_start()

            try:
                status, body = await make_http_request(_bound_port(service), "/readyz")
                assert status == 503
                assert body == "not ready"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, mock_env_settings) -> None:
        """Test unknown paths return 404."""
        service = MockServiceWithHealthServer()

        with mock_env_settings(enabled=True):
            await service._health_server_start()

            try:
                status, body = await make_http_request(_bound_port(service), "/unknown")
                assert status == 404
                assert body == "Not Found"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_custom_host_and_port(self, mock_env_settings) -> None:
        """Test health server honors the configured host and port."""
        service = MockServiceWithHealthServer()
        port = _free_port()

        with mock_env_settings(enabled=True, host="127.0.0.1", port=port):
            await service._health_server_start()

            try:
                assert service._health_server is not None
                host, bound_port = service._health_server.sockets[0].getsockname()[:2]
                assert (host, bound_port) == ("127.0.0.1", port)

                status, body = await make_http_request(port, "/healthz")
                assert status == 200
                assert body == "ok"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_state_change_affects_responses(self, mock_env_settings) -> None:
        """Test that changing state affects health responses."""
        service = MockServiceWithHealthServer(LifecycleState.INITIALIZING)

        with mock_env_settings(enabled=True):
            await service._health_server_start()

            try:
                # Initially not ready
                status, _ = await make_http_request(_bound_port(service), "/readyz")
                assert status == 503

                # Change to RUNNING
                service._state = LifecycleState.RUNNING

                # Now should be ready
                status, body = await make_http_request(_bound_port(service), "/readyz")
                assert status == 200
                assert body == "ok"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_server_not_started_in_subprocess(self, mock_env_settings) -> None:
        """Test health server does not start in spawned subprocess."""
        service = MockServiceWithHealthServer()

        with (
            mock_env_settings(enabled=True),
            patch(
                "aiperf.common.mixins.health_server_mixin.parent_process",
                return_value=MagicMock(),
            ),
        ):
            await service._health_server_start()

        assert service._health_server is None
        service.debug.assert_any_call("Health server skipped in subprocess.")
