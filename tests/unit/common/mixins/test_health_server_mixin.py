# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HealthServerMixin."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import LifecycleState
from aiperf.common.mixins import health_server_mixin
from aiperf.common.mixins.health_server_mixin import HealthServerMixin


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


@pytest.fixture(autouse=True)
def clear_health_server_registry():
    """Clear the process-level health server registry before each test."""
    health_server_mixin._ActiveHealthServers._registry.clear()
    yield
    health_server_mixin._ActiveHealthServers._registry.clear()


@pytest.fixture
def mock_env_settings():
    """Fixture to mock Environment.SERVICE settings for health server."""

    def _mock(
        enabled: bool = True,
        host: str = "127.0.0.1",
        port: int = 18080,
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

        with mock_env_settings(enabled=True, port=18080):
            await service._health_server_start()

            assert service._health_server is not None
            service.info.assert_called_once()

            await service._health_server_stop()
            assert service._health_server is None

    @pytest.mark.asyncio
    async def test_server_not_started_when_disabled(self, mock_env_settings) -> None:
        """Test health server does not start when disabled."""
        service = MockServiceWithHealthServer()

        with mock_env_settings(enabled=False, port=18088):
            await service._health_server_start()

            assert service._health_server is None
            service.info.assert_not_called()

    @pytest.mark.asyncio
    async def test_cli_health_port_overrides_env(self, mock_env_settings) -> None:
        """Test --health-port CLI arg overrides environment config."""
        service = MockServiceWithHealthServer()
        service._health_port = 18094

        with mock_env_settings(enabled=True, port=18088):
            await service._health_server_start()

            try:
                assert service._health_server is not None
                status, body = await make_http_request(18094, "/healthz")
                assert status == 200
                assert body == "ok"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_cli_health_port_overrides_disabled_env(
        self, mock_env_settings
    ) -> None:
        """Test --health-port starts server even when HEALTH_ENABLED=false."""
        service = MockServiceWithHealthServer()
        service._health_port = 18095

        with mock_env_settings(enabled=False, port=18088):
            await service._health_server_start()

            try:
                assert service._health_server is not None
                status, body = await make_http_request(18095, "/healthz")
                assert status == 200
                assert body == "ok"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_healthz_returns_ok_when_healthy(self, mock_env_settings) -> None:
        """Test /healthz returns 200 when service is healthy."""
        service = MockServiceWithHealthServer(LifecycleState.RUNNING)

        with mock_env_settings(enabled=True, port=18081):
            await service._health_server_start()

            try:
                status, body = await make_http_request(18081, "/healthz")
                assert status == 200
                assert body == "ok"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_healthz_returns_503_when_failed(self, mock_env_settings) -> None:
        """Test /healthz returns 503 when service has failed."""
        service = MockServiceWithHealthServer(LifecycleState.FAILED)

        with mock_env_settings(enabled=True, port=18082):
            await service._health_server_start()

            try:
                status, body = await make_http_request(18082, "/healthz")
                assert status == 503
                assert body == "unhealthy"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_readyz_returns_ok_when_running(self, mock_env_settings) -> None:
        """Test /readyz returns 200 when service is running."""
        service = MockServiceWithHealthServer(LifecycleState.RUNNING)

        with mock_env_settings(enabled=True, port=18083):
            await service._health_server_start()

            try:
                status, body = await make_http_request(18083, "/readyz")
                assert status == 200
                assert body == "ok"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_readyz_returns_503_when_not_ready(self, mock_env_settings) -> None:
        """Test /readyz returns 503 when service is not ready."""
        service = MockServiceWithHealthServer(LifecycleState.INITIALIZING)

        with mock_env_settings(enabled=True, port=18084):
            await service._health_server_start()

            try:
                status, body = await make_http_request(18084, "/readyz")
                assert status == 503
                assert body == "not ready"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, mock_env_settings) -> None:
        """Test unknown paths return 404."""
        service = MockServiceWithHealthServer()

        with mock_env_settings(enabled=True, port=18085):
            await service._health_server_start()

            try:
                status, body = await make_http_request(18085, "/unknown")
                assert status == 404
                assert body == "Not Found"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_custom_host_and_port(self, mock_env_settings) -> None:
        """Test health server starts on custom host and port."""
        service = MockServiceWithHealthServer()

        with mock_env_settings(enabled=True, host="127.0.0.1", port=18086):
            await service._health_server_start()

            assert service._health_server is not None
            # Verify we can connect
            status, body = await make_http_request(18086, "/healthz")
            assert status == 200
            assert body == "ok"

            await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_state_change_affects_responses(self, mock_env_settings) -> None:
        """Test that changing state affects health responses."""
        service = MockServiceWithHealthServer(LifecycleState.INITIALIZING)

        with mock_env_settings(enabled=True, port=18087):
            await service._health_server_start()

            try:
                # Initially not ready
                status, _ = await make_http_request(18087, "/readyz")
                assert status == 503

                # Change to RUNNING
                service._state = LifecycleState.RUNNING

                # Now should be ready
                status, body = await make_http_request(18087, "/readyz")
                assert status == 200
                assert body == "ok"
            finally:
                await service._health_server_stop()

    @pytest.mark.asyncio
    async def test_server_not_started_in_subprocess(self, mock_env_settings) -> None:
        """Test health server does not start in spawned subprocess."""
        service = MockServiceWithHealthServer()

        with (
            mock_env_settings(enabled=True, port=18089),
            patch(
                "aiperf.common.mixins.health_server_mixin.parent_process",
                return_value=MagicMock(),
            ),
        ):
            await service._health_server_start()

        assert service._health_server is None
        service.debug.assert_any_call("Health server skipped in subprocess.")

    @pytest.mark.asyncio
    async def test_multiple_services_same_port_skips_duplicate(
        self, mock_env_settings
    ) -> None:
        """Test that second service skips binding when port already in use."""
        service1 = MockServiceWithHealthServer()
        service1.id = "service-1"
        service2 = MockServiceWithHealthServer()
        service2.id = "service-2"

        with mock_env_settings(enabled=True, port=18090):
            await service1._health_server_start()
            assert service1._health_server is not None
            service1.info.assert_called_once()

            await service2._health_server_start()
            assert service2._health_server is None
            service2.debug.assert_any_call(
                "Health server already running on 127.0.0.1:18090 (owned by service-1). "
                "Skipping duplicate bind."
            )

            status, body = await make_http_request(18090, "/healthz")
            assert status == 200
            assert body == "ok"

            await service1._health_server_stop()

    @pytest.mark.asyncio
    async def test_registry_cleared_on_stop(self, mock_env_settings) -> None:
        """Test that registry entry is removed when service stops."""
        service = MockServiceWithHealthServer()
        service.id = "test-service"

        with mock_env_settings(enabled=True, port=18091):
            await service._health_server_start()
            assert (
                "127.0.0.1",
                18091,
            ) in health_server_mixin._ActiveHealthServers._registry

            await service._health_server_stop()
            assert (
                "127.0.0.1",
                18091,
            ) not in health_server_mixin._ActiveHealthServers._registry

    @pytest.mark.asyncio
    async def test_second_service_can_start_after_first_stops(
        self, mock_env_settings
    ) -> None:
        """Test that second service can start after first service stops."""
        service1 = MockServiceWithHealthServer()
        service1.id = "service-1"
        service2 = MockServiceWithHealthServer()
        service2.id = "service-2"

        with mock_env_settings(enabled=True, port=18092):
            await service1._health_server_start()
            assert service1._health_server is not None
            await service1._health_server_stop()
            assert service1._health_server is None

            await service2._health_server_start()
            assert service2._health_server is not None

            status, body = await make_http_request(18092, "/healthz")
            assert status == 200

            await service2._health_server_stop()

    @pytest.mark.asyncio
    async def test_binding_failure_raises_exception(self, mock_env_settings) -> None:
        """Test that binding failure raises OSError to crash the process."""
        import errno

        service = MockServiceWithHealthServer()
        service.id = "test-service"

        with (
            mock_env_settings(enabled=True, port=18093),
            patch(
                "aiperf.common.mixins.health_server_mixin.asyncio.start_server",
                side_effect=OSError(errno.EADDRINUSE, "Address already in use"),
            ),
            pytest.raises(OSError) as exc_info,
        ):
            await service._health_server_start()

        assert exc_info.value.errno == errno.EADDRINUSE

        service.error.assert_called_once()
        error_msg = service.error.call_args[0][0]
        assert "failed to bind" in error_msg
        assert "127.0.0.1:18093" in error_msg
