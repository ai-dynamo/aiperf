# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FastAPIService lifecycle, init, CORS, start/stop, and main."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from aiperf.api.api_service import FastAPIService
from aiperf.common.config import ServiceConfig, UserConfig

# =============================================================================
# Compression encoding selection
# =============================================================================


class TestSelectEncoding:
    """Test compression encoding selection."""

    @pytest.mark.parametrize(
        "accept_encoding,expected",
        [
            pytest.param("zstd, gzip", "zstd", id="prefers-zstd"),
            pytest.param("gzip", "gzip", id="fallback-gzip"),
            pytest.param("deflate, br", "identity", id="unknown-identity-fallback"),
            pytest.param(None, "gzip", id="none-fallback-gzip"),
            pytest.param("", "gzip", id="empty-fallback-gzip"),
            pytest.param("ZSTD, GZIP", "zstd", id="case-insensitive"),
            pytest.param("  zstd  ,  gzip  ", "zstd", id="whitespace-handling"),
        ],
    )  # fmt: skip
    def test_select_encoding(self, accept_encoding: str | None, expected: str) -> None:
        """Test encoding selection based on Accept-Encoding header."""
        from aiperf.common.compression import (
            CompressionEncoding,
            is_zstd_available,
            select_encoding,
        )

        result = select_encoding(accept_encoding)
        expected_encoding = CompressionEncoding(expected)
        if expected_encoding == CompressionEncoding.ZSTD and not is_zstd_available():
            assert result == CompressionEncoding.GZIP
        else:
            assert result == expected_encoding


# =============================================================================
# Service properties
# =============================================================================


class TestServiceBaseUrl:
    """Test the _base_url property."""

    def test_base_url_format(self, mock_fastapi_service: FastAPIService) -> None:
        """Test _base_url returns correct format."""
        mock_fastapi_service.api_host = "0.0.0.0"
        mock_fastapi_service.api_port = 8080

        assert mock_fastapi_service._base_url == "http://0.0.0.0:8080"

    def test_base_url_localhost(self, mock_fastapi_service: FastAPIService) -> None:
        """Test _base_url with localhost."""
        mock_fastapi_service.api_host = "127.0.0.1"
        mock_fastapi_service.api_port = 9999

        assert mock_fastapi_service._base_url == "http://127.0.0.1:9999"


# =============================================================================
# FastAPIService lifecycle tests (init, start, stop, main)
# =============================================================================


class TestFastAPIServiceInit:
    """Test FastAPIService.__init__ via direct instantiation."""

    def test_init_sets_host_port_from_config(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        assert mock_fastapi_service.api_host == "127.0.0.1"
        assert mock_fastapi_service.api_port == 9999

    def test_init_creates_app(self, mock_fastapi_service: FastAPIService) -> None:
        assert mock_fastapi_service.app is not None
        assert mock_fastapi_service.app.title == "AIPerf API"

    def test_init_defaults_server_to_none(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        assert mock_fastapi_service._server is None
        assert mock_fastapi_service._server_task is None

    def test_init_loads_routers(self, mock_fastapi_service: FastAPIService) -> None:
        assert len(mock_fastapi_service._routers) > 0

    def test_init_with_custom_host(
        self, mock_zmq: None, api_user_config: UserConfig
    ) -> None:
        sc = ServiceConfig(api_port=8080, api_host="0.0.0.0")
        service = FastAPIService(
            service_config=sc,
            user_config=api_user_config,
            service_id="api-custom",
        )
        assert service.api_host == "0.0.0.0"
        assert service.api_port == 8080


class TestFastAPIServiceCORSMiddleware:
    """Test CORS middleware is added when cors_origins is set."""

    def test_cors_middleware_added_when_origins_set(
        self,
        mock_zmq: None,
        api_user_config: UserConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "aiperf.common.environment.Environment.API_SERVER",
            type(
                "_Fake",
                (),
                {"HOST": "127.0.0.1", "PORT": 8080, "CORS_ORIGINS": ["*"]},
            )(),
        )
        sc = ServiceConfig(api_port=8080)
        service = FastAPIService(
            service_config=sc,
            user_config=api_user_config,
            service_id="api-cors",
        )
        middleware_names = [m.cls.__name__ for m in service.app.user_middleware]
        assert "CORSMiddleware" in middleware_names

    def test_no_cors_middleware_when_origins_empty(
        self,
        mock_zmq: None,
        api_user_config: UserConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "aiperf.common.environment.Environment.API_SERVER",
            type(
                "_Fake", (), {"HOST": "127.0.0.1", "PORT": 8080, "CORS_ORIGINS": []}
            )(),
        )
        sc = ServiceConfig(api_port=8080)
        service = FastAPIService(
            service_config=sc,
            user_config=api_user_config,
            service_id="api-no-cors",
        )
        middleware_names = [m.cls.__name__ for m in service.app.user_middleware]
        assert "CORSMiddleware" not in middleware_names


class TestFastAPIServiceStartStop:
    """Test _start_api_server and _stop_api_server."""

    @pytest.mark.asyncio
    async def test_start_raises_when_port_not_configured(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        mock_fastapi_service.api_port = None
        with pytest.raises(ValueError, match="API port is not configured"):
            await mock_fastapi_service._start_api_server()

    @pytest.mark.asyncio
    async def test_start_creates_server_and_task(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        mock_server = MagicMock()
        mock_server.serve = AsyncMock()

        with (
            patch("aiperf.api.api_service.uvicorn.Config"),
            patch("aiperf.api.api_service.uvicorn.Server", return_value=mock_server),
        ):
            await mock_fastapi_service._start_api_server()

        assert mock_fastapi_service._server is mock_server
        assert mock_fastapi_service._server_task is not None

        mock_fastapi_service._server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await mock_fastapi_service._server_task

    @pytest.mark.asyncio
    async def test_stop_sets_should_exit_and_waits(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        mock_server = MagicMock()
        completed = asyncio.Event()

        async def fake_serve():
            await completed.wait()

        task = asyncio.create_task(fake_serve())
        mock_fastapi_service._server = mock_server
        mock_fastapi_service._server_task = task

        completed.set()
        await mock_fastapi_service._stop_api_server()

        assert mock_server.should_exit is True

    @pytest.mark.asyncio
    async def test_stop_cancels_on_timeout(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        mock_server = MagicMock()

        async def hang_forever():
            await asyncio.Future()

        task = asyncio.create_task(hang_forever())
        mock_fastapi_service._server = mock_server
        mock_fastapi_service._server_task = task

        real_wait_for = asyncio.wait_for
        call_count = 0

        async def first_call_times_out(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError
            return await real_wait_for(*args, **kwargs)

        with patch(
            "aiperf.api.api_service.asyncio.wait_for",
            side_effect=first_call_times_out,
        ):
            await mock_fastapi_service._stop_api_server()

        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_handles_no_server(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        mock_fastapi_service._server = None
        mock_fastapi_service._server_task = None
        await mock_fastapi_service._stop_api_server()

    @pytest.mark.asyncio
    async def test_stop_propagates_cancelled_error(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test _stop_api_server re-raises CancelledError for cooperative cancellation."""
        mock_server = MagicMock()
        mock_fastapi_service._server = mock_server
        mock_fastapi_service._server_task = asyncio.create_task(asyncio.sleep(100))

        with (
            patch(
                "aiperf.api.api_service.asyncio.wait_for",
                side_effect=asyncio.CancelledError,
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await mock_fastapi_service._stop_api_server()

        assert mock_server.should_exit is True

    @pytest.mark.asyncio
    async def test_on_server_task_done_schedules_stop_on_exception(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test _on_server_task_done schedules stop when server task fails."""
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("server crashed")

        with patch.object(
            mock_fastapi_service, "stop", new_callable=AsyncMock
        ) as mock_stop:
            mock_fastapi_service._on_server_task_done(task)
            assert mock_fastapi_service._stop_task is not None
            await asyncio.sleep(0)
            mock_stop.assert_called_once()

    def test_on_server_task_done_ignores_cancelled(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test _on_server_task_done does nothing for cancelled tasks."""
        task = MagicMock()
        task.cancelled.return_value = True
        mock_fastapi_service._on_server_task_done(task)
        task.exception.assert_not_called()
        assert mock_fastapi_service._stop_task is None

    def test_on_server_task_done_no_exception(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test _on_server_task_done does nothing when task succeeds."""
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        mock_fastapi_service._on_server_task_done(task)
        assert mock_fastapi_service._stop_task is None


class TestFastAPIServiceLifespan:
    """Test FastAPI lifespan hooks."""

    def test_lifespan_logs_startup_and_shutdown(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that lifespan logs on startup and shutdown."""
        mock_fastapi_service.info = MagicMock()

        with TestClient(mock_fastapi_service.app):
            pass

        info_calls = [call[0][0] for call in mock_fastapi_service.info.call_args_list]
        assert any("FastAPI starting" in msg for msg in info_calls)
        assert any("FastAPI stopped" in msg for msg in info_calls)


class TestFastAPIServiceMain:
    """Test the main() entry point."""

    def test_main_calls_bootstrap(self) -> None:
        from aiperf.api.api_service import main
        from aiperf.plugin.enums import ServiceType

        with patch(
            "aiperf.api.api_service.bootstrap_and_run_service"
        ) as mock_bootstrap:
            main()
            mock_bootstrap.assert_called_once_with(ServiceType.API)
