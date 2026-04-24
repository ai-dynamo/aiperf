# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP health server for Kubernetes liveness and readiness probes.

This module provides a minimal async HTTP server that exposes:
- /healthz: Liveness probe - returns 200 if the process is alive
- /readyz: Readiness probe - returns 200 if the service is ready to accept traffic

The server runs on the async event loop and is designed to be lightweight
with minimal overhead for the service's main operations.
"""

from collections.abc import Callable

from aiohttp import web

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)


class HealthServer:
    """Minimal HTTP server for Kubernetes health probes.

    The server exposes /healthz and /readyz endpoints. By default, both
    return 200 OK. Services can customize the readiness check by providing
    a callback function.

    Example:
        ```python
        server = HealthServer(port=8080)
        await server.start()

        # Later, when shutting down:
        await server.stop()
        ```

    With custom readiness check:
        ```python
        def is_ready() -> bool:
            return service.is_initialized and service.is_connected

        server = HealthServer(port=8080, readiness_check=is_ready)
        ```
    """

    def __init__(
        self,
        port: int,
        host: str = "0.0.0.0",
        readiness_check: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize the health server.

        Args:
            port: Port to listen on.
            host: Host to bind to (default: 0.0.0.0 for all interfaces).
            readiness_check: Optional callback that returns True if the service
                is ready to accept traffic. If not provided, /readyz always returns 200.
        """
        self._port = port
        self._host = host
        self._readiness_check = readiness_check
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    async def start(self) -> None:
        """Start the health server."""
        self._app = web.Application()
        self._app.router.add_get("/healthz", self._handle_healthz)
        self._app.router.add_get("/readyz", self._handle_readyz)

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        _logger.info(f"Health server started on {self._host}:{self._port}")

    async def stop(self) -> None:
        """Stop the health server."""
        if self._runner:
            await self._runner.cleanup()
            _logger.info("Health server stopped")

    def close(self) -> None:
        """Compatibility alias for asyncio.Server interface. Use stop() instead."""
        # Actual cleanup happens in wait_closed()
        pass

    async def wait_closed(self) -> None:
        """Compatibility alias for asyncio.Server interface. Use stop() instead."""
        await self.stop()

    async def _handle_healthz(self, request: web.Request) -> web.Response:
        """Handle liveness probe. Always returns 200 if the process is alive."""
        return web.Response(text="ok", status=200)

    async def _handle_readyz(self, request: web.Request) -> web.Response:
        """Handle readiness probe. Returns 200 if ready, 503 otherwise."""
        if self._readiness_check is None or self._readiness_check():
            return web.Response(text="ok", status=200)
        else:
            return web.Response(text="not ready", status=503)
