# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight health server mixin for non-API services.

Provides a minimal asyncio HTTP server that serves Kubernetes health probes
without requiring FastAPI overhead. This is ideal for services that don't
need a full HTTP API but still need K8s liveness/readiness probes.

Configuration via environment variables:
    AIPERF_SERVICE_HEALTH_ENABLED=true   # Enable health server
    AIPERF_SERVICE_HEALTH_HOST=0.0.0.0   # Bind to all interfaces
    AIPERF_SERVICE_HEALTH_PORT=8080      # Port for health endpoints
    AIPERF_SERVICE_HEALTH_REQUEST_TIMEOUT=5.0  # Request read timeout

Usage:
    class MyService(HealthServerMixin, BaseComponentService):
        pass  # Health server starts/stops automatically via hooks
"""

from __future__ import annotations

import asyncio
from asyncio import StreamReader, StreamWriter
from multiprocessing import parent_process

from aiperf.common.environment import Environment
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.mixins.health_check_mixin import HealthCheckMixin

# Process-level registry of active health servers to prevent duplicate binds.
# Maps (host, port) -> service_id that owns it. When multiple services run in
# the same process (component-integration tests, Kubernetes controller pod),
# only the first service to initialize starts the health server.
_active_health_servers: dict[tuple[str, int], str] = {}


def _make_response(
    status_code: int, status_text: str, body: str | None = None
) -> bytes:
    """Pre-compute an HTTP response as bytes."""
    body = body or status_text
    return (
        f"HTTP/1.1 {status_code} {status_text}\r\n"
        f"Content-Type: text/plain\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
        f"{body}"
    ).encode()


# Pre-computed responses (avoid string formatting on every request)
_RESP_OK = _make_response(200, "OK", "ok")
_RESP_UNHEALTHY = _make_response(503, "Service Unavailable", "unhealthy")
_RESP_NOT_READY = _make_response(503, "Service Unavailable", "not ready")
_RESP_NOT_FOUND = _make_response(404, "Not Found")
_RESP_BAD_REQUEST = _make_response(400, "Bad Request")
_RESP_METHOD_NOT_ALLOWED = _make_response(405, "Method Not Allowed")


class HealthServerMixin(HealthCheckMixin, AIPerfLifecycleMixin):
    """Lightweight asyncio HTTP server for Kubernetes health probes.

    This mixin provides a minimal HTTP server that only handles:
    - GET /healthz - Liveness probe
    - GET /readyz - Readiness probe

    The server automatically starts/stops via hooks when AIPERF_SERVICE_HEALTH_ENABLED is True.
    Extends HealthCheckMixin (for is_healthy/is_ready) and AIPerfLifecycleMixin (for hooks).
    See module docstring for configuration options.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._health_server: asyncio.Server | None = None
        self._health_server_bind_key: tuple[str, int] | None = None

    @on_init
    async def _health_server_start(self) -> None:
        """Start health server if enabled via environment."""
        if not Environment.SERVICE.HEALTH_ENABLED:
            self.debug("Health server is disabled. Skipping start.")
            return

        # Don't start health server in spawned child processes to avoid port conflicts
        if parent_process() is not None:
            self.debug("Health server skipped in subprocess.")
            return

        host = Environment.SERVICE.HEALTH_HOST
        port = Environment.SERVICE.HEALTH_PORT
        bind_key = (host, port)

        # Check if another service already owns this port in this process
        service_id = getattr(self, "id", str(id(self)))
        if bind_key in _active_health_servers:
            owner = _active_health_servers[bind_key]
            self.debug(
                f"Health server already running on {host}:{port} (owned by {owner}). "
                "Skipping duplicate bind."
            )
            return

        self.debug("Starting health server...")
        try:
            self._health_server = await asyncio.start_server(
                self._handle_health_request, host=host, port=port
            )
        except OSError as e:
            # Fail fast if port is already in use - don't silently continue without health probes
            self.error(
                f"Health server failed to bind to {host}:{port}: {e!r}. "
                "Another process may already be using this port. "
                "Set AIPERF_SERVICE_HEALTH_ENABLED=false to disable health server."
            )
            raise
        _active_health_servers[bind_key] = service_id
        self._health_server_bind_key = bind_key
        self.info(f"Health server started on {host}:{port}")

    @on_stop
    async def _health_server_stop(self) -> None:
        """Stop health server on service shutdown."""
        if self._health_server is None:
            self.debug("Health server is not running. Ignoring stop request.")
            return

        self.debug("Stopping health server...")
        health_server = self._health_server
        self._health_server = None

        # Unregister from process-level registry
        bind_key = self._health_server_bind_key
        if bind_key in _active_health_servers:
            del _active_health_servers[bind_key]

        health_server.close()
        await health_server.wait_closed()
        self.debug("Health server stopped.")

    async def _handle_health_request(
        self, reader: StreamReader, writer: StreamWriter
    ) -> None:
        """Handle incoming HTTP requests for health probes."""
        try:
            request_line = await asyncio.wait_for(
                reader.readline(),
                timeout=Environment.SERVICE.HEALTH_REQUEST_TIMEOUT,
            )
            if not request_line:
                return

            # Split raw HTTP request line to extract method and path without a full HTTP parser
            # Example: "GET /healthz HTTP/1.1\r\n"
            parts = request_line.split(maxsplit=2)
            if len(parts) < 2:
                writer.write(_RESP_BAD_REQUEST)
                await writer.drain()
                return

            method, path = parts[0], parts[1]

            if method != b"GET":
                writer.write(_RESP_METHOD_NOT_ALLOWED)
            elif path == b"/healthz":
                writer.write(_RESP_OK if self.is_healthy() else _RESP_UNHEALTHY)
            elif path == b"/readyz":
                writer.write(_RESP_OK if self.is_ready() else _RESP_NOT_READY)
            else:
                writer.write(_RESP_NOT_FOUND)

            await writer.drain()

        except TimeoutError:
            self.warning("Health request timed out")
        except Exception as e:
            self.error(f"Health server error: {e!r}")
        finally:
            writer.close()
            await writer.wait_closed()
