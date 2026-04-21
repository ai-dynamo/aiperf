# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from typing import Any

import orjson

from aiperf.common.enums import ConnectionReuseStrategy
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    ErrorDetails,
    RequestInfo,
    RequestRecord,
)
from aiperf.common.redact import redact_headers
from aiperf.plugin import plugins
from aiperf.plugin.enums import TransportType
from aiperf.transports.base_http_transport import BaseHTTPTransport
from aiperf.transports.base_transports import (
    FirstTokenCallback,
    TransportMetadata,
)
from aiperf.transports.httpcore_client import HttpCoreClient


class HttpCoreLeaseManager(AIPerfLoggerMixin):
    """Manages per-session httpcore clients for sticky-user-sessions strategy.

    Each user session (identified by x_correlation_id) gets a dedicated
    HttpCoreClient with max_connections=1, pinning the session to a single
    backend connection through load balancers.
    """

    def __init__(self, timeout: float | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timeout = timeout
        self._leases: dict[str, HttpCoreClient] = {}

    def get_client(self, session_id: str) -> HttpCoreClient:
        """Get or create a single-connection client for a session."""
        if session_id not in self._leases:
            self._leases[session_id] = HttpCoreClient.create_ephemeral(
                timeout=self._timeout
            )
            self.debug(lambda: f"Created connection lease for session {session_id}")
        return self._leases[session_id]

    async def release(self, session_id: str) -> None:
        """Release and close the client for a session."""
        if session_id in self._leases:
            client = self._leases.pop(session_id)
            await client.close()
            self.debug(lambda: f"Released connection lease for session {session_id}")

    async def close_all(self) -> None:
        """Close all active session clients."""
        clients = list(self._leases.values())
        self._leases.clear()
        for client in clients:
            await client.close()


class HttpCoreTransport(BaseHTTPTransport):
    """HTTP/2 transport implementation using httpcore.

    Provides high-performance HTTP transport with HTTP/2 multiplexing support,
    offering significantly higher concurrency than HTTP/1.1-only implementations.

    Key Features:
        - HTTP/2 multiplexing: ~100 streams per connection
        - Connection pooling: 25 connections (configurable)
        - Total capacity: 2,500+ concurrent requests
        - Automatic HTTP/1.1 fallback for compatibility
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize HTTP/2 transport.

        Args:
            **kwargs: Additional arguments passed to parent BaseTransport
        """
        super().__init__(**kwargs)
        self.httpcore_client: HttpCoreClient | None = None
        self.lease_manager: HttpCoreLeaseManager | None = None

    @property
    def http_client(self) -> HttpCoreClient | None:
        """Return the underlying httpcore client instance."""
        return self.httpcore_client

    @on_init
    async def _init_httpcore_client(self) -> None:
        """Initialize the HttpCoreClient with HTTP/2 support."""
        self.httpcore_client = HttpCoreClient(
            timeout=self.run.cfg.endpoint.timeout,
        )
        if (
            self.run.cfg.endpoint.connection_reuse
            == ConnectionReuseStrategy.STICKY_USER_SESSIONS
        ):
            self.lease_manager = HttpCoreLeaseManager(
                timeout=self.run.cfg.endpoint.timeout
            )

    @on_stop
    async def _close_httpcore_client(self) -> None:
        """Cleanup hook to close httpcore connection pool on stop."""
        if self.lease_manager:
            lease_manager = self.lease_manager
            self.lease_manager = None
            await lease_manager.close_all()
        if self.httpcore_client:
            client = self.httpcore_client
            self.httpcore_client = None
            await client.close()

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return HTTP/2 transport metadata."""
        return TransportMetadata(
            transport_type=TransportType.HTTP2,
            url_schemes=["http", "https"],
        )

    async def send_request(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any],
        *,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send HTTP POST request with JSON payload via httpcore HTTP/2 client.

        Connection behavior depends on the configured connection_reuse strategy:
        - POOLED: Uses the shared connection pool (default)
        - NEVER: Creates a single-connection ephemeral client, closed after use
        - STICKY_USER_SESSIONS: Per-session client with max_connections=1,
          pins each conversation to one backend through load balancers

        Args:
            request_info: Request context and metadata
            payload: JSON-serializable request payload
            first_token_callback: Optional callback fired on first SSE message with ttft_ns

        Returns:
            Request record with responses, timing, and any errors
        """
        if self.httpcore_client is None:
            raise NotInitializedError(
                "HttpCoreTransport not initialized. Call initialize() before send_request()."
            )

        endpoint_metadata = plugins.get_endpoint_metadata(self.run.cfg.endpoint.type)
        if endpoint_metadata.requires_polling:
            return await self._send_video_request_with_polling(request_info, payload)

        start_perf_ns = time.perf_counter_ns()
        headers = None
        reuse_strategy = self.run.cfg.endpoint.connection_reuse

        # Capture reference to avoid race with concurrent shutdown
        lease_manager = self.lease_manager

        try:
            url = self.build_url(request_info)
            headers = self.build_headers(request_info)
            json_bytes = orjson.dumps(payload)

            match reuse_strategy:
                case ConnectionReuseStrategy.NEVER:
                    client = HttpCoreClient.create_ephemeral(
                        timeout=self.run.cfg.endpoint.timeout
                    )
                    try:
                        record = await client.post_request(
                            url,
                            json_bytes,
                            headers,
                            cancel_after_ns=request_info.cancel_after_ns,
                            first_token_callback=first_token_callback,
                        )
                    finally:
                        await client.close()

                case ConnectionReuseStrategy.STICKY_USER_SESSIONS:
                    if lease_manager is None:
                        raise NotInitializedError(
                            "HttpCoreLeaseManager not initialized for sticky-user-sessions strategy"
                        )
                    session_client = lease_manager.get_client(
                        request_info.x_correlation_id
                    )
                    record = await session_client.post_request(
                        url,
                        json_bytes,
                        headers,
                        cancel_after_ns=request_info.cancel_after_ns,
                        first_token_callback=first_token_callback,
                    )

                case _:
                    record = await self.httpcore_client.post_request(
                        url,
                        json_bytes,
                        headers,
                        cancel_after_ns=request_info.cancel_after_ns,
                        first_token_callback=first_token_callback,
                    )

            record.request_headers = redact_headers(headers)

            if (
                reuse_strategy == ConnectionReuseStrategy.STICKY_USER_SESSIONS
                and lease_manager is not None
            ):
                should_release = (
                    request_info.is_final_turn
                    or record.cancellation_perf_ns is not None
                    or record.error is not None
                )
                if should_release:
                    await lease_manager.release(request_info.x_correlation_id)

        except asyncio.CancelledError:
            if (
                reuse_strategy == ConnectionReuseStrategy.STICKY_USER_SESSIONS
                and lease_manager is not None
            ):
                await lease_manager.release(request_info.x_correlation_id)
            raise
        except Exception as e:
            record = RequestRecord(
                request_headers=redact_headers(
                    headers or request_info.endpoint_headers
                ),
                start_perf_ns=start_perf_ns,
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails.from_exception(e),
            )
            self.exception(f"HTTP/2 request failed: {e!r}")
            if (
                reuse_strategy == ConnectionReuseStrategy.STICKY_USER_SESSIONS
                and lease_manager is not None
            ):
                await lease_manager.release(request_info.x_correlation_id)

        return record
