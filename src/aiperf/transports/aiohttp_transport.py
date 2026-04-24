# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

import aiohttp
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
from aiperf.transports.aiohttp_client import AioHttpClient, create_tcp_connector
from aiperf.transports.base_http_transport import BaseHTTPTransport
from aiperf.transports.base_transports import (
    FirstTokenCallback,
    TransportMetadata,
)


class ConnectionLeaseManager(AIPerfLoggerMixin):
    """Manages connection leases for sticky-user-sessions connection strategy.

    Each user session (identified by x_correlation_id) gets a dedicated TCP connector
    that persists across all turns. The connector is closed when the final turn
    completes, enabling sticky load balancing where all turns of a user session
    hit the same backend server.
    """

    def __init__(self, tcp_kwargs: Mapping[str, Any] | None = None, **kwargs) -> None:
        """Initialize the lease manager.

        Args:
            tcp_kwargs: TCP connector configuration passed to new connectors
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._tcp_kwargs = dict(tcp_kwargs) if tcp_kwargs else {}
        # Map session_id (x_correlation_id) -> TCPConnector
        self._leases: dict[str, aiohttp.TCPConnector] = {}

    def get_connector(self, session_id: str) -> aiohttp.TCPConnector:
        """Get or create a connector for a user session.

        Args:
            session_id: Unique identifier for the user session (x_correlation_id)

        Returns:
            TCP connector dedicated to this user session
        """
        if session_id not in self._leases:
            # Create a new connector with limit=1 for single connection
            # This ensures all requests for this session use the same TCP connection
            connector = create_tcp_connector(limit=1, **self._tcp_kwargs)
            self._leases[session_id] = connector
            self.debug(lambda: f"Created connection lease for session {session_id}")
        return self._leases[session_id]

    async def release_lease(self, session_id: str) -> None:
        """Release and close the connector for a session.

        Should be called when the final turn of a conversation completes,
        or when a request is cancelled (connection becomes dirty).

        Args:
            session_id: Unique identifier for the session (x_correlation_id)
        """
        if session_id in self._leases:
            connector = self._leases.pop(session_id)
            await connector.close()
            self.debug(lambda: f"Released connection lease for session {session_id}")

    async def close_all(self) -> None:
        """Close all active connection leases."""
        leases = list(self._leases.values())
        self._leases.clear()
        for lease in leases:
            await lease.close()


class AioHttpTransport(BaseHTTPTransport):
    """HTTP/1.1 transport implementation using aiohttp.

    Provides high-performance async HTTP client with:
    - Connection pooling and TCP optimization
    - SSE (Server-Sent Events) streaming support
    - Automatic error handling and timing
    - Custom TCP connector configuration
    - Connection reuse strategy support (pooled, never, sticky-user-sessions)
    """

    def __init__(
        self, tcp_kwargs: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Initialize HTTP transport with optional TCP configuration.

        Args:
            tcp_kwargs: TCP connector configuration (socket options, timeouts, etc.)
            **kwargs: Additional arguments passed to parent classes
        """
        super().__init__(**kwargs)
        self.tcp_kwargs = tcp_kwargs or {}
        self.aiohttp_client: AioHttpClient | None = None
        self.lease_manager: ConnectionLeaseManager | None = None

    @property
    def http_client(self) -> AioHttpClient | None:
        """Return the underlying aiohttp client instance."""
        return self.aiohttp_client

    @on_init
    async def _init_aiohttp_client(self) -> None:
        """Initialize the AioHttpClient and lease manager if sticky-user-sessions strategy is used."""
        self.aiohttp_client = AioHttpClient(
            timeout=self.run.cfg.endpoint.timeout,
            tcp_kwargs=self.tcp_kwargs,
            collect_trace_chunks=self.run.cfg.artifacts.trace,
        )
        if (
            self.run.cfg.endpoint.connection_reuse
            == ConnectionReuseStrategy.STICKY_USER_SESSIONS
        ):
            self.lease_manager = ConnectionLeaseManager(tcp_kwargs=self.tcp_kwargs)

    @on_stop
    async def _close_aiohttp_client(self) -> None:
        """Cleanup hook to close aiohttp session on stop (and lease manager if sticky-user-sessions strategy is used)."""
        if self.lease_manager:
            lease_manager = self.lease_manager
            self.lease_manager = None
            await lease_manager.close_all()
        if self.aiohttp_client:
            aiohttp_client = self.aiohttp_client
            self.aiohttp_client = None
            await aiohttp_client.close()

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return HTTP transport metadata."""
        return TransportMetadata(
            transport_type=TransportType.HTTP,
            url_schemes=["http", "https"],
        )

    def _resolve_connector(
        self,
        reuse_strategy: ConnectionReuseStrategy,
        lease_manager: ConnectionLeaseManager | None,
        request_info: RequestInfo,
    ) -> tuple[aiohttp.TCPConnector | None, bool]:
        """Resolve the TCP connector and ownership flag for a request.

        Returns:
            (connector, connector_owner) tuple suitable for aiohttp post_request.
        """
        match reuse_strategy:
            case ConnectionReuseStrategy.NEVER:
                # Create a new connector for this request, and have aiohttp
                # close it when the request is done by setting connector_owner to True
                kwargs = self.tcp_kwargs.copy()
                kwargs["force_close"] = True
                kwargs["limit"] = 1
                kwargs["keepalive_timeout"] = None
                return create_tcp_connector(**kwargs), True

            case ConnectionReuseStrategy.STICKY_USER_SESSIONS:
                if lease_manager is None:
                    raise NotInitializedError(
                        "ConnectionLeaseManager not initialized for sticky-user-sessions strategy"
                    )
                # Use x_correlation_id as the session key - it's the shared ID
                # for all turns in a multi-turn conversation.
                # We manage the connector lifecycle ourselves, so don't let aiohttp close it.
                return lease_manager.get_connector(request_info.x_correlation_id), False

            case ConnectionReuseStrategy.POOLED:
                # Setting connector to None uses the shared pool internally, and connector_owner
                # is set to False to ensure the connector is not closed automatically by aiohttp.
                return None, False

            case _:
                raise ValueError(f"Invalid connection reuse strategy: {reuse_strategy}")

    async def _maybe_release_sticky_lease(
        self,
        reuse_strategy: ConnectionReuseStrategy,
        lease_manager: ConnectionLeaseManager | None,
        request_info: RequestInfo,
        *,
        force: bool,
        record: RequestRecord | None = None,
    ) -> None:
        """Release the sticky-user-session lease when appropriate.

        When `force=True`, always release (used on cancellation/exception paths where
        the connection is dirty). Otherwise release only on final turn, cancellation,
        or recorded error.
        """
        if (
            reuse_strategy != ConnectionReuseStrategy.STICKY_USER_SESSIONS
            or lease_manager is None
        ):
            return
        if not force:
            should_release = request_info.is_final_turn or (
                record is not None
                and (
                    record.cancellation_perf_ns is not None or record.error is not None
                )
            )
            if not should_release:
                return
        await lease_manager.release_lease(request_info.x_correlation_id)

    def _build_error_record(
        self,
        exc: BaseException,
        request_info: RequestInfo,
        headers: dict[str, str] | None,
        start_perf_ns: int,
    ) -> RequestRecord:
        """Construct a RequestRecord capturing an unexpected exception."""
        return RequestRecord(
            request_headers=redact_headers(headers or request_info.endpoint_headers),
            start_perf_ns=start_perf_ns,
            end_perf_ns=time.perf_counter_ns(),
            error=ErrorDetails.from_exception(exc),
        )

    async def send_request(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any],
        *,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send HTTP POST request with JSON payload.

        Connection behavior follows endpoint.connection_reuse:
        POOLED (shared pool), NEVER (one-shot connector), STICKY_USER_SESSIONS
        (lease reused across a conversation's turns; released on final turn/error).
        """
        if self.aiohttp_client is None:
            raise NotInitializedError(
                "AioHttpTransport not initialized. Call initialize() before send_request()."
            )

        start_perf_ns = time.perf_counter_ns()
        headers = None
        reuse_strategy = self.run.cfg.endpoint.connection_reuse
        # Capture lease_manager reference to avoid race with concurrent shutdown
        lease_manager = self.lease_manager

        # Route polling-based endpoints (e.g., video_generation) to polling implementation
        endpoint_metadata = plugins.get_endpoint_metadata(self.run.cfg.endpoint.type)
        if endpoint_metadata.requires_polling:
            return await self._send_video_request_with_polling(request_info, payload)

        try:
            url = self.build_url(request_info)
            headers = self.build_headers(request_info)
            json_bytes = orjson.dumps(payload)
            connector, connector_owner = self._resolve_connector(
                reuse_strategy, lease_manager, request_info
            )
            record = await self.aiohttp_client.post_request(
                url,
                json_bytes,
                headers,
                cancel_after_ns=request_info.cancel_after_ns,
                first_token_callback=first_token_callback,
                connector=connector,
                connector_owner=connector_owner,
            )
            record.request_headers = redact_headers(headers)
            await self._maybe_release_sticky_lease(
                reuse_strategy, lease_manager, request_info, force=False, record=record
            )
        except asyncio.CancelledError:
            # External cancellation (e.g., credit cancellation); connection now dirty.
            await self._maybe_release_sticky_lease(
                reuse_strategy, lease_manager, request_info, force=True
            )
            raise
        except Exception as e:  # noqa: BLE001 - per-request; attach ErrorDetails and return record
            record = self._build_error_record(e, request_info, headers, start_perf_ns)
            self.exception(f"HTTP request failed: {e!r}")
            await self._maybe_release_sticky_lease(
                reuse_strategy, lease_manager, request_info, force=True
            )

        return record
