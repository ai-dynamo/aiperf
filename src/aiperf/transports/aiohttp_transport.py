# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import aiohttp

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

# Mirror the worker's UserSessionManager bound (workers/session_manager.py
# DEFAULT_MAX_SESSIONS): a sticky session whose conversation is abandoned before
# its final turn (e.g. --request-count dataset recycling, or a worker reassigned
# mid-conversation) never reaches the is_final_turn/cancel release path, so its
# connector would otherwise live in _leases until on_stop. Capping retention with
# the same LRU ceiling the session manager enforces ties connector teardown to the
# same authoritative bound and reclaims abandoned leases instead of leaking them.
DEFAULT_MAX_LEASES = 100_000


class ConnectionLeaseManager(AIPerfLoggerMixin):
    """Manages connection leases for sticky-user-sessions connection strategy.

    Each user session (identified by x_correlation_id) gets a dedicated TCP connector
    that persists across all turns. The connector is closed when the final turn
    completes, enabling sticky load balancing where all turns of a user session
    hit the same backend server.

    Retention is LRU-bounded by ``max_leases`` so sessions abandoned before their
    final turn (never hitting the release path) cannot accumulate connectors
    unboundedly; the least-recently-used lease is closed when the bound is exceeded.
    """

    def __init__(
        self,
        tcp_kwargs: Mapping[str, Any] | None = None,
        max_leases: int = DEFAULT_MAX_LEASES,
        **kwargs,
    ) -> None:
        """Initialize the lease manager.

        Args:
            tcp_kwargs: TCP connector configuration passed to new connectors
            max_leases: Maximum number of concurrent leases retained before the
                least-recently-used connector is closed and evicted
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        if max_leases < 1:
            raise ValueError(f"max_leases ({max_leases}) must be >= 1")
        self._tcp_kwargs = dict(tcp_kwargs) if tcp_kwargs else {}
        self._max_leases = max_leases
        # Map session_id (x_correlation_id) -> TCPConnector, ordered by recency of use
        self._leases: OrderedDict[str, aiohttp.TCPConnector] = OrderedDict()
        # Connectors evicted by the LRU bound are closed off the request path;
        # close_all() drains any still-pending eviction closes.
        self._eviction_tasks: set[asyncio.Task[None]] = set()

    def get_connector(self, session_id: str) -> aiohttp.TCPConnector:
        """Get or create a connector for a user session.

        Args:
            session_id: Unique identifier for the user session (x_correlation_id)

        Returns:
            TCP connector dedicated to this user session
        """
        connector = self._leases.get(session_id)
        if connector is None:
            # Create a new connector with limit=1 for single connection
            # This ensures all requests for this session use the same TCP connection
            connector = create_tcp_connector(limit=1, **self._tcp_kwargs)
            self._leases[session_id] = connector
            self.debug(lambda: f"Created connection lease for session {session_id}")
            self._evict_overflow()
        else:
            self._leases.move_to_end(session_id)
        return connector

    def _evict_overflow(self) -> None:
        """Close and drop least-recently-used leases beyond the retention bound."""
        while len(self._leases) > self._max_leases:
            evicted_id, evicted = self._leases.popitem(last=False)
            self.debug(
                lambda eid=evicted_id: f"Evicting LRU connection lease for session {eid}"
            )
            task = asyncio.ensure_future(evicted.close())
            self._eviction_tasks.add(task)
            task.add_done_callback(self._eviction_tasks.discard)

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
        """Close all active connection leases and drain pending LRU-eviction closes."""
        leases = list(self._leases.values())
        self._leases.clear()
        for lease in leases:
            await lease.close()
        if self._eviction_tasks:
            pending = list(self._eviction_tasks)
            self._eviction_tasks.clear()
            await asyncio.gather(*pending, return_exceptions=True)


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
            # Multipart endpoints carry form fields plus base64 file descriptors.
            # Serializing to bytes here gives aiohttp the exact boundary header and
            # lets cancellation track request-sent progress by byte count.
            body = await self._build_request_body(payload, headers)
            connector, connector_owner = self._resolve_connector(
                reuse_strategy, lease_manager, request_info
            )
            record = await self.aiohttp_client.post_request(
                url,
                body,
                headers,
                cancel_after_ns=request_info.cancel_after_ns,
                first_token_callback=first_token_callback,
                connector=connector,
                connector_owner=connector_owner,
            )
            record.request_headers = redact_headers(headers)
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

        # Post-success lease release runs outside the request try: the request
        # already completed, so a teardown failure must not discard the record.
        try:
            await self._maybe_release_sticky_lease(
                reuse_strategy, lease_manager, request_info, force=False, record=record
            )
        except Exception as e:  # noqa: BLE001 - cleanup failure must not fail the completed request
            self.exception(
                f"Sticky-lease release failed after successful request: {e!r}"
            )

        return record
