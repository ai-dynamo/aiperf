# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

import aiohttp
import orjson

from aiperf.common.enums import ConnectionReuseStrategy, VideoJobStatus
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    BinaryResponse,
    ErrorDetails,
    RequestInfo,
    RequestRecord,
    TextResponse,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import TransportType
from aiperf.transports.aiohttp_client import AioHttpClient, create_tcp_connector
from aiperf.transports.base_transports import (
    BaseTransport,
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


class AioHttpTransport(BaseTransport):
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

    @on_init
    async def _init_aiohttp_client(self) -> None:
        """Initialize the AioHttpClient and lease manager if sticky-user-sessions strategy is used."""
        self.aiohttp_client = AioHttpClient(
            timeout=self.model_endpoint.endpoint.timeout, tcp_kwargs=self.tcp_kwargs
        )
        if (
            self.model_endpoint.endpoint.connection_reuse_strategy
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

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Build HTTP-specific headers based on streaming mode.

        Args:
            request_info: Request context with endpoint configuration

        Returns:
            HTTP headers (Content-Type and Accept)
        """
        accept = (
            "text/event-stream"
            if request_info.model_endpoint.endpoint.streaming
            else "application/json"
        )
        return {"Content-Type": "application/json", "Accept": accept}

    def get_url(self, request_info: RequestInfo) -> str:
        """Build HTTP URL from base_url and endpoint path.

        Constructs the full URL by combining the base URL with the endpoint path
        from metadata or custom endpoint. Adds http:// scheme if missing.

        When multiple URLs are configured, uses request_info.url_index to select
        the appropriate URL for load balancing.

        Args:
            request_info: Request context with model endpoint info

        Returns:
            Complete HTTP URL with scheme and endpoint path
        """
        endpoint_info = request_info.model_endpoint.endpoint

        # Start with base URL - use url_index for multi-URL load balancing
        base_url = endpoint_info.get_url(request_info.url_index).rstrip("/")

        # Determine the endpoint path
        if endpoint_info.custom_endpoint:
            # Use custom endpoint path if provided
            path = endpoint_info.custom_endpoint.lstrip("/")
            url = f"{base_url}/{path}"
        else:
            # Get endpoint path from endpoint metadata
            endpoint_metadata = plugins.get_endpoint_metadata(endpoint_info.type)
            endpoint_path = endpoint_metadata.endpoint_path
            if (
                self.model_endpoint.endpoint.streaming
                and endpoint_metadata.streaming_path is not None
            ):
                endpoint_path = endpoint_metadata.streaming_path
            if not endpoint_path:
                # No endpoint path, just use base URL
                url = base_url

            else:
                path = endpoint_path.lstrip("/")
                # Handle /v1 base URL with v1/ path prefix to avoid duplication
                if base_url.endswith("/v1") and path.startswith("v1/"):
                    path = path.removeprefix("v1/")
                url = f"{base_url}/{path}"
        return url if url.startswith("http") else f"http://{url}"

    async def send_request(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any],
        *,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send HTTP POST request with JSON payload.

        Connection behavior depends on the configured connection_reuse_strategy:
        - POOLED: Uses shared connection pool (default aiohttp behavior)
        - NEVER: Creates a new connection for each request, closed after
        - STICKY_USER_SESSIONS: Reuses connection across conversation turns, closed on final turn

        Args:
            request_info: Request context and metadata (includes cancel_after_ns)
            payload: JSON-serializable request payload
            first_token_callback: Optional callback fired on first SSE message with ttft_ns

        Returns:
            Request record with responses, timing, and any errors
        """
        if self.aiohttp_client is None:
            raise NotInitializedError(
                "AioHttpTransport not initialized. Call initialize() before send_request()."
            )

        start_perf_ns = time.perf_counter_ns()
        headers = None
        reuse_strategy = self.model_endpoint.endpoint.connection_reuse_strategy

        # Capture lease_manager reference to avoid race with concurrent shutdown
        lease_manager = self.lease_manager

        # Route polling-based endpoints (e.g., video_generation) to polling implementation
        endpoint_metadata = plugins.get_endpoint_metadata(
            request_info.model_endpoint.endpoint.type
        )
        if endpoint_metadata.requires_polling:
            return await self._send_video_request_with_polling(request_info, payload)

        try:
            url = self.build_url(request_info)
            headers = self.build_headers(request_info)
            json_str = orjson.dumps(payload).decode("utf-8")

            match reuse_strategy:
                case ConnectionReuseStrategy.NEVER:
                    # Create a new connector for this request, and have aiohttp
                    # close it when the request is done by setting connector_owner to True
                    kwargs = self.tcp_kwargs.copy()
                    kwargs["force_close"] = True
                    kwargs["limit"] = 1
                    kwargs["keepalive_timeout"] = None
                    connector = create_tcp_connector(**kwargs)
                    connector_owner = True

                case ConnectionReuseStrategy.STICKY_USER_SESSIONS:
                    if lease_manager is None:
                        raise NotInitializedError(
                            "ConnectionLeaseManager not initialized for sticky-user-sessions strategy"
                        )
                    # Use x_correlation_id as the session key - it's the shared ID
                    # for all turns in a multi-turn conversation.
                    connector = lease_manager.get_connector(
                        request_info.x_correlation_id
                    )
                    # We are going to manage the connector lifecycle ourselves, so we don't want aiohttp to close it.
                    connector_owner = False

                case ConnectionReuseStrategy.POOLED:
                    # Setting connector to None uses the shared pool internally, and connector_owner
                    # is set to False to ensure the connector is not closed automatically by aiohttp.
                    connector = None
                    connector_owner = False

                case _:
                    raise ValueError(
                        f"Invalid connection reuse strategy: {self.model_endpoint.endpoint.connection_reuse_strategy}"
                    )

            record = await self.aiohttp_client.post_request(
                url,
                json_str,
                headers,
                cancel_after_ns=request_info.cancel_after_ns,
                first_token_callback=first_token_callback,
                connector=connector,
                connector_owner=connector_owner,
            )
            record.request_headers = headers

            # Release lease for sticky-user-sessions strategy if it's the final turn of the conversation,
            # or the request was cancelled (connection is now dirty/closed), or there was an error.
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
                    await lease_manager.release_lease(request_info.x_correlation_id)

        except asyncio.CancelledError:
            # Task was cancelled externally (e.g., credit cancellation from router)
            # Release the lease since the connection is now dirty/unusable
            if (
                reuse_strategy == ConnectionReuseStrategy.STICKY_USER_SESSIONS
                and lease_manager is not None
            ):
                await lease_manager.release_lease(request_info.x_correlation_id)
            raise
        except Exception as e:
            record = RequestRecord(
                request_headers=headers or request_info.endpoint_headers,
                start_perf_ns=start_perf_ns,
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails.from_exception(e),
            )
            self.exception(f"HTTP request failed: {e!r}")
            # Release lease on exception - connection is likely broken
            if (
                reuse_strategy == ConnectionReuseStrategy.STICKY_USER_SESSIONS
                and lease_manager is not None
            ):
                await lease_manager.release_lease(request_info.x_correlation_id)

        return record

    def _parse_video_response(
        self,
        record: RequestRecord,
        context: str,
    ) -> tuple[dict[str, Any], TextResponse] | ErrorDetails:
        """Parse JSON response from a video API request record.

        Args:
            record: The request record to parse
            context: Description for error messages (e.g., "submit", "poll")

        Returns:
            Tuple of (parsed_json, text_response) on success, or ErrorDetails on failure
        """
        if record.error:
            return record.error
        if not record.responses:
            return ErrorDetails(
                type="VideoGenerationError",
                message=f"No response from video {context}",
                code=500,
            )
        response = record.responses[0]
        if not isinstance(response, TextResponse):
            return ErrorDetails(
                type="VideoGenerationError",
                message=f"Unexpected response type from video {context}",
                code=500,
            )
        try:
            return orjson.loads(response.text), response
        except orjson.JSONDecodeError:
            snippet = response.text[:200] if response.text else "<empty>"
            return ErrorDetails(
                type="VideoGenerationError",
                message=f"Invalid JSON in video {context} response (status {record.status}): {snippet}",
                code=500,
            )

    async def _submit_video_job(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[str, TextResponse] | ErrorDetails:
        """Submit video generation job via POST /v1/videos.

        Returns (job_id, response) on success, ErrorDetails on failure.
        """
        if self.aiohttp_client is None:
            raise NotInitializedError("AioHttpClient not initialized")
        record = await self.aiohttp_client.post_request(
            url, orjson.dumps(payload).decode("utf-8"), headers
        )
        result = self._parse_video_response(record, "submit")
        if isinstance(result, ErrorDetails):
            return result

        job_data, response = result
        job_id = job_data.get("id")
        if not job_id:
            return ErrorDetails(
                type="VideoGenerationError",
                message=f"No job ID returned: {job_data}",
                code=500,
            )
        latency_ms = (record.end_perf_ns - record.start_perf_ns) / 1e6
        self.info(f"Video job {job_id} submitted ({latency_ms:.0f}ms)")
        return job_id, response

    async def _poll_video_job(
        self,
        job_id: str,
        poll_url: str,
        headers: dict[str, str],
        timeout: float,
        poll_interval: float,
    ) -> tuple[dict[str, Any], float] | ErrorDetails:
        """Poll video job until completed/failed. Returns (data, elapsed) or error."""
        if self.aiohttp_client is None:
            raise NotInitializedError("AioHttpClient not initialized")
        self.info(f"Polling video job {job_id}")
        poll_start = time.perf_counter_ns()

        while (time.perf_counter_ns() - poll_start) / 1e9 < timeout:
            record = await self.aiohttp_client.get_request(poll_url, headers)
            result = self._parse_video_response(record, "poll")
            if isinstance(result, ErrorDetails):
                return result

            data, _ = result
            status = data.get("status", "")

            if status == VideoJobStatus.COMPLETED:
                elapsed = (time.perf_counter_ns() - poll_start) / 1e9
                self.info(f"Video job {job_id} completed in {elapsed:.1f}s")
                return data, elapsed

            if status == VideoJobStatus.FAILED:
                error_info = data.get("error", {})
                msg = (
                    error_info.get("message", "Unknown error")
                    if isinstance(error_info, dict)
                    else str(error_info)
                )
                self.error(f"Video job {job_id} failed: {msg}")
                return ErrorDetails(
                    type="VideoGenerationError",
                    message=f"Video generation failed: {msg}",
                    code=500,
                )

            await asyncio.sleep(poll_interval)

        self.error(f"Video job {job_id} timed out after {timeout}s")
        return ErrorDetails(
            type="TimeoutError",
            message=f"Video generation timed out after {timeout}s",
            code=504,
        )

    async def _download_video_content(
        self,
        job_id: str,
        content_url: str,
        headers: dict[str, str],
    ) -> bytes | ErrorDetails:
        """Download video content via GET /v1/videos/{id}/content.

        Returns video bytes on success, ErrorDetails on failure.
        Used when --download-video-content is enabled.
        """
        if self.aiohttp_client is None:
            raise NotInitializedError("AioHttpClient not initialized")
        try:
            record = await self.aiohttp_client.get_request(content_url, headers)
            if record.error:
                return ErrorDetails(
                    type="VideoDownloadError",
                    message=f"Failed to download video {job_id}: {record.error}",
                    code=record.status or 500,
                )
            if record.responses and isinstance(record.responses[0], BinaryResponse):
                self.info(
                    f"Video {job_id} downloaded ({len(record.responses[0].raw_bytes)} bytes)"
                )
                return record.responses[0].raw_bytes
            return ErrorDetails(
                type="VideoDownloadError",
                message=f"No content returned for video {job_id}",
                code=500,
            )
        except Exception as e:
            return ErrorDetails(
                type="VideoDownloadError",
                message=f"Failed to download video {job_id}: {e!r}",
                code=500,
            )

    async def _send_video_request_with_polling(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any],
    ) -> RequestRecord:
        """Send video generation request and poll until complete."""
        if self.aiohttp_client is None:
            raise NotInitializedError("AioHttpClient not initialized")

        start_ns = time.perf_counter_ns()
        headers = self.build_headers(request_info)
        responses: list[TextResponse | BinaryResponse] = []

        def make_record(
            error: ErrorDetails | None = None, status: int | None = None
        ) -> RequestRecord:
            return RequestRecord(
                request_info=request_info,
                request_headers=headers,
                start_perf_ns=start_ns,
                end_perf_ns=time.perf_counter_ns(),
                responses=responses,
                error=error,
                status=status,
            )

        # Use build_url to respect custom endpoints and plugin metadata
        submit_url = self.build_url(request_info)

        # Check if video download is enabled via --download-video-content
        download_content = request_info.model_endpoint.endpoint.download_video_content

        try:
            # Submit job
            result = await self._submit_video_job(submit_url, payload, headers)
            if isinstance(result, ErrorDetails):
                return make_record(error=result)
            job_id, submit_response = result
            responses.append(submit_response)

            # Poll for completion — derive poll URL from submit URL + job ID
            poll_url = f"{submit_url.rstrip('/')}/{job_id}"
            poll_result = await self._poll_video_job(
                job_id,
                poll_url,
                headers,
                timeout=request_info.model_endpoint.endpoint.timeout,
                poll_interval=Environment.HTTP.VIDEO_POLL_INTERVAL,
            )
            if isinstance(poll_result, ErrorDetails):
                return make_record(error=poll_result)

            data, _ = poll_result
            responses.append(
                TextResponse(
                    perf_ns=time.perf_counter_ns(),
                    content_type="application/json",
                    text=orjson.dumps(data).decode(),
                )
            )

            # Optional: download video content if requested
            if download_content:
                content_url = data.get("url") or f"{poll_url}/content"
                download_result = await self._download_video_content(
                    job_id, content_url, headers
                )
                if isinstance(download_result, ErrorDetails):
                    return make_record(error=download_result)
                # Video bytes downloaded successfully (not added to responses - too large)

            return make_record(status=200)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.exception(f"Video generation failed: {e!r}")
            return make_record(error=ErrorDetails.from_exception(e))
