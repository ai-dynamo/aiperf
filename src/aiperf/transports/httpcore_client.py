# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import ssl
import time
from typing import TYPE_CHECKING, Any

import httpcore

from aiperf.common.environment import Environment
from aiperf.common.exceptions import SSEResponseError
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    BinaryResponse,
    ErrorDetails,
    RequestRecord,
    TextResponse,
)
from aiperf.common.models.trace_models import BaseTraceData
from aiperf.transports.http_defaults import HttpCoreDefaults, SocketDefaults
from aiperf.transports.sse_utils import AsyncSSEStreamReader

if TYPE_CHECKING:
    from aiperf.transports.base_transports import FirstTokenCallback


class HttpCoreClient(AIPerfLoggerMixin):
    """High-performance HTTP client using httpcore with HTTP/2 multiplexing support.

    Supports multiple concurrent streams over a single TCP connection, with automatic
    protocol negotiation (HTTP/2 or fallback to HTTP/1.1).
    """

    def __init__(self, timeout: float | None = None, **kwargs: Any) -> None:
        """Initialize the httpcore client with HTTP/2 support and connection pooling.

        Args:
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to parent AIPerfLoggerMixin
        """
        super().__init__(**kwargs)

        max_connections = HttpCoreDefaults.calculate_max_connections()

        self.debug(
            lambda: (
                f"Initializing httpcore client: {max_connections} connections, "
                f"~{max_connections * HttpCoreDefaults.STREAMS_PER_CONNECTION} stream capacity"
            )
        )

        ssl_context = ssl.create_default_context()
        if not Environment.HTTP.SSL_VERIFY:
            self.warning("TLS certificate verification is DISABLED - this is insecure!")
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        self.pool = httpcore.AsyncConnectionPool(
            http1=HttpCoreDefaults.HTTP1,
            http2=HttpCoreDefaults.HTTP2,
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
            keepalive_expiry=HttpCoreDefaults.KEEPALIVE_EXPIRY,
            retries=HttpCoreDefaults.RETRIES,
            socket_options=SocketDefaults.build_socket_options(),
            ssl_context=ssl_context,
        )

        self.timeout_seconds = timeout or 300.0
        self.debug(lambda: "httpcore client initialized successfully")

    async def close(self) -> None:
        """Close the connection pool and cleanup all resources."""
        if self.pool:
            self.debug(lambda: "Closing httpcore connection pool")
            await self.pool.aclose()
            self.pool = None
            self.debug(lambda: "httpcore connection pool closed")

    @staticmethod
    async def _tracked_stream(raw_stream, trace_data: BaseTraceData):
        """Yield chunks while tracking timing and byte counts for trace data."""
        awaiting_first_chunk = True
        async for chunk in raw_stream:
            chunk_ns = time.perf_counter_ns()
            chunk_len = len(chunk)
            trace_data.response_chunks_count += 1
            trace_data.response_bytes_total += chunk_len
            if awaiting_first_chunk:
                trace_data.response_receive_start_perf_ns = chunk_ns
                awaiting_first_chunk = False
            trace_data.response_receive_end_perf_ns = chunk_ns
            yield chunk

    async def _consume_sse_response(
        self,
        response: httpcore.Response,
        record: RequestRecord,
        *,
        trace_data: BaseTraceData,
        first_token_callback: "FirstTokenCallback | None",
    ) -> None:
        """Parse httpcore SSE stream into record.responses."""
        first_token_acquired = not first_token_callback
        async for message in AsyncSSEStreamReader(
            self._tracked_stream(response.aiter_stream(), trace_data)
        ):
            AsyncSSEStreamReader.inspect_message_for_error(message)
            record.responses.append(message)
            if not first_token_acquired:
                ttft_ns = message.perf_ns - record.start_perf_ns
                first_token_acquired = await first_token_callback(ttft_ns, message)
        record.end_perf_ns = time.perf_counter_ns()
        self.debug(lambda: f"Parsed {len(record.responses)} SSE messages")

    async def _consume_body_response(
        self,
        response: httpcore.Response,
        record: RequestRecord,
        *,
        content_type: str,
        trace_data: BaseTraceData,
    ) -> None:
        """Collect non-SSE response body into record.responses."""
        response_body = bytearray()
        async for chunk in self._tracked_stream(response.aiter_stream(), trace_data):
            response_body.extend(chunk)

        record.end_perf_ns = time.perf_counter_ns()
        is_binary = (
            content_type.startswith(("video/", "image/", "audio/"))
            or content_type == "application/octet-stream"
        )
        if is_binary:
            record.responses.append(
                BinaryResponse(
                    perf_ns=record.end_perf_ns,
                    content_type=content_type,
                    raw_bytes=bytes(response_body),
                )
            )
            self.debug(lambda: f"Binary response complete: {len(response_body)} bytes")
        else:
            response_text = response_body.decode("utf-8", errors="replace")
            record.responses.append(
                TextResponse(
                    perf_ns=record.end_perf_ns,
                    content_type=content_type,
                    text=response_text,
                )
            )
            self.debug(lambda: f"Response complete: {len(response_text)} bytes")

    async def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        *,
        data: bytes | None = None,
        first_token_callback: "FirstTokenCallback | None" = None,
        trace_data: BaseTraceData | None = None,
        **kwargs: Any,
    ) -> RequestRecord:
        """Execute HTTP requests with nanosecond-precision timing and error handling.

        Automatically detects and handles SSE streams. All exceptions are caught and
        converted to ErrorDetails in the returned RequestRecord.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL with scheme
            headers: Request headers dict
            data: Optional request body bytes
            first_token_callback: Optional callback fired on first SSE message with ttft_ns
            trace_data: Optional pre-created trace data (for cancellation scenarios)
            **kwargs: Additional arguments for future extension

        Returns:
            RequestRecord with status, timing data, responses, and optional error
        """
        self.debug(lambda: f"Sending {method} request to {url}")

        if trace_data is None:
            trace_data = BaseTraceData(trace_type="httpcore")
        record = RequestRecord(
            start_perf_ns=time.perf_counter_ns(),
            trace_data=trace_data,
        )

        try:
            is_sse_request = headers.get("Accept", "").startswith("text/event-stream")

            httpcore_headers = [
                (name.encode("utf-8"), value.encode("utf-8"))
                for name, value in headers.items()
            ]

            extensions: dict[str, Any] = {
                "timeout": {
                    "connect": self.timeout_seconds,
                    "read": self.timeout_seconds,
                    "write": self.timeout_seconds,
                    "pool": 60.0,
                }
            }

            async with self.pool.stream(
                method=method.encode("utf-8"),
                url=url.encode("utf-8"),
                headers=httpcore_headers,
                content=data,
                extensions=extensions,
            ) as response:
                record.status = response.status
                record.recv_start_perf_ns = time.perf_counter_ns()

                self.debug(
                    lambda: (
                        f"Response status: {record.status}, "
                        f"HTTP version: HTTP/{response.extensions.get('http_version', b'').decode()}"
                    )
                )

                if record.status != 200:
                    error_body = bytearray()
                    async for chunk in response.aiter_stream():
                        error_body.extend(chunk)

                    error_text = error_body.decode("utf-8", errors="replace")
                    self.debug(
                        lambda: f"HTTP error {record.status}: {error_text[:100]}"
                    )
                    record.error = ErrorDetails(
                        code=record.status,
                        type=f"HTTP {record.status}",
                        message=error_text or f"HTTP {record.status} error",
                    )
                    record.end_perf_ns = time.perf_counter_ns()
                    return record

                response_headers = {
                    name.decode("utf-8").lower(): value.decode("utf-8")
                    for name, value in response.headers
                }
                content_type = response_headers.get("content-type", "")

                if is_sse_request and content_type.startswith("text/event-stream"):
                    self.debug("Processing SSE stream")
                    await self._consume_sse_response(
                        response,
                        record,
                        trace_data=trace_data,
                        first_token_callback=first_token_callback,
                    )
                else:
                    self.debug("Processing regular response")
                    await self._consume_body_response(
                        response,
                        record,
                        content_type=content_type,
                        trace_data=trace_data,
                    )

                if not record.end_perf_ns:
                    record.end_perf_ns = time.perf_counter_ns()

        except (
            httpcore.ConnectTimeout,
            httpcore.ReadTimeout,
            httpcore.WriteTimeout,
            httpcore.PoolTimeout,
            httpcore.TimeoutException,
            httpcore.ConnectError,
            httpcore.RemoteProtocolError,
            httpcore.LocalProtocolError,
            httpcore.ProtocolError,
        ) as e:
            record.end_perf_ns = time.perf_counter_ns()
            record.error = self._classify_httpcore_error(e, url)
            self.error(f"{record.error.type}: {e!r}")
        except SSEResponseError as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Error in SSE response: {e!r}")
            record.error = ErrorDetails.from_exception(e)
        except asyncio.CancelledError:
            record.end_perf_ns = time.perf_counter_ns()
            record.cancellation_perf_ns = record.end_perf_ns
            record.error = ErrorDetails(
                type="RequestCancellationError",
                message="Request cancelled by external signal",
                code=499,
            )
            self.debug("Request cancelled by external signal")
            raise
        except Exception as e:  # noqa: BLE001 - per-request; attach ErrorDetails and return record
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Unexpected error in HTTP request: {e!r}")
            record.error = ErrorDetails.from_exception(e)

        return record

    def _classify_httpcore_error(self, exc: Exception, url: str) -> ErrorDetails:
        """Map an httpcore exception subclass to an ErrorDetails with a stable type label."""
        if isinstance(exc, httpcore.ConnectTimeout):
            return ErrorDetails(
                type="ConnectTimeout",
                message=f"Connection to {url} timed out after {self.timeout_seconds}s",
            )
        if isinstance(exc, httpcore.ReadTimeout):
            return ErrorDetails(
                type="ReadTimeout",
                message=f"Reading response from {url} timed out after {self.timeout_seconds}s",
            )
        if isinstance(exc, httpcore.WriteTimeout):
            return ErrorDetails(
                type="WriteTimeout",
                message=f"Sending request to {url} timed out after {self.timeout_seconds}s",
            )
        if isinstance(exc, httpcore.PoolTimeout):
            return ErrorDetails(
                type="PoolTimeout",
                message=(
                    f"No available connection in pool after 60s. "
                    f"Consider increasing AIPERF_HTTP_CONNECTION_LIMIT (current: {Environment.HTTP.CONNECTION_LIMIT})"
                ),
            )
        if isinstance(exc, httpcore.TimeoutException):
            return ErrorDetails(
                type="TimeoutError",
                message=f"Request to {url} timed out: {exc!r}",
            )
        if isinstance(exc, httpcore.ConnectError):
            return ErrorDetails(
                type="ConnectError",
                message=f"Failed to connect to {url}: {exc!r}",
            )
        if isinstance(exc, httpcore.RemoteProtocolError):
            return ErrorDetails(
                type="RemoteProtocolError",
                message=f"Server sent invalid HTTP/2 frames: {exc!r}",
            )
        if isinstance(exc, httpcore.LocalProtocolError):
            return ErrorDetails(
                type="LocalProtocolError",
                message=f"Client attempted invalid HTTP/2 operation: {exc!r}",
            )
        return ErrorDetails(
            type="ProtocolError", message=f"HTTP/2 protocol error: {exc!r}"
        )

    async def _request_with_cancellation(
        self,
        url: str,
        payload: bytes,
        headers: dict[str, str],
        cancel_after_ns: int,
        *,
        first_token_callback: "FirstTokenCallback | None" = None,
    ) -> RequestRecord:
        """Send POST request with cancellation after specified delay.

        The timer starts from request submission (conservative vs. aiohttp which
        waits for send-complete). This is safe but may cancel slightly earlier.

        When cancelled, returns a RequestRecord with cancellation_perf_ns set and
        error code 499.
        """
        start_perf_ns = time.perf_counter_ns()
        timeout_s = cancel_after_ns / 1e9

        # Create trace_data outside the task so it survives cancellation
        trace_data = BaseTraceData(trace_type="httpcore")

        request_task = asyncio.create_task(
            self._request(
                "POST",
                url,
                headers,
                data=payload,
                first_token_callback=first_token_callback,
                trace_data=trace_data,
            )
        )

        try:
            return await asyncio.wait_for(request_task, timeout=timeout_s)
        except asyncio.TimeoutError:
            request_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await request_task

            end_perf_ns = time.perf_counter_ns()
            self.debug(f"Request cancelled {timeout_s:.3f}s after submission")
            return RequestRecord(
                start_perf_ns=start_perf_ns,
                end_perf_ns=end_perf_ns,
                cancellation_perf_ns=end_perf_ns,
                trace_data=trace_data,
                error=ErrorDetails(
                    type="RequestCancellationError",
                    message=f"Request cancelled {timeout_s:.3f}s after submission",
                    code=499,
                ),
            )

    async def post_request(
        self,
        url: str,
        payload: bytes,
        headers: dict[str, str],
        *,
        cancel_after_ns: int | None = None,
        first_token_callback: "FirstTokenCallback | None" = None,
        **kwargs: Any,
    ) -> RequestRecord:
        """Send an HTTP POST request with optional SSE streaming support.

        Args:
            url: Target URL with scheme
            payload: Request body bytes
            headers: HTTP headers dict
            cancel_after_ns: If set, cancel the request this many nanoseconds after
                submission. Returns a record with cancellation_perf_ns set and error code 499.
            first_token_callback: Optional callback fired on first SSE message with ttft_ns
            **kwargs: Additional arguments passed to _request()

        Returns:
            RequestRecord with status, timing data, responses, and optional error
        """
        if cancel_after_ns is not None:
            return await self._request_with_cancellation(
                url,
                payload,
                headers,
                cancel_after_ns,
                first_token_callback=first_token_callback,
            )
        return await self._request(
            "POST",
            url,
            headers,
            data=payload,
            first_token_callback=first_token_callback,
            **kwargs,
        )

    async def get_request(
        self, url: str, headers: dict[str, str], **kwargs: Any
    ) -> RequestRecord:
        """Send an HTTP GET request.

        Args:
            url: Target URL with scheme
            headers: HTTP headers dict
            **kwargs: Additional arguments passed to _request()

        Returns:
            RequestRecord with status, timing data, responses, and optional error
        """
        return await self._request("GET", url, headers, **kwargs)

    @classmethod
    def create_ephemeral(cls, timeout: float | None = None) -> "HttpCoreClient":
        """Create a lightweight single-connection client for one-shot requests.

        Bypasses normal __init__ to avoid heavy pool creation. The pool uses
        max_connections=1 and keepalive_expiry=0 so the connection is closed
        after the response body is consumed.
        """
        instance = cls.__new__(cls)
        AIPerfLoggerMixin.__init__(instance)

        ssl_context = ssl.create_default_context()
        if not Environment.HTTP.SSL_VERIFY:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        instance.pool = httpcore.AsyncConnectionPool(
            http1=HttpCoreDefaults.HTTP1,
            http2=HttpCoreDefaults.HTTP2,
            max_connections=1,
            max_keepalive_connections=0,
            keepalive_expiry=0,
            retries=0,
            socket_options=SocketDefaults.build_socket_options(),
            ssl_context=ssl_context,
        )
        instance.timeout_seconds = timeout or 300.0
        return instance
