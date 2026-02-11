# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import socket
import time
from typing import TYPE_CHECKING, Any

import aiohttp

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.environment import Environment
from aiperf.common.exceptions import SSEResponseError
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    AioHttpTraceData,
    BinaryResponse,
    ErrorDetails,
    RequestRecord,
    TextResponse,
)
from aiperf.transports.aiohttp_trace import create_aiohttp_trace_config
from aiperf.transports.http_defaults import AioHttpDefaults, SocketDefaults
from aiperf.transports.sse_utils import AsyncSSEStreamReader

if TYPE_CHECKING:
    from aiperf.transports.base_transports import FirstTokenCallback


class AioHttpClient(AIPerfLoggerMixin):
    """A high-performance HTTP client for communicating with HTTP based REST APIs using aiohttp.

    This class is optimized for maximum performance and accurate timing measurements,
    making it ideal for benchmarking scenarios.
    """

    def __init__(
        self,
        timeout: float | None = None,
        tcp_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the AioHttpClient."""
        super().__init__(**kwargs)
        self.tcp_connector = create_tcp_connector(**tcp_kwargs or {})
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def close(self) -> None:
        """Close the client."""
        if self.tcp_connector:
            await self.tcp_connector.close()
            self.tcp_connector = None

    async def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: str | None = None,
        on_request_sent: asyncio.Event | None = None,
        first_token_callback: "FirstTokenCallback | None" = None,
        trace_data: AioHttpTraceData | None = None,
        connector: aiohttp.TCPConnector | None = None,
        connector_owner: bool = False,
        **kwargs: Any,
    ) -> RequestRecord:
        """Generic request method that handles common logic for all HTTP methods.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: The URL to send the request to
            headers: Request headers
            data: Request payload (for POST, PUT, etc.)
            on_request_sent: Optional event to set when the full request is sent
            first_token_callback: Optional callback fired on first SSE message with ttft_ns
            trace_data: Optional trace data to populate (for cancellation scenarios)
            connector: Optional TCP connector to use instead of the shared pool.
                If None, uses self.tcp_connector (shared pool).
            connector_owner: If True, the session will close the connector when done.
                Use True for per-request connections that should be closed after use.
            **kwargs: Additional arguments to pass to the request

        Returns:
            RequestRecord with the response data
        """
        self.debug(lambda: f"Sending {method} request to {url}")

        # Use provided trace_data or create new one
        if trace_data is None:
            trace_data = AioHttpTraceData()

        record: RequestRecord = RequestRecord(
            start_perf_ns=time.perf_counter_ns(),
            trace_data=trace_data,
        )

        # Create trace config for comprehensive timing
        # Pass expected body size for chunk-based completion detection
        expected_request_body_size = len(data.encode("utf-8")) if data else None
        trace_config = create_aiohttp_trace_config(
            record.trace_data,
            on_request_sent_event=on_request_sent,
            expected_request_body_size=expected_request_body_size,
        )

        try:
            # Make raw HTTP request with precise timing using aiohttp
            # Create a new session for each request with unique trace config
            # connector_owner controls whether session closes the connector:
            # - False (default): connector is shared/pooled, don't close it
            # - True: connector is owned by this request, close when done
            async with aiohttp.ClientSession(
                connector=connector or self.tcp_connector,
                timeout=self.timeout,
                headers=headers,
                skip_auto_headers=[
                    *list(headers.keys()),
                    "User-Agent",
                    "Accept-Encoding",
                ],
                connector_owner=connector_owner,
                trace_configs=[trace_config],
                trust_env=AioHttpDefaults.TRUST_ENV,
            ) as session:
                record.start_perf_ns = time.perf_counter_ns()
                async with session.request(
                    method, url, data=data, headers=headers, **kwargs
                ) as response:
                    record.status = response.status

                    # Check for HTTP errors
                    if response.status != 200:
                        error_text = await response.text()
                        record.error = ErrorDetails(
                            code=response.status,
                            type=response.reason,
                            message=error_text,
                        )
                        return record

                    record.recv_start_perf_ns = time.perf_counter_ns()

                    if (
                        method == "POST"
                        and response.content_type == "text/event-stream"
                    ):
                        # Parse SSE stream with optimal performance
                        # Wrap the content stream to track chunks for trace data
                        async def tracked_content_stream():
                            """Wrapper that tracks chunk timing while yielding chunks for SSE parsing."""
                            # iter_any() yields raw bytes immediately as they arrive from the network,
                            # unlike default iteration which buffers until newlines. Critical for
                            # accurate chunk timing measurements.
                            #
                            # Note: We manually track chunks here because iter_any() bypasses aiohttp's
                            # trace callback system. We also set response_receive_start/end_perf_ns
                            # since on_response_chunk_received won't be called.
                            awaiting_first_chunk = True
                            async for chunk in response.content.iter_any():
                                chunk_ns = time.perf_counter_ns()
                                record.trace_data.response_chunks.append(
                                    (chunk_ns, len(chunk))
                                )
                                if awaiting_first_chunk:
                                    record.trace_data.response_receive_start_perf_ns = (
                                        chunk_ns
                                    )
                                    awaiting_first_chunk = False
                                record.trace_data.response_receive_end_perf_ns = (
                                    chunk_ns
                                )
                                yield chunk

                        # Separate code paths for performance: avoid callback checks
                        # when no callback is registered
                        if first_token_callback:
                            first_token_acquired = False
                            async for message in AsyncSSEStreamReader(
                                tracked_content_stream()
                            ):
                                AsyncSSEStreamReader.inspect_message_for_error(message)
                                record.responses.append(message)
                                # Fire callback until it returns True (meaningful content found)
                                if not first_token_acquired:
                                    ttft_ns = message.perf_ns - record.start_perf_ns
                                    first_token_acquired = await first_token_callback(
                                        ttft_ns, message
                                    )
                        else:
                            # Fast path: no callback, just collect responses
                            async for message in AsyncSSEStreamReader(
                                tracked_content_stream()
                            ):
                                AsyncSSEStreamReader.inspect_message_for_error(message)
                                record.responses.append(message)
                        record.end_perf_ns = time.perf_counter_ns()
                    else:
                        # Non-SSE response (e.g., JSON or binary)
                        response_start_ns = time.perf_counter_ns()

                        # Check if content type is binary (video, image, audio, octet-stream)
                        content_type = response.content_type or ""
                        is_binary = (
                            content_type.startswith("video/")
                            or content_type.startswith("image/")
                            or content_type.startswith("audio/")
                            or content_type == "application/octet-stream"
                        )

                        if is_binary:
                            raw_bytes = await response.read()
                            record.end_perf_ns = time.perf_counter_ns()
                            record.responses.append(
                                BinaryResponse(
                                    perf_ns=record.end_perf_ns,
                                    content_type=content_type,
                                    raw_bytes=raw_bytes,
                                )
                            )
                        else:
                            raw_response = await response.text()
                            record.end_perf_ns = time.perf_counter_ns()
                            record.responses.append(
                                TextResponse(
                                    perf_ns=record.end_perf_ns,
                                    content_type=content_type,
                                    text=raw_response,
                                )
                            )

                        if record.trace_data.response_receive_start_perf_ns is None:
                            record.trace_data.response_receive_start_perf_ns = (
                                response_start_ns
                            )
                        # Note: response.text()/read() should trigger aiohttp trace callbacks,
                        # but we set response_receive_end_perf_ns explicitly for consistency
                        record.trace_data.response_receive_end_perf_ns = (
                            record.end_perf_ns
                        )

                    self.debug(
                        lambda: (
                            f"{method} request to {url} completed in {(record.end_perf_ns - record.start_perf_ns) / NANOS_PER_SECOND} seconds"
                        )
                    )
        except SSEResponseError as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Error in SSE response: {e!r}")
            record.error = ErrorDetails.from_exception(e)
        except asyncio.CancelledError:
            # Task was cancelled externally (e.g., credit cancellation from router)
            # Record the cancellation and re-raise to allow proper cleanup
            record.end_perf_ns = time.perf_counter_ns()
            record.cancellation_perf_ns = record.end_perf_ns
            record.error = ErrorDetails(
                type="RequestCancellationError",
                message="Request cancelled by external signal",
                code=499,  # Client Closed Request
            )
            self.debug("Request cancelled by external signal")
            raise
        except Exception as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Error in aiohttp request: {e!r}")
            record.error = ErrorDetails.from_exception(e)

        return record

    async def post_request(
        self,
        url: str,
        payload: str,
        headers: dict[str, str],
        *,
        cancel_after_ns: int | None = None,
        first_token_callback: "FirstTokenCallback | None" = None,
        connector: aiohttp.TCPConnector | None = None,
        connector_owner: bool = False,
        **kwargs: Any,
    ) -> RequestRecord:
        """Send a POST request to the specified URL.

        Args:
            url: Target URL
            payload: Request body as string
            headers: Request headers
            cancel_after_ns: If set, cancel the request this many nanoseconds after
                it's fully sent. The request is always sent before cancellation.
            first_token_callback: Optional callback fired on first SSE message with ttft_ns
            connector: Optional TCP connector to use instead of the shared pool.
            connector_owner: If True, the session will close the connector when done.
            **kwargs: Additional arguments passed to aiohttp

        Returns:
            RequestRecord with response data, timing, and any errors
        """
        if cancel_after_ns is None:
            return await self._request(
                "POST",
                url,
                headers,
                data=payload,
                first_token_callback=first_token_callback,
                connector=connector,
                connector_owner=connector_owner,
                **kwargs,
            )
        return await self._request_with_cancellation(
            url,
            payload,
            headers,
            cancel_after_ns,
            first_token_callback=first_token_callback,
            connector=connector,
            connector_owner=connector_owner,
        )

    async def _request_with_cancellation(
        self,
        url: str,
        payload: str,
        headers: dict[str, str],
        cancel_after_ns: int,
        first_token_callback: "FirstTokenCallback | None" = None,
        connector: aiohttp.TCPConnector | None = None,
        connector_owner: bool = False,
    ) -> RequestRecord:
        """Send POST request with cancellation after specified delay.

        Wraps _request with cancellation logic. The timer starts when the full
        request (headers + body) is written to the socket.

        When cancelled, the task is cancelled and aiohttp's response context manager
        exit handler will close the connection (not return it to pool) since the
        response was not fully consumed. This is aiohttp's default behavior for
        partial/dirty responses.
        """
        start_perf_ns = time.perf_counter_ns()
        timeout_s = cancel_after_ns / NANOS_PER_SECOND

        # Track when request is sent via trace callback
        request_sent = asyncio.Event()

        # Create trace data outside the task so we can access it after cancellation
        trace_data = AioHttpTraceData()

        request_task = asyncio.create_task(
            self._request(
                "POST",
                url,
                headers,
                data=payload,
                on_request_sent=request_sent,
                first_token_callback=first_token_callback,
                trace_data=trace_data,
                connector=connector,
                connector_owner=connector_owner,
            )
        )

        # Wait for request to be sent, then apply cancellation timeout.
        # Use wait_for with a safety net timeout - the request_task should complete
        # (with error) or set the event on exception, but if something goes wrong
        # we don't want to hang forever.
        send_timeout = (
            self.timeout.total or Environment.HTTP.REQUEST_CANCELLATION_SEND_TIMEOUT
        )
        try:
            await asyncio.wait_for(request_sent.wait(), timeout=send_timeout)
        except asyncio.TimeoutError:
            # Request never got sent - cancel and return error
            request_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await request_task
            end_perf_ns = time.perf_counter_ns()
            return RequestRecord(
                start_perf_ns=start_perf_ns,
                end_perf_ns=end_perf_ns,
                trace_data=trace_data,
                error=ErrorDetails(
                    type="RequestSendTimeout",
                    message="Timed out waiting for request to be sent",
                    code=0,
                ),
            )

        # Check if request already completed (e.g., with connection error)
        if request_task.done():
            return await request_task

        try:
            return await asyncio.wait_for(request_task, timeout=timeout_s)
        except asyncio.TimeoutError:
            request_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await request_task

            end_perf_ns = time.perf_counter_ns()
            self.debug(f"Request cancelled {timeout_s:.3f}s after being sent")
            return RequestRecord(
                start_perf_ns=start_perf_ns,
                end_perf_ns=end_perf_ns,
                cancellation_perf_ns=end_perf_ns,
                trace_data=trace_data,
                error=ErrorDetails(
                    type="RequestCancellationError",
                    message=f"Request cancelled {timeout_s:.3f}s after being sent",
                    code=499,  # Client Closed Request
                ),
            )

    async def get_request(
        self, url: str, headers: dict[str, str], **kwargs: Any
    ) -> RequestRecord:
        """Send a GET request to the specified URL with the given headers.

        The response will be parsed into a TextResponse object.
        """
        return await self._request("GET", url, headers, **kwargs)


def create_tcp_connector(**kwargs) -> aiohttp.TCPConnector:
    """Create a new connector with the given configuration."""

    def socket_factory(addr_info):
        """Custom socket factory optimized for SSE streaming performance."""
        family, sock_type, proto, _, _ = addr_info
        sock = socket.socket(family=family, type=sock_type, proto=proto)
        SocketDefaults.apply_to_socket(sock)
        return sock

    default_kwargs: dict[str, Any] = AioHttpDefaults.get_default_kwargs()
    default_kwargs["socket_factory"] = socket_factory
    default_kwargs.update(kwargs)

    return aiohttp.TCPConnector(
        **default_kwargs,
    )
