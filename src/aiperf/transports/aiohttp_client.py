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
        collect_trace_chunks: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the AioHttpClient."""
        super().__init__(**kwargs)
        self.tcp_connector = create_tcp_connector(**tcp_kwargs or {})
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.collect_trace_chunks = collect_trace_chunks

    async def close(self) -> None:
        """Close the client."""
        if self.tcp_connector:
            await self.tcp_connector.close()
            self.tcp_connector = None

    async def _consume_sse_response(
        self,
        response: aiohttp.ClientResponse,
        record: RequestRecord,
        *,
        collect_chunks: bool,
        first_token_callback: "FirstTokenCallback | None",
    ) -> None:
        """Parse an SSE content stream into record.responses with chunk timing."""

        # iter_any() yields raw bytes as they arrive; manually track chunks because
        # it bypasses aiohttp's trace callback system.
        async def tracked_content_stream():
            _trace = record.trace_data
            _chunks_append = _trace.response_chunks.append if collect_chunks else None
            awaiting_first_chunk = True
            async for chunk in response.content.iter_any():
                chunk_ns = time.perf_counter_ns()
                chunk_len = len(chunk)
                _trace.response_chunks_count += 1
                _trace.response_bytes_total += chunk_len
                if _chunks_append is not None:
                    _chunks_append((chunk_ns, chunk_len))
                if awaiting_first_chunk:
                    _trace.response_receive_start_perf_ns = chunk_ns
                    awaiting_first_chunk = False
                _trace.response_receive_end_perf_ns = chunk_ns
                yield chunk

        if first_token_callback:
            first_token_acquired = False
            async for message in AsyncSSEStreamReader(tracked_content_stream()):
                AsyncSSEStreamReader.inspect_message_for_error(message)
                record.responses.append(message)
                if not first_token_acquired:
                    ttft_ns = message.perf_ns - record.start_perf_ns
                    first_token_acquired = await first_token_callback(ttft_ns, message)
        else:
            # Fast path: no callback, just collect responses
            async for message in AsyncSSEStreamReader(tracked_content_stream()):
                AsyncSSEStreamReader.inspect_message_for_error(message)
                record.responses.append(message)
        record.end_perf_ns = time.perf_counter_ns()

    async def _consume_non_sse_response(
        self,
        response: aiohttp.ClientResponse,
        record: RequestRecord,
    ) -> None:
        """Read a non-SSE response (JSON/text or binary) into record.responses."""
        response_start_ns = time.perf_counter_ns()

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
            record.trace_data.response_receive_start_perf_ns = response_start_ns
        # Note: response.text()/read() should trigger aiohttp trace callbacks,
        # but we set response_receive_end_perf_ns explicitly for consistency
        record.trace_data.response_receive_end_perf_ns = record.end_perf_ns

    def _build_session(
        self,
        headers: dict[str, str],
        *,
        connector: aiohttp.TCPConnector | None,
        connector_owner: bool,
        trace_config: aiohttp.TraceConfig,
    ) -> aiohttp.ClientSession:
        """Create a ClientSession configured for a single request."""
        return aiohttp.ClientSession(
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
        )

    async def _dispatch_response(
        self,
        response: aiohttp.ClientResponse,
        record: RequestRecord,
        *,
        method: str,
        collect_chunks: bool,
        first_token_callback: "FirstTokenCallback | None",
    ) -> None:
        """Populate record from response, handling error status and SSE vs non-SSE bodies."""
        record.status = response.status

        # Treat the full 2xx range as success so async job APIs can
        # return accepted/created responses without being rejected.
        if response.status < 200 or response.status >= 300:
            error_text = await response.text()
            record.error = ErrorDetails(
                code=response.status,
                type=response.reason,
                message=error_text,
            )
            return

        record.recv_start_perf_ns = time.perf_counter_ns()

        if method == "POST" and response.content_type == "text/event-stream":
            await self._consume_sse_response(
                response,
                record,
                collect_chunks=collect_chunks,
                first_token_callback=first_token_callback,
            )
        else:
            await self._consume_non_sse_response(response, record)

    async def _execute_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        record: RequestRecord,
        *,
        data: bytes | aiohttp.FormData | None,
        connector: aiohttp.TCPConnector | None,
        connector_owner: bool,
        trace_config: aiohttp.TraceConfig,
        collect_chunks: bool,
        first_token_callback: "FirstTokenCallback | None",
        **kwargs: Any,
    ) -> None:
        """Open a session, issue the request, and populate record from the response."""
        async with self._build_session(
            headers,
            connector=connector,
            connector_owner=connector_owner,
            trace_config=trace_config,
        ) as session:
            record.start_perf_ns = time.perf_counter_ns()
            async with session.request(
                method, url, data=data, headers=headers, **kwargs
            ) as response:
                await self._dispatch_response(
                    response,
                    record,
                    method=method,
                    collect_chunks=collect_chunks,
                    first_token_callback=first_token_callback,
                )
                if record.error is None:
                    self.debug(
                        lambda: (
                            f"{method} request to {url} completed in {(record.end_perf_ns - record.start_perf_ns) / NANOS_PER_SECOND} seconds"
                        )
                    )

    async def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        *,
        data: bytes | aiohttp.FormData | None = None,
        on_request_sent: asyncio.Event | None = None,
        first_token_callback: "FirstTokenCallback | None" = None,
        trace_data: AioHttpTraceData | None = None,
        connector: aiohttp.TCPConnector | None = None,
        connector_owner: bool = False,
        **kwargs: Any,
    ) -> RequestRecord:
        """Generic request method that handles common logic for all HTTP methods.

        trace_data may be supplied so callers (e.g. the cancellation wrapper) can
        observe trace state even if the task is cancelled before returning.
        connector_owner=True lets the session close the per-request connector;
        False (default) keeps it in the shared pool.
        """
        self.debug(lambda: f"Sending {method} request to {url}")

        if trace_data is None:
            trace_data = AioHttpTraceData()
        record: RequestRecord = RequestRecord(
            start_perf_ns=time.perf_counter_ns(),
            trace_data=trace_data,
        )
        collect_chunks = self.collect_trace_chunks
        trace_config = create_aiohttp_trace_config(
            record.trace_data,
            on_request_sent_event=on_request_sent,
            expected_request_body_size=len(data) if isinstance(data, bytes) else None,
            collect_chunks=collect_chunks,
        )

        try:
            await self._execute_request(
                method,
                url,
                headers,
                record,
                data=data,
                connector=connector,
                connector_owner=connector_owner,
                trace_config=trace_config,
                collect_chunks=collect_chunks,
                first_token_callback=first_token_callback,
                **kwargs,
            )
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
        except (SSEResponseError, Exception) as e:  # noqa: BLE001 - per-request; attach ErrorDetails and return record
            record.end_perf_ns = time.perf_counter_ns()
            label = (
                "SSE response" if isinstance(e, SSEResponseError) else "aiohttp request"
            )
            self.error(f"Error in {label}: {e!r}")
            record.error = ErrorDetails.from_exception(e)

        return record

    async def post_request(
        self,
        url: str,
        payload: bytes | aiohttp.FormData,
        headers: dict[str, str],
        *,
        cancel_after_ns: int | None = None,
        first_token_callback: "FirstTokenCallback | None" = None,
        connector: aiohttp.TCPConnector | None = None,
        connector_owner: bool = False,
        **kwargs: Any,
    ) -> RequestRecord:
        """Send a POST request to the specified URL.

        If cancel_after_ns is set, cancel the request that many ns after it is
        fully sent. The request is always sent before cancellation.
        Returns a RequestRecord with response data, timing, and any errors.
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
        payload: bytes | aiohttp.FormData,
        headers: dict[str, str],
        cancel_after_ns: int,
        *,
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

        send_timeout_result = await self._await_request_sent(
            request_task, request_sent, start_perf_ns, trace_data
        )
        if send_timeout_result is not None:
            return send_timeout_result

        # Check if request already completed (e.g., with connection error)
        if request_task.done():
            return await request_task

        try:
            return await asyncio.wait_for(request_task, timeout=timeout_s)
        except asyncio.TimeoutError:
            return await self._cancel_after_timeout(
                request_task, start_perf_ns, timeout_s, trace_data
            )

    async def _await_request_sent(
        self,
        request_task: "asyncio.Task[RequestRecord]",
        request_sent: asyncio.Event,
        start_perf_ns: int,
        trace_data: AioHttpTraceData,
    ) -> RequestRecord | None:
        """Wait for the request to be fully sent; return an error record on timeout, else None."""
        send_timeout = (
            self.timeout.total or Environment.HTTP.REQUEST_CANCELLATION_SEND_TIMEOUT
        )
        try:
            await asyncio.wait_for(request_sent.wait(), timeout=send_timeout)
            return None
        except asyncio.TimeoutError:
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

    async def _cancel_after_timeout(
        self,
        request_task: "asyncio.Task[RequestRecord]",
        start_perf_ns: int,
        timeout_s: float,
        trace_data: AioHttpTraceData,
    ) -> RequestRecord:
        """Cancel the in-flight request task and return a cancellation RequestRecord."""
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
