# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import socket
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

import aiohttp
from yarl import URL

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.environment import Environment
from aiperf.common.exceptions import ResponseIdleTimeoutError, SSEResponseError
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    AioHttpTraceData,
    BinaryResponse,
    ErrorDetails,
    RequestRecord,
    TextResponse,
)
from aiperf.common.monotonic_clock import process_clock
from aiperf.transports.aiohttp_trace import create_aiohttp_trace_config
from aiperf.transports.aws.eventstream import (
    EVENTSTREAM_CONTENT_TYPE,
    AwsEventStreamError,
    AwsEventStreamReader,
)
from aiperf.transports.http_defaults import AioHttpDefaults, SocketDefaults
from aiperf.transports.sse_utils import AsyncSSEStreamReader

if TYPE_CHECKING:
    from aiperf.transports.base_transports import FirstTokenCallback


def _expected_request_body_size(data: Any) -> int | None:
    """Return the byte length of an HTTP request body, or None if unknown.

    The aiohttp trace callback uses this to fire ``on_request_sent`` once
    ``bytes_sent`` reaches the expected total. For multipart bodies (image_edit,
    video_generation) the size is not directly len-able, so we serialize the
    FormData payload once and read its computed size.
    """
    if isinstance(data, bytes):
        return len(data)
    if isinstance(data, aiohttp.FormData):
        # TODO: compute size analytically from `data._fields` to avoid the
        # extra MultipartWriter materialization that session.request() will do.
        try:
            return data().size
        except (ValueError, TypeError, AttributeError):
            return None
    return None


VIDEO_DOWNLOAD_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class _PassthroughTimeout(Exception):
    """Marks a TimeoutError raised by the read itself, not by the idle bound.

    ``asyncio.timeout`` converts only its own timer's cancellation into
    ``TimeoutError``; a foreign one (e.g. the session's total, which aiohttp
    raises from inside the read) passes through unchanged and is
    indistinguishable by type. Wrapping the foreign one inside the window —
    so only the wrapper's own timer can surface as a bare ``TimeoutError``
    outside it — is what makes attribution exact regardless of when the
    foreign timeout lands relative to the idle window.
    """

    def __init__(self, original: TimeoutError) -> None:
        super().__init__(str(original))
        self.original = original


async def _await_with_idle_bound(
    awaitable: Awaitable[Any],
    idle_timeout_s: float,
    make_error: Callable[[], ResponseIdleTimeoutError],
) -> Any:
    """Await one HTTP read under the per-operation idle bound.

    With ``idle_timeout_s <= 0`` the awaitable runs unbounded and untouched.
    Otherwise it runs inside ``asyncio.timeout(idle_timeout_s)``; only a
    timeout produced by that wrapper's own timer is relabeled
    ``ResponseIdleTimeoutError`` (a ``TimeoutError`` the awaitable raises
    itself is wrapped in ``_PassthroughTimeout`` before it can reach the
    context, and re-raised with its original identity outside it). External
    cancellation is untouched: the context re-raises a foreign
    ``CancelledError`` unchanged, so credit cancellation still surfaces as
    ``RequestCancellationError``.
    """
    if idle_timeout_s <= 0:
        return await awaitable
    try:
        async with asyncio.timeout(idle_timeout_s):
            try:
                return await awaitable
            except TimeoutError as inner:
                raise _PassthroughTimeout(inner) from inner
    except _PassthroughTimeout as passthrough:
        raise passthrough.original from None
    except TimeoutError as idle:
        raise make_error() from idle


@contextlib.asynccontextmanager
async def _request_with_idle_bound(
    request_cm: Any,
    idle_timeout_s: float,
    make_error: Callable[[], ResponseIdleTimeoutError],
) -> AsyncIterator[aiohttp.ClientResponse]:
    """Enter the request context manager under the idle bound.

    Bounding only the entry covers the wait for response headers — for a
    non-streaming response that wait is the entire generation, which emits no
    bytes until completion. The exit stack forwards body-phase exceptions to
    the request CM's ``__aexit__``, so an aborted response still releases its
    connection instead of leaking a wedged socket into the pool.
    """
    async with contextlib.AsyncExitStack() as stack:
        response = await _await_with_idle_bound(
            stack.enter_async_context(request_cm), idle_timeout_s, make_error
        )
        yield response


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
        # Read by the worker's health report; see WorkerTaskStats.stream_chunks_received.
        self.stream_chunks_received = 0

    async def close(self) -> None:
        """Close the client."""
        if self.tcp_connector:
            await self.tcp_connector.close()
            self.tcp_connector = None

    async def _request(
        self,
        method: str,
        url: str | URL,
        headers: dict[str, str],
        *,
        data: bytes | aiohttp.FormData | None = None,
        on_request_sent: asyncio.Event | None = None,
        first_token_callback: "FirstTokenCallback | None" = None,
        trace_data: AioHttpTraceData | None = None,
        connector: aiohttp.TCPConnector | None = None,
        connector_owner: bool = False,
        capture_redirects: bool = False,
        trust_env: bool | None = None,
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
            capture_redirects: Capture download redirect headers without reading the body.
                Requires GET and explicit allow_redirects=False.
            trust_env: Override environment proxy and netrc use for this session.
            **kwargs: Additional arguments to pass to the request

        Returns:
            RequestRecord with the response data
        """
        if capture_redirects and (
            method != "GET" or kwargs.get("allow_redirects") is not False
        ):
            raise ValueError("Redirect capture requires GET with allow_redirects=False")
        self.debug(lambda: f"Sending {method} request to {url}")

        # Use provided trace_data or create new one
        if trace_data is None:
            trace_data = AioHttpTraceData()

        record: RequestRecord = RequestRecord(
            start_perf_ns=time.perf_counter_ns(),
            trace_data=trace_data,
        )

        # Create trace config for comprehensive timing.
        # The trace fires `on_request_sent` once `bytes_sent` reaches this size, so
        # the cancellation timer can start. Without it, multipart bodies wait until
        # the send-timeout safety net and surface as RequestSendTimeout.
        expected_request_body_size = _expected_request_body_size(data)
        collect_chunks = self.collect_trace_chunks
        trace_config = create_aiohttp_trace_config(
            record.trace_data,
            on_request_sent_event=on_request_sent,
            expected_request_body_size=expected_request_body_size,
            collect_chunks=collect_chunks,
        )

        try:
            # Per-read liveness bound for the whole response. The session's
            # total timeout defaults to 6h, so without this a response that
            # stops delivering bytes mid-flight (server-side wedge, runaway
            # generation, half-dead connection) hangs the request until the
            # run-level stall watchdog aborts the whole run. Opt-in (0 = off):
            # silence cannot tell a wedged response from a legitimately slow
            # one, and aborted requests are not retried.
            response_idle_timeout = Environment.HTTP.RESPONSE_IDLE_TIMEOUT

            def _idle_error() -> ResponseIdleTimeoutError:
                return ResponseIdleTimeoutError(
                    idle_timeout_sec=response_idle_timeout,
                    chunks_received=record.trace_data.response_chunks_count,
                    bytes_received=record.trace_data.response_bytes_total,
                )

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
                trust_env=AioHttpDefaults.TRUST_ENV if trust_env is None else trust_env,
            ) as session:
                # Session setup moved start_perf_ns past record construction; re-derive
                # timestamp_ns from it.
                record.start_perf_ns = time.perf_counter_ns()
                record.timestamp_ns = process_clock().wall_ns_at(record.start_perf_ns)
                # The idle bound on entry covers the wait for response headers,
                # which for a non-streaming response is the whole generation.
                async with _request_with_idle_bound(
                    session.request(method, url, data=data, headers=headers, **kwargs),
                    response_idle_timeout,
                    _idle_error,
                ) as response:
                    record.status = response.status

                    if (
                        capture_redirects
                        and response.status in VIDEO_DOWNLOAD_REDIRECT_STATUSES
                    ):
                        record._redirect_locations = tuple(
                            response.headers.getall("Location", ())
                        )
                        record.end_perf_ns = time.perf_counter_ns()
                        # An unfinished redirect body must not delay the next hop.
                        response.close()
                        return record

                    # Treat the full 2xx range as success so async job APIs can
                    # return accepted/created responses without being rejected.
                    if response.status < 200 or response.status >= 300:
                        error_text = await self._read_text_with_idle_bound(
                            response,
                            idle_timeout_s=response_idle_timeout,
                            make_error=_idle_error,
                            trace_data=record.trace_data,
                            collect_chunks=collect_chunks,
                        )
                        record.end_perf_ns = time.perf_counter_ns()
                        record.error = ErrorDetails(
                            code=response.status,
                            type=response.reason,
                            message=error_text,
                        )
                        return record

                    record.recv_start_perf_ns = time.perf_counter_ns()

                    is_eventstream = response.content_type == EVENTSTREAM_CONTENT_TYPE
                    if method == "POST" and (
                        response.content_type == "text/event-stream" or is_eventstream
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
                            _trace = record.trace_data
                            _collect = collect_chunks
                            _chunks_append = (
                                _trace.response_chunks.append if _collect else None
                            )
                            awaiting_first_chunk = True
                            chunk_iter = response.content.iter_any()

                            async def _next_chunk() -> bytes | None:
                                """Next raw chunk, or None at end of stream.

                                Raises ResponseIdleTimeoutError when the stream
                                delivers no bytes within the configured idle
                                bound (0 disables the bound).
                                """
                                try:
                                    return await _await_with_idle_bound(
                                        anext(chunk_iter),
                                        response_idle_timeout,
                                        _idle_error,
                                    )
                                except StopAsyncIteration:
                                    return None

                            while True:
                                chunk = await _next_chunk()
                                if chunk is None:
                                    break
                                chunk_ns = time.perf_counter_ns()
                                chunk_len = len(chunk)
                                self.stream_chunks_received += 1
                                _trace.response_chunks_count += 1
                                _trace.response_bytes_total += chunk_len
                                if _chunks_append is not None:
                                    _chunks_append((chunk_ns, chunk_len))
                                if awaiting_first_chunk:
                                    _trace.response_receive_start_perf_ns = chunk_ns
                                    awaiting_first_chunk = False
                                _trace.response_receive_end_perf_ns = chunk_ns
                                yield chunk

                        # AWS services stream over binary eventstream framing
                        # rather than raw SSE. AwsEventStreamReader mirrors
                        # AsyncSSEStreamReader's interface exactly (__aiter__
                        # yielding a message implementing InferenceServerResponse,
                        # inspect_message_for_error), so the reader class is the
                        # only thing that varies below -- downstream code relies
                        # on that shared protocol, not on SSEMessage specifically.
                        # Selection is by content type alone, never by service,
                        # which is what lets Bedrock reuse this unchanged.
                        reader_cls = (
                            AwsEventStreamReader
                            if is_eventstream
                            else AsyncSSEStreamReader
                        )
                        sse_messages = reader_cls(tracked_content_stream())

                        # Separate code paths for performance: avoid callback checks
                        # when no callback is registered
                        if first_token_callback:
                            first_token_acquired = False
                            async for message in sse_messages:
                                reader_cls.inspect_message_for_error(message)
                                record.responses.append(message)
                                # Fire callback until it returns True (meaningful content found)
                                if not first_token_acquired:
                                    ttft_ns = message.perf_ns - record.start_perf_ns
                                    first_token_acquired = await first_token_callback(
                                        ttft_ns, message
                                    )
                        else:
                            # Fast path: no callback, just collect responses
                            async for message in sse_messages:
                                reader_cls.inspect_message_for_error(message)
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
                            raw_bytes = await self._read_body_with_idle_bound(
                                response,
                                idle_timeout_s=response_idle_timeout,
                                make_error=_idle_error,
                                trace_data=record.trace_data,
                                collect_chunks=collect_chunks,
                            )
                            record.end_perf_ns = time.perf_counter_ns()
                            record.responses.append(
                                BinaryResponse(
                                    perf_ns=record.end_perf_ns,
                                    content_type=content_type,
                                    raw_bytes=raw_bytes,
                                )
                            )
                        else:
                            raw_response = await self._read_text_with_idle_bound(
                                response,
                                idle_timeout_s=response_idle_timeout,
                                make_error=_idle_error,
                                trace_data=record.trace_data,
                                collect_chunks=collect_chunks,
                            )
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
        except (SSEResponseError, AwsEventStreamError) as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Error in streaming response: {e!r}")
            record.error = ErrorDetails.from_exception(e)
            if isinstance(e, AwsEventStreamError) and e.exception_type:
                # Preserve AWS's own label (ModelStreamError,
                # InternalStreamFailure, ...) so the error table groups
                # distinct server-side failures separately.
                record.error.type = e.exception_type
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

    async def _read_body_with_idle_bound(
        self,
        response: aiohttp.ClientResponse,
        *,
        idle_timeout_s: float,
        make_error: Callable[[], ResponseIdleTimeoutError],
        trace_data: AioHttpTraceData,
        collect_chunks: bool,
    ) -> bytes:
        """Drain a non-streamed body with per-read silence semantics.

        A single deadline around ``response.read()`` would abort a slow but
        continuously arriving body; bounding each ``iter_any`` read keeps the
        documented "no bytes for N seconds" meaning — the window resets on
        every received byte. With the bound disabled the stock read runs
        untouched.

        ``iter_any()`` bypasses aiohttp's trace callback system, so the
        received chunk/byte accounting (consumed by http_trace_metrics) and
        the stream-activity counter (the run-level watchdog's liveness
        signal) are mirrored manually here, exactly as the streaming path
        does. When the bound is disabled the stock ``read()`` fires the real
        trace callbacks instead, unchanged.
        """
        if idle_timeout_s <= 0:
            return await response.read()
        parts: list[bytes] = []
        append_chunk = trace_data.response_chunks.append if collect_chunks else None
        chunk_iter = response.content.iter_any()
        while True:
            try:
                chunk = await _await_with_idle_bound(
                    anext(chunk_iter), idle_timeout_s, make_error
                )
            except StopAsyncIteration:
                break
            chunk_ns = time.perf_counter_ns()
            self.stream_chunks_received += 1
            trace_data.response_chunks_count += 1
            trace_data.response_bytes_total += len(chunk)
            if append_chunk is not None:
                append_chunk((chunk_ns, len(chunk)))
            if trace_data.response_receive_start_perf_ns is None:
                trace_data.response_receive_start_perf_ns = chunk_ns
            trace_data.response_receive_end_perf_ns = chunk_ns
            parts.append(chunk)
        return b"".join(parts)

    async def _read_text_with_idle_bound(
        self,
        response: aiohttp.ClientResponse,
        *,
        idle_timeout_s: float,
        make_error: Callable[[], ResponseIdleTimeoutError],
        trace_data: AioHttpTraceData,
        collect_chunks: bool,
    ) -> str:
        """``response.text()`` with per-read silence semantics when armed.

        The drained body never lands in ``response._body``, so
        ``get_encoding()`` cannot sniff it and raises RuntimeError for a
        charset-less non-JSON body: use its charset/content-type fast paths
        (which need no body) and fall back to UTF-8, the encoding chardet
        resolves for any ASCII/UTF-8 payload.
        """
        if idle_timeout_s <= 0:
            return await response.text()
        body = await self._read_body_with_idle_bound(
            response,
            idle_timeout_s=idle_timeout_s,
            make_error=make_error,
            trace_data=trace_data,
            collect_chunks=collect_chunks,
        )
        try:
            encoding = response.get_encoding()
        except RuntimeError:
            encoding = "utf-8"
        return body.decode(encoding, errors="strict")

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

        Args:
            url: Target URL
            payload: Request body as bytes or FormData for multipart
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
            **kwargs,
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
        **kwargs: Any,
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
                **kwargs,
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
        except TimeoutError:
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
        except TimeoutError:
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
        self, url: str | URL, headers: dict[str, str], **kwargs: Any
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
