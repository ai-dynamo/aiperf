# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic gRPC transport with pluggable serializers.

Protocol-agnostic: all proto knowledge is loaded dynamically from the
endpoint's ``grpc`` metadata in plugins.yaml. No proto imports here.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import grpc
import orjson

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import ConnectionReuseStrategy
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ErrorDetails, RequestInfo, RequestRecord, TextResponse
from aiperf.transports.base_transports import BaseTransport, FirstTokenCallback
from aiperf.transports.grpc.grpc_client import GenericGrpcClient, GrpcStreamCall
from aiperf.transports.grpc.status_mapping import grpc_status_to_http
from aiperf.transports.grpc.stream_chunk import StreamChunk
from aiperf.transports.grpc.trace_data import GrpcTraceData

# Max time to wait for gRPC channel to become ready before considering the send failed.
_CHANNEL_READY_TIMEOUT_S = 30.0


def _metadata_to_dict(
    metadata: Any,
) -> dict[str, str] | None:
    """Convert gRPC metadata (sequence of key-value tuples) to a string dict.

    Handles both request metadata (str values) and trailing metadata
    (which may contain bytes values for binary metadata keys ending in ``-bin``).

    Args:
        metadata: gRPC metadata as a sequence of (key, value) tuples, or None.

    Returns:
        Dict of string key-value pairs, or None if metadata is empty.
    """
    if not metadata:
        return None
    return {
        k: v if isinstance(v, str) else v.decode("utf-8", errors="replace")
        for k, v in metadata
    } or None


@runtime_checkable
class GrpcSerializerProtocol(Protocol):
    """Interface that gRPC serializer classes must implement."""

    def serialize_request(
        self, payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes: ...

    def deserialize_response(self, data: bytes) -> tuple[dict[str, Any], int]: ...

    def deserialize_stream_response(self, data: bytes) -> StreamChunk: ...


class GrpcChannelLeaseManager(AIPerfLoggerMixin):
    """Manages gRPC channel leases for sticky-user-sessions strategy.

    Each user session (identified by x_correlation_id) gets a dedicated
    GenericGrpcClient that persists across all turns. The client is closed
    when the final turn completes, enabling sticky load balancing where all
    turns of a user session hit the same backend server.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._leases: dict[str, GenericGrpcClient] = {}

    def get_or_create(
        self, session_id: str, factory: Callable[[], GenericGrpcClient]
    ) -> GenericGrpcClient:
        """Get or create a gRPC client for a user session.

        Args:
            session_id: Unique identifier for the user session (x_correlation_id).
            factory: Callable that creates a new GenericGrpcClient if needed.

        Returns:
            GenericGrpcClient dedicated to this user session.
        """
        if session_id not in self._leases:
            self._leases[session_id] = factory()
            self.debug(lambda: f"Created gRPC channel lease for session {session_id}")
        return self._leases[session_id]

    async def release_lease(self, session_id: str) -> None:
        """Release and close the gRPC client for a session.

        Args:
            session_id: Unique identifier for the session (x_correlation_id).
        """
        if session_id in self._leases:
            client = self._leases.pop(session_id)
            await client.close()
            self.debug(lambda: f"Released gRPC channel lease for session {session_id}")

    async def close_all(self) -> None:
        """Close all active channel leases."""
        leases = list(self._leases.values())
        self._leases.clear()
        for lease in leases:
            await lease.close()


class GrpcTransport(BaseTransport):
    """Generic gRPC transport with pluggable serializers.

    Uses grpc.aio for async gRPC with HTTP/2 multiplexing.
    Supports insecure (grpc://) and TLS (grpcs://) channels.
    Supports unary and streaming RPCs for any proto schema.

    The serializer class and gRPC method paths are loaded dynamically from
    the endpoint's ``grpc`` metadata block in plugins.yaml. This keeps the
    transport fully proto-agnostic.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._channel_pool: dict[str, GenericGrpcClient] = {}
        self._lease_manager: GrpcChannelLeaseManager | None = None
        self._secure: bool = False
        self._serializer: GrpcSerializerProtocol | None = None
        self._unary_method: str | None = None
        self._stream_method: str | None = None

    @staticmethod
    def _parse_target(url: str) -> str:
        """Extract host:port target from a gRPC URL.

        Args:
            url: URL with grpc:// or grpcs:// scheme.

        Returns:
            gRPC target string (host:port).

        Raises:
            ValueError: If the URL cannot be parsed into a target.
        """
        parsed = urlparse(url)
        target = parsed.netloc or parsed.path
        if not target:
            raise ValueError(f"Cannot parse gRPC target from URL: {url}")
        return target

    def _create_client(self, target: str) -> GenericGrpcClient:
        """Create a new GenericGrpcClient for the given target.

        Args:
            target: gRPC target string (host:port).

        Returns:
            New GenericGrpcClient instance.
        """
        return GenericGrpcClient(
            target=target,
            secure=self._secure,
            timeout=self.model_endpoint.endpoint.timeout,
        )

    @on_init
    async def _init_grpc_client(self) -> None:
        """Initialize gRPC channels based on the connection reuse strategy.

        - POOLED: Pre-creates one channel per unique target from base_urls.
        - NEVER: Stores config only; channels are created per-request.
        - STICKY_USER_SESSIONS: Creates a GrpcChannelLeaseManager.
        """
        base_urls = self.model_endpoint.endpoint.base_urls
        schemes = {urlparse(u).scheme.lower() for u in base_urls}
        if len(schemes) > 1:
            raise ValueError(
                f"All gRPC URLs must use the same scheme, got mixed: {schemes}"
            )
        self._secure = "grpcs" in schemes
        reuse_strategy = self.model_endpoint.endpoint.connection_reuse_strategy

        match reuse_strategy:
            case ConnectionReuseStrategy.POOLED:
                for url in base_urls:
                    target = self._parse_target(url)
                    if target not in self._channel_pool:
                        self._channel_pool[target] = self._create_client(target)
                targets = list(self._channel_pool.keys())
                self.info(
                    f"gRPC channel pool initialized: targets={targets}, "
                    f"secure={self._secure}"
                )

            case ConnectionReuseStrategy.NEVER:
                self.info(
                    f"gRPC NEVER strategy: channels created per-request, "
                    f"secure={self._secure}"
                )

            case ConnectionReuseStrategy.STICKY_USER_SESSIONS:
                self._lease_manager = GrpcChannelLeaseManager()
                self.info(
                    f"gRPC sticky-user-sessions: lease manager initialized, "
                    f"secure={self._secure}"
                )

            case _:
                raise ValueError(
                    f"Unsupported connection reuse strategy for gRPC: {reuse_strategy}"
                )

    @on_init
    async def _init_serializer(self) -> None:
        """Load gRPC serializer from endpoint metadata in plugins.yaml."""
        from aiperf.plugin import plugins

        endpoint_type = str(self.model_endpoint.endpoint.type)
        metadata = plugins.get_endpoint_metadata(endpoint_type)

        if metadata.grpc is None:
            raise ValueError(
                f"Endpoint '{endpoint_type}' does not have gRPC configuration in metadata. "
                f"Add a 'grpc' block with serializer, method, and stream_method to plugins.yaml."
            )

        # Load serializer class from module:Class path
        module_path, _, class_name = metadata.grpc.serializer.rpartition(":")
        if not module_path or not class_name:
            raise ValueError(
                f"Invalid serializer format: {metadata.grpc.serializer!r}. "
                f"Expected 'module.path:ClassName'."
            )

        module = importlib.import_module(module_path)
        serializer_cls = getattr(module, class_name)
        self._serializer = serializer_cls()
        self._unary_method = metadata.grpc.method
        self._stream_method = metadata.grpc.stream_method

        self.debug(lambda: f"gRPC serializer loaded: {metadata.grpc.serializer}")

    @on_stop
    async def _close_grpc_client(self) -> None:
        """Close all gRPC channels and the lease manager."""
        if self._lease_manager:
            lease_manager = self._lease_manager
            self._lease_manager = None
            await lease_manager.close_all()
        clients = list(self._channel_pool.values())
        self._channel_pool.clear()
        for client in clients:
            await client.close()

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """gRPC uses metadata, not HTTP headers. Returns empty dict."""
        return {}

    def get_url(self, request_info: RequestInfo) -> str:
        """Return the gRPC target (host:port) for the request.

        Strips the grpc:// or grpcs:// scheme, returning just the target.
        Supports multi-URL via url_index for load balancing.

        Args:
            request_info: Request context with model endpoint info.

        Returns:
            gRPC target string (host:port).
        """
        base_url = request_info.model_endpoint.endpoint.get_url(request_info.url_index)
        return self._parse_target(base_url)

    def _build_grpc_metadata(
        self, request_info: RequestInfo
    ) -> list[tuple[str, str]] | None:
        """Convert build_headers() dict to gRPC metadata tuples.

        gRPC metadata keys must be lowercase ASCII. HTTP-style headers are
        converted to lowercase for compatibility.

        Args:
            request_info: Request context.

        Returns:
            List of (key, value) tuples or None if empty.
        """
        headers = self.build_headers(request_info)
        if not headers:
            return None
        # gRPC metadata keys must be lowercase
        metadata = [(k.lower(), v) for k, v in headers.items()]
        return metadata or None

    @staticmethod
    async def _maybe_release_sticky_lease(
        reuse_strategy: ConnectionReuseStrategy,
        session_id: str,
        lease_manager: GrpcChannelLeaseManager | None,
    ) -> None:
        """Release sticky lease if using STICKY_USER_SESSIONS strategy.

        Args:
            reuse_strategy: The active connection reuse strategy.
            session_id: The session/correlation ID to release.
            lease_manager: Captured reference to the lease manager.
        """
        if (
            reuse_strategy == ConnectionReuseStrategy.STICKY_USER_SESSIONS
            and lease_manager is not None
        ):
            await lease_manager.release_lease(session_id)

    async def send_request(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any],
        *,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send a gRPC request to the inference server.

        Connection behavior depends on the configured connection_reuse_strategy:
        - POOLED: Uses shared channel pool (one channel per target).
        - NEVER: Creates a new channel for each request, closed after.
        - STICKY_USER_SESSIONS: Reuses channel across conversation turns, closed on final turn.

        Args:
            request_info: Request context and metadata.
            payload: V2 JSON-format dict from the endpoint's format_payload().
            first_token_callback: Optional callback fired on first streaming response.

        Returns:
            RequestRecord with responses, timing, and any errors.
        """
        if self._serializer is None:
            raise NotInitializedError(
                "GrpcTransport not initialized. Call initialize() before send_request()."
            )

        trace_data = GrpcTraceData()
        start_perf_ns = time.perf_counter_ns()
        record = RequestRecord(
            start_perf_ns=start_perf_ns,
            trace_data=trace_data,
        )

        # Resolve client based on connection reuse strategy
        reuse_strategy = self.model_endpoint.endpoint.connection_reuse_strategy
        target = self.get_url(request_info)
        client_owner = False  # True = close client after request (NEVER)

        # Capture lease_manager reference to avoid race with concurrent shutdown
        lease_manager = self._lease_manager

        match reuse_strategy:
            case ConnectionReuseStrategy.POOLED:
                client = self._channel_pool.get(target)
                if client is None:
                    client = self._create_client(target)
                    self._channel_pool[target] = client
            case ConnectionReuseStrategy.NEVER:
                client = self._create_client(target)
                client_owner = True
            case ConnectionReuseStrategy.STICKY_USER_SESSIONS:
                if lease_manager is None:
                    raise NotInitializedError(
                        "GrpcChannelLeaseManager not initialized for "
                        "sticky-user-sessions strategy"
                    )
                client = lease_manager.get_or_create(
                    request_info.x_correlation_id,
                    lambda: self._create_client(target),
                )
            case _:
                raise ValueError(f"Invalid connection reuse strategy: {reuse_strategy}")

        try:
            # Serialize request to bytes
            model_name = request_info.model_endpoint.primary_model_name
            request_bytes = self._serializer.serialize_request(
                payload,
                model_name=model_name,
                request_id=request_info.x_request_id or "",
            )

            # Build gRPC metadata from headers
            grpc_metadata = self._build_grpc_metadata(request_info)

            # Record request metadata and send timing
            trace_data.request_headers = dict(grpc_metadata) if grpc_metadata else None
            trace_data.request_send_start_perf_ns = time.perf_counter_ns()
            trace_data.request_headers_sent_perf_ns = (
                trace_data.request_send_start_perf_ns
            )
            request_size = len(request_bytes)
            trace_data.request_chunks.append(
                (trace_data.request_send_start_perf_ns, request_size)
            )

            is_streaming = self.model_endpoint.endpoint.streaming

            if is_streaming:
                await self._send_streaming_request(
                    client=client,
                    record=record,
                    trace_data=trace_data,
                    request_bytes=request_bytes,
                    grpc_metadata=grpc_metadata,
                    first_token_callback=first_token_callback,
                    cancel_after_ns=request_info.cancel_after_ns,
                )
            else:
                await self._send_unary_request(
                    client=client,
                    record=record,
                    trace_data=trace_data,
                    request_bytes=request_bytes,
                    grpc_metadata=grpc_metadata,
                    cancel_after_ns=request_info.cancel_after_ns,
                )

            # Release sticky lease on final turn, cancellation, or error
            should_release = (
                request_info.is_final_turn
                or record.cancellation_perf_ns is not None
                or record.error is not None
            )
            if should_release:
                await self._maybe_release_sticky_lease(
                    reuse_strategy, request_info.x_correlation_id, lease_manager
                )

        except asyncio.CancelledError:
            record.end_perf_ns = time.perf_counter_ns()
            record.cancellation_perf_ns = record.end_perf_ns
            record.error = ErrorDetails(
                type="RequestCancellationError",
                message="Request cancelled by external signal",
                code=499,
            )
            self.debug(lambda: "gRPC request cancelled by external signal")
            await self._maybe_release_sticky_lease(
                reuse_strategy, request_info.x_correlation_id, lease_manager
            )
            raise
        except grpc.aio.AioRpcError as e:
            record.end_perf_ns = time.perf_counter_ns()
            status_code = e.code().value[0]
            http_status = grpc_status_to_http(status_code)
            trace_data.grpc_status_code = status_code
            trace_data.grpc_status_message = e.details() or str(e.code())
            trace_data.response_status_code = http_status
            trace_data.response_reason = e.code().name
            trace_data.error_timestamp_perf_ns = record.end_perf_ns
            record.status = http_status
            record.error = ErrorDetails(
                type=f"gRPC:{e.code().name}",
                message=e.details() or str(e),
                code=http_status,
            )
            self.error(f"gRPC error: {e.code().name} - {e.details()}")
            await self._maybe_release_sticky_lease(
                reuse_strategy, request_info.x_correlation_id, lease_manager
            )
        except Exception as e:
            record.end_perf_ns = time.perf_counter_ns()
            trace_data.error_timestamp_perf_ns = record.end_perf_ns
            record.error = ErrorDetails.from_exception(e)
            self.error(f"gRPC request failed: {e!r}")
            await self._maybe_release_sticky_lease(
                reuse_strategy, request_info.x_correlation_id, lease_manager
            )
        finally:
            if client_owner:
                await client.close()

        return record

    async def _wait_for_channel_ready(
        self,
        *,
        client: GenericGrpcClient,
        record: RequestRecord,
        trace_data: GrpcTraceData,
    ) -> bool:
        """Wait for gRPC channel to be ready, returning False on failure.

        Separates connection establishment time from request processing time,
        enabling the two-stage cancellation pattern: channel-ready timeout
        produces RequestSendTimeout, while cancel_after_ns produces
        RequestCancellationError.

        Args:
            client: gRPC client whose channel to wait on.
            record: RequestRecord to populate on failure.
            trace_data: Trace data to populate on failure.

        Returns:
            True if channel is ready, False if send timed out (record populated with error).
        """
        send_timeout = min(
            self.model_endpoint.endpoint.timeout or _CHANNEL_READY_TIMEOUT_S,
            _CHANNEL_READY_TIMEOUT_S,
        )
        try:
            await client.wait_for_ready(timeout=send_timeout)
        except (asyncio.TimeoutError, ConnectionError) as e:
            end_ns = time.perf_counter_ns()
            record.end_perf_ns = end_ns
            trace_data.error_timestamp_perf_ns = end_ns
            message = (
                str(e)
                if isinstance(e, ConnectionError)
                else "Timed out waiting for gRPC channel to be ready"
            )
            record.error = ErrorDetails(
                type="RequestSendTimeout",
                message=message,
                code=0,
            )
            self.debug(lambda: f"gRPC request send failed: {message}")
            return False
        return True

    async def _send_unary_request(
        self,
        *,
        client: GenericGrpcClient,
        record: RequestRecord,
        trace_data: GrpcTraceData,
        request_bytes: bytes,
        grpc_metadata: list[tuple[str, str]] | None,
        cancel_after_ns: int | None,
    ) -> None:
        """Send a unary ModelInfer RPC, optionally with two-stage cancellation.

        Two-stage cancellation (when cancel_after_ns is set):
          1. Wait for channel ready (RequestSendTimeout on failure)
          2. Send RPC with cancel timer (RequestCancellationError on timeout)

        This mirrors aiohttp's two-stage approach: the cancel timer only measures
        server processing time, not connection establishment.

        Args:
            client: gRPC client to send the request with.
            record: RequestRecord to populate.
            trace_data: Trace data to populate.
            request_bytes: Serialized request bytes.
            grpc_metadata: gRPC metadata tuples.
            cancel_after_ns: Cancel after this many ns, or None for no cancellation.
        """
        assert self._serializer is not None  # noqa: S101
        assert self._unary_method is not None  # noqa: S101

        if cancel_after_ns is None:
            result = await client.unary(
                self._unary_method, request_bytes, metadata=grpc_metadata
            )
        else:
            # Stage 1: Wait for channel ready
            if not await self._wait_for_channel_ready(
                client=client, record=record, trace_data=trace_data
            ):
                return

            # Stage 2: Send with cancel timer (channel ready, request goes out immediately)
            timeout_s = cancel_after_ns / NANOS_PER_SECOND
            task = asyncio.create_task(
                client.unary(self._unary_method, request_bytes, metadata=grpc_metadata)
            )
            try:
                result = await asyncio.wait_for(task, timeout=timeout_s)
            except asyncio.TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                end_ns = time.perf_counter_ns()
                record.end_perf_ns = end_ns
                record.cancellation_perf_ns = end_ns
                record.error = ErrorDetails(
                    type="RequestCancellationError",
                    message=f"Request cancelled {timeout_s:.3f}s after being sent",
                    code=499,
                )
                self.debug(
                    lambda: f"gRPC request cancelled {timeout_s:.3f}s after being sent"
                )
                return

        perf_ns = time.perf_counter_ns()
        trace_data.response_receive_start_perf_ns = perf_ns
        trace_data.response_receive_end_perf_ns = perf_ns
        trace_data.response_headers_received_perf_ns = perf_ns
        response_dict, response_size = self._serializer.deserialize_response(
            result.data
        )
        trace_data.response_chunks.append((perf_ns, response_size))
        trace_data.grpc_status_code = 0  # OK
        trace_data.response_status_code = 200
        trace_data.response_reason = "OK"
        trace_data.response_headers = _metadata_to_dict(result.trailing_metadata)
        record.status = 200

        json_str = orjson.dumps(response_dict).decode("utf-8")
        record.responses.append(
            TextResponse(
                perf_ns=perf_ns,
                text=json_str,
                content_type="application/json",
            )
        )
        record.end_perf_ns = perf_ns

    async def _send_streaming_request(
        self,
        *,
        client: GenericGrpcClient,
        record: RequestRecord,
        trace_data: GrpcTraceData,
        request_bytes: bytes,
        grpc_metadata: list[tuple[str, str]] | None,
        first_token_callback: FirstTokenCallback | None,
        cancel_after_ns: int | None,
    ) -> None:
        """Send a streaming ModelStreamInfer RPC, collecting responses.

        Two-stage cancellation (when cancel_after_ns is set):
          1. Wait for channel ready (RequestSendTimeout on failure)
          2. Consume stream with cancel timer (RequestCancellationError on timeout)

        Args:
            client: gRPC client to send the request with.
            record: RequestRecord to populate.
            trace_data: Trace data to populate.
            request_bytes: Serialized request bytes.
            grpc_metadata: gRPC metadata tuples.
            first_token_callback: Optional callback on first non-error response.
            cancel_after_ns: Cancel after this many ns, or None for no cancellation.
        """
        assert self._serializer is not None  # noqa: S101
        assert self._stream_method is not None  # noqa: S101

        first_token_acquired = False

        async def _consume_stream(stream_call: GrpcStreamCall) -> None:
            nonlocal first_token_acquired

            async for chunk_bytes in stream_call:
                chunk = self._serializer.deserialize_stream_response(chunk_bytes)

                if chunk.error_message:
                    error_perf_ns = time.perf_counter_ns()
                    trace_data.error_timestamp_perf_ns = error_perf_ns
                    record.end_perf_ns = error_perf_ns
                    record.error = ErrorDetails(
                        type="gRPC:STREAM_ERROR",
                        message=chunk.error_message,
                        code=500,
                    )
                    break

                perf_ns = time.perf_counter_ns()
                trace_data.response_chunks.append((perf_ns, chunk.response_size))

                if trace_data.response_receive_start_perf_ns is None:
                    trace_data.response_receive_start_perf_ns = perf_ns
                    trace_data.response_headers_received_perf_ns = perf_ns

                json_str = orjson.dumps(chunk.response_dict).decode("utf-8")
                text_response = TextResponse(
                    perf_ns=perf_ns,
                    text=json_str,
                    content_type="application/json",
                )
                record.responses.append(text_response)

                if first_token_callback and not first_token_acquired:
                    ttft_ns = perf_ns - record.start_perf_ns
                    first_token_acquired = await first_token_callback(
                        ttft_ns, text_response
                    )

            # Stream completed successfully (or with inline error)
            end_ns = time.perf_counter_ns()
            trace_data.response_receive_end_perf_ns = end_ns
            if record.end_perf_ns is None:
                record.end_perf_ns = end_ns
            if record.error is None:
                trace_data.grpc_status_code = 0  # OK
                trace_data.response_status_code = 200
                trace_data.response_reason = "OK"
                record.status = 200
                # Best-effort trailing metadata capture
                with contextlib.suppress(Exception):
                    trailing = await stream_call.trailing_metadata()
                    trace_data.response_headers = _metadata_to_dict(trailing)

        if cancel_after_ns is None:
            stream_call = client.server_stream(
                self._stream_method, request_bytes, metadata=grpc_metadata
            )
            await _consume_stream(stream_call)
        else:
            # Stage 1: Wait for channel ready
            if not await self._wait_for_channel_ready(
                client=client, record=record, trace_data=trace_data
            ):
                return

            # Stage 2: Create stream and consume with cancel timer
            stream_call = client.server_stream(
                self._stream_method, request_bytes, metadata=grpc_metadata
            )
            timeout_s = cancel_after_ns / NANOS_PER_SECOND
            task = asyncio.create_task(_consume_stream(stream_call))
            try:
                await asyncio.wait_for(task, timeout=timeout_s)
            except asyncio.TimeoutError:
                task.cancel()
                stream_call.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                end_ns = time.perf_counter_ns()
                record.end_perf_ns = end_ns
                record.cancellation_perf_ns = end_ns
                record.error = ErrorDetails(
                    type="RequestCancellationError",
                    message=f"Request cancelled {timeout_s:.3f}s after being sent",
                    code=499,
                )
                self.debug(
                    lambda: f"gRPC streaming request cancelled {timeout_s:.3f}s after being sent"
                )
